# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################
import sys
sys.path.append("/root/data1/zbc/erfnet_pytorch/ProtoSeg") 
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributed as dist
from torchsummary import summary

from lib.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead
from lib.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat



class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 64, 64, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

#ERFNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

        self.input_channels = 64

        
        self.gamma = 0.999
        self.num_prototype = 10
        self.clip_channel = 768
        self.num_classes = num_classes
        self.use_prototype = True
        self.update_prototype = True
        self.pretrain_prototype = False

        self.prototypes = nn.Parameter(torch.zeros(num_classes, self.num_prototype, self.input_channels),
                                       requires_grad=True)

        self.mlp = nn.ModuleList([
                                  nn.Linear(self.clip_channel,128),
                                  nn.Linear(128,self.input_channels)
                                  ])
        self.proj_head = ProjectionHead(self.input_channels, self.input_channels)
        self.feat_norm = nn.LayerNorm(self.input_channels)
        self.mask_norm = nn.LayerNorm(num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, input,  clip_proto):

        input = self.encoder(input)    #predict=False by default
        feats = self.decoder(input)
        
        c = self.proj_head(feats)
        #print("------------c size ------------{}".format(c.size()))
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)
        clip_proto = self.mlp[0](clip_proto)
        clip_proto = self.mlp[1](clip_proto)
        #print(f" -----------------------------------  clip proto  -----------------------------------{clip_proto.shape}")
        #print(f" -----------------------------------  clip proto  -----datatype----------------------{clip_proto.dtype}")
        self.prototypes.data = clip_proto
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        #print(f" -----------------------------------  prototypes  -----------------------------------{self.prototypes.shape}")
        #print(f" -----------------------------------  prototypes  -----datatype----------------------{self.prototypes.dtype}")
        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
        #print("------------_c size ------------{}".format(_c.size()))
        #print("------------mask size ------------{}".format(masks.size()))
        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])
        #print("------------out_seg size ------------{}".format(out_seg.size()))
        '''
        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            # print("hello ----------------------")
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            # print(f" prototype require grad ----------------{self.prototypes.requires_grad}")
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}
        '''
        # print("hi --------------------------")
        
        # print(f" prototype require grad ----------------{self.prototypes.requires_grad}")
        return out_seg


if __name__ == '__main__':
    net = Net(20)
    net = net.cuda()
    x= torch.rand(2,3,512,512).cuda()
    gt = torch.randint(0,1,(2,1,512,512)).cuda()
    clip_proto = torch.load("pre_proto.pt")
    y = net(x,gt,clip_proto,False)
    print(f" y out ---------------{y.shape}--------------------")
    '''
    print("y seg size ----------------------------------{}".format(y['seg'].size()))
    print("y seg type ----------------------------------{}".format(type(y['seg'])))
    print("y logits size ----------------------------------{}".format(y['logits'].size()))
    print("y logits type ----------------------------------{}".format(type(y['logits'])))
    print("y target size ----------------------------------{}".format(y['target'].size()))
    print("y target type ----------------------------------{}".format(type(y['target'])))
    '''
   # summary(net,[ (3,512,512),(1,512,512)])

