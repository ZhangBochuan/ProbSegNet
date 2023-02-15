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




class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x_feat = self.conv(x)
        x_out = self.conv_out(x_feat)
        return x_out, x_feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

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

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.output_conv = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):


        # for layer in self.layers:
        #     output = layer(output)
        feat_res8 = input
        #print(f"----------- input  -{input.shape}---------------")  # 64 * 64
        feat_res4 = self.layers[0](input)
        #print(f"-----------  index 0 -{feat_res4.shape}---------------")  # 128
        input = self.layers[1](feat_res4)
        #print(f"-----------  index 1 -{input.shape}---------------")  # 128
        input = self.layers[2](input)
        #print(f"-----------  index 2 -{input.shape}---------------")  # 128
        feat_res2 = self.layers[3](input)
        #print(f"-----------  index 3 -{feat_res2.shape}---------------")  # 256
        input = self.layers[4](feat_res2)
        #print(f"-----------  index 4 -{input.shape}---------------")  # 256
        input = self.layers[5](input)
        #print(f"-----------  index 5 -{input.shape}---------------")  # 256

        output = self.output_conv(input)
        #print(f"----------- output  -{output.shape}---------------")  # 512

        return output, feat_res2, feat_res4, feat_res8


# ERFNet

class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

        self.input_channels = 64

        self.gamma = 0.999
        self.num_prototype = 10
        self.num_classes = num_classes
        self.use_prototype = True
        self.update_prototype = True
        self.pretrain_prototype = False

        self.prototypes = nn.Parameter(torch.zeros(num_classes, self.num_prototype, self.input_channels),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(self.input_channels, self.input_channels)
        self.feat_norm = nn.LayerNorm(self.input_channels)
        self.mask_norm = nn.LayerNorm(num_classes)
        
        self.conv_out_sp8 = BiSeNetOutput(128, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(64, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(64, 64, 1)

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

    def forward(self, input, gt_semantic_seg, pretrain_prototype=False):

        input = self.encoder(input)  # predict=False by default
        feats, feat_res2, feat_res4, feat_res8= self.decoder(input)
        feat_out_sp2, feat_sp2 = self.conv_out_sp2(feat_res2)
        feat_out_sp4, feat_sp4 = self.conv_out_sp4(feat_res4)
        feat_out_sp8, feat_sp8 = self.conv_out_sp8(feat_res8)
        c = self.proj_head(feats)
        #print("------------c size ------------{}".format(c.shape))
        _c = rearrange(c, 'b c h w -> (b h w) c')
        #print("------------ _c size ------------{}".format(_c.shape))
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
        # print("------------_c size ------------{}".format(_c.size()))
        #print("------------mask size ------------{}".format(masks.shape))
        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])
        # print("------------out_seg size ------------{}".format(out_seg.size()))

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
           # print("----------------------COME IN TO PROTOTYPE LEARNING!!!!----------------------")
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            
            return out_seg, contrast_logits,contrast_target,feat_out_sp2 ,feat_out_sp4, feat_out_sp8, feat_sp2, feat_sp4, feat_sp8

        return out_seg,feat_out_sp2 ,feat_out_sp4, feat_out_sp8, feat_sp2, feat_sp4, feat_sp8
'''


class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        self.conv_out_sp8 = BiSeNetOutput(128, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(64, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(64, 64, 1)

    def forward(self, input, only_encode=False):
        # if only_encode:
        #     return self.encoder.forward(input, predict=True)
        # else:
        #     output = self.encoder(input)    #predict=False by default
        #     return self.decoder.forward(output)
        input = self.encoder(input)  # predict=False by default
        output, feat_res2, feat_res4, feat_res8 = self.decoder(input)
        feat_out_sp2, feat_sp2 = self.conv_out_sp2(feat_res2)
        feat_out_sp4, feat_sp4 = self.conv_out_sp4(feat_res4)
        feat_out_sp8, feat_sp8 = self.conv_out_sp8(feat_res8)

        return output, feat_out_sp2 ,feat_out_sp4, feat_out_sp8, feat_sp2, feat_sp4, feat_sp8
'''


if __name__ == '__main__':
    from detail_loss import DetailAggregateLoss
    import cv2
    import numpy as np

    
    net = Net(20)
    net = net.cuda()

    # gt_path = "/root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"
    # gt = cv2.imread(gt_path, 0)
    #
    # gt_tensor = torch.from_numpy(gt).cuda()
    # gt_tensor = torch.unsqueeze(gt_tensor, 0).type(torch.cuda.FloatTensor)

    x= torch.rand(2,3,128,128).cuda()
    gt = torch.randint(0,1,(2,1,128,128)).cuda()
    boundary = torch.rand(2,128,128).cuda()
    y,sp2,sp4,sp8,f_sp2,f_sp4,f_sp8 = net(x,gt,True)


    print("x shape {}".format(x.shape))
    print("y shape {}".format(y.shape))
    print("sp2 shape {}".format(sp2.shape))
    print("sp4 shape {}".format(sp4.shape))
    print("sp8 shape {}".format(sp8.shape))

    print("f sp2 shape {}".format(f_sp2.shape))
    print("f sp4 shape {}".format(f_sp4.shape))
    print("f sp8 shape {}".format(f_sp8.shape))

    detailAggregateLoss = DetailAggregateLoss()
    bce2, dice2 = detailAggregateLoss(sp2, boundary)
    bce4, dice4 = detailAggregateLoss(sp4, boundary)
    bce8, dice8 = detailAggregateLoss(sp8, boundary)

    print("loss 2  {}, {}".format(bce2,dice2))
    print("loss 4  {}, {}".format(bce4,dice4))
    print("loss 8  {}, {}".format(bce8,dice8))



