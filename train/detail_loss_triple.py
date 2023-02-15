
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import json

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label.long())   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def get_boundary(gtmasks):

    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets


class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks, boundary_feats =None):

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        #print("boundary targets pyramid shape ----------------{}-------------".format(boudary_targets_pyramid.shape))
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        if False:
            save_img = boudary_targets_pyramid *255
            save_img = save_img.int()
            save_img = save_img.squeeze(0)
            save_img = save_img.squeeze(0)
            print("boundary targets pyramid shape ------LOSS GT---------------------{}".format(save_img.shape))
            print("boundary targets pyramid shape ------LOSS GT---------------------{}".format(save_img.dtype))
            print("boundary targets pyramid shape ------LOSS GT---------------------{}".format(save_img.max()))
            print("boundary targets pyramid shape ------LOSS GT---------------------{}".format(save_img.min()))
            #print("boundary logits shape ------LOSS Logits---------------------{}".format(boundary_logits.shape))
            cv2.imwrite('/root/data1/zbc/erfnet_pytorch/train/boundary_1.png',save_img.cpu().numpy())
        
        output_pos_mask = boundary_logits > 0.90
        output_mid_mask = boundary_logits <= 0.9
        output_neg_mask = boundary_logits <= 0.5

        #print("output pos mask shape ----------------{}---------".format(output_pos_mask.shape))
        #print("output mid mask shape ----------------{}---------".format(output_mid_mask.shape))
        #print("output neg mask shape ----------------{}---------".format(output_neg_mask.shape))
        

        target_mask = (boudary_targets_pyramid == 1)
        #print("target mask dtype FIRST!!!!!!!----------------{}---------".format(target_mask.dtype))
        target_mask= target_mask * boudary_targets_pyramid
        target_visualize = target_mask * 255

        target_mask_bkg = (boudary_targets_pyramid != 1)
        board = torch.ones(target_mask_bkg.size()).cuda()
        target_mask_bkg = board * target_mask_bkg

        output_tp = output_pos_mask * target_mask
        output_mid = output_mid_mask * target_mask
        output_tn = output_neg_mask * target_mask_bkg

        boundary_feats = F.interpolate(boundary_feats, boundary_targets.shape[2:], mode='nearest')

        feature_tp =  boundary_feats * output_tp
        feature_mid = boundary_feats * output_mid
        feature_tn =  boundary_feats * output_tn

        tp_count = int(len(feature_tp[feature_tp > 0]) / feature_tp.shape[1])
        mid_count = int(len(feature_mid[feature_mid > 0]) / feature_mid.shape[1])
        tn_count = int(len(feature_tn[feature_tn > 0]) / feature_tn.shape[1])

        #print("feat tp  shape ------------------{}".format(feature_tp.shape))
        #print("feat mid  shape ------------------{}".format(feature_mid.shape))
        #print("feat tn  shape ------------------{}".format(feature_tn.shape))
        
        #print("tp count ------------------{}".format(tp_count))
        #print("mid count  ------------------{}".format(mid_count))
        #print("tn count  ------------------{}".format(tn_count))
        smooth_count = 5000

        feature_tp = torch.sum(feature_tp, dim=-1)
        feature_tp = torch.sum(feature_tp, dim=-1)
        feature_tp /= (tp_count + smooth_count)

        feature_mid = torch.sum(feature_mid, dim=-1)
        feature_mid = torch.sum(feature_mid, dim=-1)
        feature_mid /= (mid_count + smooth_count)

        feature_tn = torch.sum(feature_tn, dim=-1)
        feature_tn = torch.sum(feature_tn, dim=-1)
        feature_tn /= (tn_count + smooth_count)

        #print("feat tp  ---------shape -----------{}".format(feature_tp.shape))
        # print("feat tp  ---------shape -----------{}".format(feature_tp))
        # print("feat tp  ---------shape -----------{}".format(feature[:,0,0]))
        #print("feat tp  ---------shape -----------{}".format(feature_tp.dtype))
        #print("feat tp  ---------shape -----------{}".format(feature_tp.max()))
        #print("feat tp  ---------shape -----------{}".format(feature_tp.min()))

        #print("target mask shape ----------------{}---------".format(target_mask.shape))
        #print("target mask dtype ----------------{}---------".format(target_mask.dtype))
        #print("target mask max ----------------{}---------".format(target_mask.max()))
        #print("target mask min ----------------{}---------".format(target_mask.min()))
        #print("target visualize shape ----------------{}---------".format(target_visualize.shape))
        #print("target visualize dtype ----------------{}---------".format(target_visualize.dtype))
        #print("target mask bkg shape ----------------{}---------".format(target_mask_bkg.shape))
        #print("target mask bkg dtype ----------------{}---------".format(target_mask_bkg.dtype))
        triplet_crition = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(),margin=0.2)
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        triplet_loss = triplet_crition(feature_mid,feature_tp,feature_tn)
        return bce_loss,  dice_loss, triplet_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params

if __name__ == '__main__':
    #torch.manual_seed(15)
    #with open('../cityscapes_info.json', 'r') as fr:
    #        labels_info = json.load(fr)
    #lb_map = {el['id']: el['trainId'] for el in labels_info}

    img_path = '/root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/val/lindau/lindau_000007_000019_gtFine_labelTrainIds.png'
    img = cv2.imread(img_path, 0)
    #print("----------------------------img size {}-----------------------------".format(img.shape))
    #label = np.zeros(img.shape, np.uint8)
    #for k, v in lb_map.items():
    #    label[img == k] = v

    img_tensor = torch.from_numpy(img).cuda()
    print("----------------------------img size {}-----------------------------".format(img_tensor.dtype))
    print("----------------------------img size {}-----------------------------".format(img_tensor.min()))
    print("----------------------------img size {}-----------------------------".format(img_tensor.max()))
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)
    print("----------------------------img size 1 {}-----------------------------".format(img_tensor.dtype))
    print("----------------------------img size 1 {}-----------------------------".format(img_tensor.min()))
    print("----------------------------img size 1 {}-----------------------------".format(img_tensor.max()))

    detailAggregateLoss = DetailAggregateLoss()
    for param in detailAggregateLoss.parameters():
        print(param)
    print("img tensor ------------------ 1 ------------------ {}".format(torch.unsqueeze(img_tensor, 0).shape))
    print("img tensor ------------------ 2 ------------------ {}".format(img_tensor.shape))
    feature = torch.rand(1, 10, 1024, 2048).cuda()
    bce_loss,  dice_loss,triplet_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor,feature)
    print(bce_loss,  dice_loss,triplet_loss)