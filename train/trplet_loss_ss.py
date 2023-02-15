if __name__ == '__main__':
    from detail_loss import DetailAggregateLoss
    import cv2
    import numpy as np
    import torch 

    '''


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
    y,sp2,sp4,sp8 = net(x)


    print("x shape {}".format(x.shape))
    print("y shape {}".format(y.shape))
    print("sp2 shape {}".format(sp2.shape))
    print("sp4 shape {}".format(sp4.shape))
    print("sp8 shape {}".format(sp8.shape))

    detailAggregateLoss = DetailAggregateLoss()
    bce2, dice2 = detailAggregateLoss(sp2, boundary)
    bce4, dice4 = detailAggregateLoss(sp4, boundary)
    bce8, dice8 = detailAggregateLoss(sp8, boundary)

    print("loss 2  {}, {}".format(bce2,dice2))
    print("loss 4  {}, {}".format(bce4,dice4))
    print("loss 8  {}, {}".format(bce8,dice8))
    '''
    input = torch.rand(3, 1024, 2048)
    target = cv2.imread(
        '/root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/train/aachen/aachen_000003_000019_gtFine_labelTrainIds.png',
        0)
    output = torch.rand(1024, 2048)
    feature = torch.rand(10, 1024, 2048)

    output_pos_mask = output > 0.90
    # output_mid_mask = (output<=0.90) & (output>0.5)
    output_mid_mask = output <= 0.9
    output_neg_mask = output <= 0.5

    target = torch.from_numpy(target)
    target_mask = (target == 1)
    target *= target_mask
    target_visualize = target * 255

    target_mask_bkg = (target != 1)
    board = torch.ones(target_mask_bkg.size())
    target_bkg = board * target_mask_bkg

    # print("board   size  {} ".format(board.size()))
    # print("bkg ---------shape -----------{}".format(target_mask_bkg.size()))
    # print("bkg ---------shape -----------{}".format(target_mask_bkg.dtype))
    # print("bkg ---------shape -----------{}".format(target_mask_bkg.max()))
    # print("bkg ---------shape -----------{}".format(target_mask_bkg.min()))
    print("TARGET BKG PNG ----------------------------{}".format(target_bkg.shape))
    cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/target_bkg.png", target_bkg.numpy()*255)

    output_tp = output_pos_mask * target_mask
    output_mid = output_mid_mask * target_mask
    output_tn = output_neg_mask * target_bkg

    feature_tp = feature * output_tp
    feature_mid = feature * output_mid
    feature_tn = feature * output_tn

    # tp_count = feature_tp[feature_tp>0]
    # print("feat tp  ---------shape -----------{}".format(tp_count.size()))
    #
    # print("feat tp  ---------shape -----------{}".format(tp_count.size()))
    # print("feat tp  ---------shape -----------{}".format(tp_count.dtype))
    # print("feat tp  ---------shape -----------{}".format(tp_count.max()))
    # print("feat tp  ---------shape -----------{}".format(tp_count.min()))

    # print(" pos ---------shape -----------{}".format(output_pos_mask.size()))
    # print(" pos ---------shape -----------{}".format(output_pos_mask.dtype))
    # print(" pos ---------shape -----------{}".format(output_pos_mask.max()))
    # print(" pos ---------shape -----------{}".format(output_pos_mask.min()))
    #
    # print("target pos ---------shape -----------{}".format(target.size()))
    # print("target pos ---------shape -----------{}".format(target.dtype))
    # print("target pos ---------shape -----------{}".format(target.max()))
    # print("target pos ---------shape -----------{}".format(target.min()))
    #
    # print("out tp  ---------shape -----------{}".format(output_tp.size()))
    # print("out tp  ---------shape -----------{}".format(output_tp.dtype))
    # print("out tp  ---------shape -----------{}".format(output_tp.max()))
    # print("out tp  ---------shape -----------{}".format(output_tp.min()))
    #
    # print("feat tp  ---------shape -----------{}".format(feature_tp.size()))
    # print("feat tp  ---------shape -----------{}".format(feature_tp.dtype))
    # print("feat tp  ---------shape -----------{}".format(feature_tp.max()))
    # print("feat tp  ---------shape -----------{}".format(feature_tp.min()))

    # feature_tp = feature_tp[0:3,:,:].permute(1, 2, 0)
    # feature_mid = feature_mid[0:3, :, :].permute(1, 2, 0)
    # feature_tn = feature_tn[0:3, :, :].permute(1, 2, 0)
    tp_count = len(feature_tp[feature_tp > 0])
    mid_count = len(feature_mid[feature_mid > 0])
    tn_count = len(feature_tn[feature_tn > 0])
    print(tp_count)
    print(mid_count)
    print(tn_count)
    feature_tp = torch.sum(feature_tp, dim=1)
    feature_tp = torch.sum(feature_tp, dim=1)
    feature_tp /= tp_count

    print("feat tp  ---------shape -----------{}".format(feature_tp.size()))
    # print("feat tp  ---------shape -----------{}".format(feature_tp))
    # print("feat tp  ---------shape -----------{}".format(feature[:,0,0]))
    print("feat tp  ---------shape -----------{}".format(feature_tp.dtype))
    print("feat tp  ---------shape -----------{}".format(feature_tp.max()))
    print("feat tp  ---------shape -----------{}".format(feature_tp.min()))
    # cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/target.png", target_visualize.numpy())
    # cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/output_tp.png", output_tp.numpy()*255)
    # cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/feat_tp.png", feature_tp.numpy() * 255)
    # cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/feat_mid.png", feature_mid.numpy() * 255)
    # cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/feat_tn.png", feature_tn.numpy() * 255)

    '''

    input_save = (input * 255).numpy().transpose(1,2,0)
    cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/input.png", input_save)
    cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/target.png", target)

    img_tensor = torch.from_numpy(image)
    img_tensor[img_tensor==1]=200
    img_tensor[img_tensor!=200]=0
    img_tensor[img_tensor==200]=255
    mask = img_tensor
    print("mask ---------shape -----------{}".format(mask.size()))
    print("mask ---------shape -----------{}".format(mask.dtype))
    print("mask ---------shape -----------{}".format(mask.max()))
    print("mask ---------shape -----------{}".format(mask.min()))
    cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/mask.png", mask.numpy().reshape(1024,2048,3))

    mask[mask==255]=1

    x_mask = (x*255).permute(1,2,0) * mask
    cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/x_mask.png", x_mask.numpy())

    x_pos_mask = x >0.95
    print("x pos mask---------shape -----------{}".format(x_pos_mask.size()))
    print("x pos mask---------shape -----------{}".format(x_pos_mask.dtype))
    print("x pos mask---------shape -----------{}".format(x_pos_mask.max()))
    print("x pos mask---------shape -----------{}".format(x_pos_mask.min()))

    mask[mask == 1] = 255
    img_pos_mask = mask * x_pos_mask.permute(1,2,0)
    print("img pos mask ---------shape -----------{}".format(img_pos_mask.size()))
    print("img pos mask ---------shape -----------{}".format(img_pos_mask.dtype))
    print("img pos mask ---------shape -----------{}".format(img_pos_mask.max()))
    print("img pos mask ---------shape -----------{}".format(img_pos_mask.min()))


    cv2.imwrite("/root/data1/zbc/erfnet_pytorch/train/triplet_loss_visualize/x_pos_mask.png", img_pos_mask.numpy())
    #x[img_tensor!=1] = 0
  #  print("img_tensor shape   {}".format(img_tensor.size()))



   # cv2.imwrite("./test.png",x)


    #gt = torch.randint(0,1,(2,1,128,128)).cuda()
'''