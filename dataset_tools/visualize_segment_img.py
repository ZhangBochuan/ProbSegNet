import os
import cv2 as cv


if __name__ == '__main__':
    # /root/data1/zbc/erfnet_pytorch/eval/save_color_origin_pretrain/val/munster/munster_000173_000019_leftImg8bit.png
    data_dir = "/root/data1/zbc/erfnet_pytorch/eval/save_color_origin_pretrain/val"
    data_list = []
    for root,dirs,files in os.walk(data_dir,topdown=True):
        for file in files:
            file_path = os.path.join(root,file)
           # print(file_path)
            data_list.append(file_path)
       # print('root={}'.format(root))
    print(len(data_list))   

    dest_dir = "/root/data1/zbc/erfnet_pytorch/dataset_tools/segment_cmp"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    gt_data_list=[]
    for index, file_path in enumerate(data_list):

        gt_file_path = file_path.replace("/root/data1/zbc/erfnet_pytorch/eval/save_color_origin_pretrain/","/root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/")
        gt_file_path = gt_file_path.replace("leftImg8bit","gtFine_color")
        if not os.path.exists(gt_file_path):
            print("FALSE!!!!!!!")
        gt_data_list.append(gt_file_path)

        img_file_path = file_path.replace("/root/data1/zbc/erfnet_pytorch/eval/save_color_origin_pretrain/","/root/data1/zbc/erfnet_pytorch/dataset/cityscapes/leftImg8bit/")
        predict2_file_path = file_path.replace("save_color_origin_pretrain","save_color_erfnet_proto_stdc_triplet_clip")
        if not os.path.exists(img_file_path):
            print("FALSE 2222!!!!!!!")
        predict = cv.imread(file_path)
        predict2 = cv.imread(predict2_file_path)
        gt = cv.imread(gt_file_path)
        img = cv.imread(img_file_path)
        gt = cv.resize(gt,(predict.shape[1],predict.shape[0]))
        img = cv.resize(img,(predict.shape[1],predict.shape[0]))
        print(f"-----------1------------{predict.shape}")
        print(f"-----------1.1------------{predict2.shape}")
        print(f"-----------2------------{gt.shape}")
        print(f"-----------3------------{img.shape}")
        dest = cv.hconcat([img,gt,predict,predict2])
        filename =file_path.split("/")[-1]
        cv.imwrite(os.path.join(dest_dir,filename).replace(".png",".jpg"),dest)

    print(gt_data_list[0])
    # /root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/val/val/frankfurt/frankfurt_000000_000294_gtFine_color.png
    # /root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/val/munster/munster_000000_000019_gtFine_color.png
