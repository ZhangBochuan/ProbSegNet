cd /root/data1/zbc/erfnet_pytorch/eval
python -u eval_cityscapes_color.py  \
                             --datadir /root/data1/zbc/erfnet_pytorch/dataset/cityscapes/ \
                             --loadDir /root/data1/zbc/erfnet_pytorch/save/erfnet_proto_stdc_triplet_clip_3/ \
                             --loadWeights model_best.pth \
                             --saveDir save_color_erfnet_proto_stdc_triplet_clip  \
                             --subset val \
                             --num-workers 4  \
                             --batch-size 1  \