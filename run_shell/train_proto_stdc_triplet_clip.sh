cd /root/data1/zbc/erfnet_pytorch/train
python -u main_proto_stdc_triplet_clip.py --savedir /erfnet_proto_stdc_triplet_clip \
                             --state /root/data1/zbc/erfnet_pytorch/save/erfnet_proto_stdc_triplet/model_best.pth \
                             --clip  /root/data1/zbc/erfnet_pytorch/save/erfnet_proto_clip/model_best.pth  \
                             --decoder \
                             --num-epochs 150  \
                             --epochs-save 1  \
                             --iouTrain \
                             --model erfnet_proto_stdc  \
                             --batch-size 3  \
                             --resume 