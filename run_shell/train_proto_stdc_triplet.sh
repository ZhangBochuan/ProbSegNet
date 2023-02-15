cd /root/data1/zbc/erfnet_pytorch/train
python -u main_proto_stdc_triplet.py --savedir /erfnet_proto_stdc_triplet \
                             --state /root/data1/zbc/erfnet_pytorch/save/erfnet_proto_stdc/model_best.pth \
                             --decoder \
                             --num-epochs 150  \
                             --epochs-save 1  \
                             --iouTrain \
                             --model erfnet_proto_stdc  \
                             --batch-size 3  \