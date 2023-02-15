cd /root/data1/zbc/erfnet_pytorch/train
python -u main_proto_clip.py --savedir /erfnet_proto_clip \
                             --state /root/data1/zbc/erfnet_pytorch/save/erfnet_proto/model_best.pth  \
                             --decoder \
                             --num-epochs 150  \
                             --epochs-save 1  \
                             --iouTrain \
                             --model erfnet_proto_clip  \
                             --batch-size 4  \
                             --resume