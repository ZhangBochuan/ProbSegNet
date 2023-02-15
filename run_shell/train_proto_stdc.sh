cd /root/data1/zbc/erfnet_pytorch/train
python -u main_proto_stdc.py --savedir /erfnet_proto_stdc \
                             --state /root/data1/zbc/erfnet_pytorch/save/erfnet_proto/model-342.pth \
                             --decoder \
                             --num-epochs 400  \
                             --epochs-save 1  \
                             --iouTrain \
                             --model erfnet_proto_stdc  \
                             --batch-size 3  \
                             --resume
