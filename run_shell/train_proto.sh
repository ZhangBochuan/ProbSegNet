cd /root/data1/zbc/erfnet_pytorch/train
python -u main_proto.py --savedir /erfnet_proto \
                  --state /root/data1/zbc/erfnet_pytorch/trained_models/erfnet_pretrained.pth \
                  --decoder \
                  --num-epochs 400  \
                  --epochs-save 1  \
                  --iouTrain \
                  --model erfnet_proto  \
                  --batch-size 3  \
                  --resume