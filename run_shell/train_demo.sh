cd /root/data1/zbc/erfnet_pytorch/train
python -u main.py --savedir /erfnet_ori \
                  --state /root/data1/zbc/erfnet_pytorch/trained_models/erfnet_pretrained.pth \
                  --decoder \
                  --num-epochs 400  \
                  --epochs-save 1  \
                  --iouTrain \