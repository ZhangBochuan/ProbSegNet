import clip
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary



if __name__ == '__main__':
    labels = {
        'road':0 ,
        'sidewalk':1 ,
        'building':2 ,
        'wall':3 ,
        'fence':4 ,
        'pole':5 ,
        'traffic light':6 ,
        'traffic sign':7 ,
        'vegetation':8 ,
        'terrain':9 ,
        'sky':10 ,
        'person':11 ,
        'rider':12 ,
        'car':13 ,
        'truck':14 ,
        'bus':15 ,
        'train':16 ,
        'motorcycle':17 ,
        'bicycle':18 ,
        'license plate':19
    }
    templates = [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.']


    

    x = torch.load("/root/data1/zbc/erfnet_pytorch/train/pre_proto.pt")

    print(f"shape x {x.shape}")
    print(f"x 's device {x.device}")



