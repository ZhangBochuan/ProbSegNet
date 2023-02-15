import os
import clip
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary

if __name__ == '__main__':
    state_dict = torch.load("/root/data1/zbc/erfnet_pytorch/save/erfnet_proto_clip/model-030.pth")
    for name, param in state_dict.items():
        if "prototypes" in name:
            print("---------------here i come--------------------")
            print(param.shape)
            torch.save(param,"erfnet_clip_30.pt")
            break
