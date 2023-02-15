import os
import clip
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, in_channel, out_channel):  #use encoder to pass pretrained encoder
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layers = nn.ModuleList([
                                     nn.Linear(self.in_channel,64),
                                     nn.Linear(64,self.out_channel)
                                    ])
        
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

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


    # Load the model
    #print(f" available  ------------- {clip.available_models()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14', device)



    text_inputs = torch.cat([clip.tokenize(f"a photo of a road",context_length=77) ]).to(device)

    print(f"text inputs ------------------------------------------------{text_inputs.shape}")
    x= torch.zeros(20,10,768).cuda()
    # Calculate features
    with torch.no_grad():
        '''
        text_features = model.encode_text(text_inputs)
        print(type(text_features))
        print(text_features.shape)
        x[0,0,:]= text_features
        print(f"x 0 0 : 's value   {x[0,0,:]}")
        '''

        for i, key in enumerate( labels.keys()):
            for j,template in  enumerate( templates):
                print(f"  {i}   {j}     :",template.format(key))
                text = template.format(key)
                text_inputs = torch.cat([clip.tokenize(text,context_length=77) ]).to(device)
                text_features = model.encode_text(text_inputs)
                print(f"x  : 's value  {text_features.shape} ")
                #x[i,j,:] = text_features
                #print(f"x {i}, {j}, : 's value  {x[i,j,:].shape} ")

    #torch.save(x,"pre_proto.pt")

    '''
    net = Net(1000,64)
    net = net.cuda()
    x= torch.rand(20,10,1000).cuda()
    y= net(x)
    print("x shape {}".format(x.size()))
    print("y shape {}".format(y.size()))
    summary(net,(20,10,1000))
    '''

