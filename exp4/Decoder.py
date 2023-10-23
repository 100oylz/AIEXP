import torch
import torch.nn as nn
from config import *
from torchsummary import summary
class Decoder(nn.Module):
    def __init__(self,channels=DEFAULTCHANNELS,imagesize=DEFAULTIMAGESIZE,linearhidden=DEFAULTLinearHidden,codeLength=DEFAULTCODELENGTH, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels=tuple(reversed(channels))
        self.imagesize=imagesize
        self.linearhidden=tuple(reversed(linearhidden))
        self.codeLength=codeLength
        self.convNet=self._make_convTransposed_layer()
        # self.fcNet=self._make_fc_layer()
        
        
    def _make_convTransposed_layer(self):
        layer=[]
        for i in range(len(self.channels)-1):
            in_channel=self.channels[i]
            out_channel=self.channels[i+1]
            layer.append(nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1))
            layer.append(nn.PReLU())
            layer.append(nn.BatchNorm2d(out_channel,eps=1e-18))   
        layer.append(nn.Sigmoid())
        return nn.Sequential(*layer)
    
    def _make_fc_layer(self):
        layer=[]
        inputsize=self.codeLength
        for hidden in self.linearhidden:
            layer.append(nn.Linear(inputsize,hidden))
            layer.append(nn.Dropout(0.5))
            layer.append(nn.PReLU())
            inputsize=hidden
        layer.append(nn.Linear(self.linearhidden[-1],self.channels[0]*self.imagesize[0]*self.imagesize[1]))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        # x=self.fcNet(x)
        out=self.convNet(x.view(x.shape[0],self.channels[0],self.imagesize[0],self.imagesize[1]))
        return out
        
if __name__=='__main__':
    model=Decoder()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    # model.to(device)
    # x=torch.rand((1,100)).to(device)
    # y=model(x)
    # print(y.shape)
    print(model)
        