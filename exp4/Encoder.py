import torch
import torch.nn as nn
from torchsummary import summary
from config import *
class Encoder(nn.Module):
    def __init__(self, channels=DEFAULTCHANNELS,imagesize=DEFAULTIMAGESIZE,linearhidden=DEFAULTLinearHidden,codelength=DEFAULTCODELENGTH,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels=channels
        self.imagesize=imagesize
        self.linearhidden=linearhidden
        self.codelength=codelength
        self.convNet=self._make_conv_layers()
        # self.fcNet=self._make_fc_layer()
        
    def _make_conv_layers(self):
        layer=[]
        for i in range(len(self.channels)-1):
            in_channel=self.channels[i]
            out_channel=self.channels[i+1]
            layer.append(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1))
            layer.append(nn.PReLU())
            layer.append(nn.BatchNorm2d(out_channel,eps=1e-18))
        return nn.Sequential(*layer)
    
    def _make_fc_layer(self):
        layer=[]
        inputsize=self.channels[-1]*self.imagesize[0]*self.imagesize[1]
        for hidden in self.linearhidden:
            layer.append(nn.Linear(inputsize,hidden))
            layer.append(nn.Dropout(0.5))
            layer.append(nn.PReLU())

            inputsize=hidden
        layer.append(nn.Linear(self.linearhidden[-1],self.codelength))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x=self.convNet(x)
        # out=self.fcNet(x.view(x.shape[0],-1))
        return x
    
    
if __name__=='__main__':
    model=Encoder()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)
    summary(model,input_size=(3,32,32))
    print(model)
    
    
        
        
        