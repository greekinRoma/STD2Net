from torch import nn
import torch
class TDecM(nn.Module):
    def __init__(self,num_frames,mid_channels,num_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames
        self.mid_channels = mid_channels
        self.num_class = num_class
        self.origin = torch.nn.Conv2d(in_channels=num_frames*mid_channels,out_channels=mid_channels,kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=mid_channels*2,out_channels=1,stride=1,kernel_size=1)
    def forward(self,inp):
        origin = torch.concat(inp,dim=1)
        b,_,h,w = origin.shape
        origin = self.origin(origin).view(b,self.mid_channels,1,h*w)
        basis = [inp[-1]-inp[0],inp[-1]-inp[1],inp[-1]-inp[2],inp[-1]-inp[3]]
        basis = torch.stack(basis,dim=2).view(b,self.mid_channels,self.num_frames-1,h*w)
        basis1 = torch.nn.functional.normalize(basis,dim=-1)
        basis2 = basis1.transpose(-2,-1)
        weight_score = torch.matmul(origin,basis2)
        out = torch.matmul(weight_score,basis1).view(b,self.mid_channels,w,h)
        out = torch.concat([out,inp[-1]],dim=1)
        out = self.out_conv(out)
        return out