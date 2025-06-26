from torch import nn
import torch
class TDecM(nn.Module):
    def __init__(self,num_frames,mid_channels,num_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm3d(mid_channels)
        self.num_frames = num_frames
        self.mid_channels = mid_channels
        self.num_class = num_class
        
        self.origin = nn.Sequential(nn.Conv2d(in_channels=mid_channels*num_frames,out_channels=mid_channels,stride=1,kernel_size=1),
                                      nn.BatchNorm2d(mid_channels),
                                      nn.Conv2d(in_channels=mid_channels,out_channels=mid_channels,kernel_size=1,stride=1))
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=mid_channels,out_channels=mid_channels,stride=1,kernel_size=1),
                                      nn.BatchNorm2d(mid_channels),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=mid_channels,out_channels=1,kernel_size=1,stride=1))
        self.relu = torch.nn.functional.relu
        self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.loc_conv = nn.Sequential(
            nn.Conv2d(kernel_size=1,stride=1,in_channels=mid_channels,out_channels=mid_channels),
            nn.Conv2d(kernel_size=1,stride=1,in_channels=mid_channels,out_channels=mid_channels))
    def forward(self,inp):
        b,c,_,h,w = inp.shape
        origin = self.origin(inp.view(b,-1,h,w)).view(b,self.mid_channels,1,h*w)
        basis = [inp[:,:,-1]-inp[:,:,0],inp[:,:,-1]-inp[:,:,1],inp[:,:,-1]-inp[:,:,2],inp[:,:,-1]-inp[:,:,3],inp[:,:,-1]-(inp[:,:,0]+inp[:,:,1]+inp[:,:,2]+inp[:,:,3])/4]
        basis = torch.stack(basis,dim=2).view(b,self.mid_channels,self.num_frames,h*w)
        basis1 = torch.nn.functional.normalize(basis,dim=-1)
        basis2 = basis1.transpose(-2,-1)
        weight_score = torch.matmul(origin,basis2)
        out = torch.matmul(weight_score,basis1).view(b,self.mid_channels,w,h)
        out = self.out_conv(out+self.loc_conv(inp[:,:,4]))
        return out