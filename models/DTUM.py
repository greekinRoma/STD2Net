import torch
from .single_frame import SingleNet
from torch import nn
class DTUMNet(nn.Module):
    def __init__(self, net, in_channel=1,num_classes=1):
        super(DTUMNet, self).__init__()

        self.UNet = SingleNet(model_name=net,in_channel=in_channel,num_classes=32)
        self.DTUM = DTUM(32, num_classes, num_frames=5)

    def forward(self, X_In, Old_Feat, OldFlag):

        FrameNum = X_In.shape[2]
        Features = X_In[:, :, -1, :, :]
        Features = self.UNet(Features)
        Features = torch.unsqueeze(Features, 2)

        if OldFlag == 1:  # append current features based on Old Features, for iteration input
            Features = torch.cat([Old_Feat, Features], 2)

        elif OldFlag == 0 and FrameNum > 1:
            for i_fra in range(FrameNum - 1):
                x_t = X_In[:, :, -2 - i_fra, :, :]
                x_t = self.UNet(x_t)
                x_t = torch.unsqueeze(x_t, 2)
                Features = torch.cat([x_t, Features], 2)

        X_Out = self.DTUM(Features)

        Old_Feat = Features[:,:,1:,:,:]

        return X_Out, Old_Feat
class DTUM(nn.Module):    # final version
    def __init__(self, in_channels, num_classes, num_frames):
        super(DTUM, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        # self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True, ceil_mode=False)
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')
        self.relu = nn.ReLU(inplace=True)

        inch = in_channels
        pad = int((num_frames-1)/2)
        self.bn0 = nn.BatchNorm3d(inch)
        self.conv1_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn1_1 = nn.BatchNorm3d(inch)
        self.conv2_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_1 = nn.BatchNorm3d(inch)
        self.conv3_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_1 = nn.BatchNorm3d(inch)
        self.conv4_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_1 = nn.BatchNorm3d(inch)

        self.conv3_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_2 = nn.BatchNorm3d(inch)
        self.conv2_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_2 = nn.BatchNorm3d(inch)
        self.conv1_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(0,0,0))
        self.bn1_2 = nn.BatchNorm3d(inch)

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )

    def direction(self, arr):
        b,c,t,m,n = arr.size()
        arr[:, :, 1:, :, :] = arr[:, :, 1:, :, :] - m * 2 * n * 2
        arr[:, :, 2:, :, :] = arr[:, :, 2:, :, :] - m * 2 * n * 2
        arr[:, :, 3:, :, :] = arr[:, :, 3:, :, :] - m * 2 * n * 2
        arr[:, :, 4:, :, :] = arr[:, :, 4:, :, :] - m * 2 * n * 2

        arr_r_l = arr % 2  # right 1; left 0     [0 1; 0 1]
        up_down = torch.Tensor(range(0,m)).cuda(arr.device) * n*2*2  #.transpose(0,1)
        up_down = up_down.repeat_interleave(n).reshape(m,n)
        arr1 = arr.float() - up_down.reshape([1,1,1,m,n])
        arr_u_d = (arr1 >= n*2).float() * 2  # up 0; down 1  [0 0; 2 2]
        arr_out = arr_r_l.float() + arr_u_d   # [0 1; 2 3]
        arr_out = (arr_out - 1.5)       # [-1.5 -0.5; 0.5 1.5]

        return arr_out


    def forward(self, x):

        x = self.relu(self.bn0(x))

        x_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        xp_1, ind = self.pool(x_1)
        x_2 = self.relu(self.bn2_1(torch.abs(self.conv2_1(xp_1 * self.direction(ind)))))
        xp_2, ind = self.pool(x_2)
        x_3 = self.relu(self.bn3_1(torch.abs(self.conv3_1(xp_2 * self.direction(ind)))))
        xp_3, ind = self.pool(x_3)
        x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_3 * self.direction(ind)))))

        o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(x_4),x_3], dim=1))))
        o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach()
        o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))

        x_out = self.final(torch.cat([o_1, torch.unsqueeze(x[:,:,-1,:,:],2)], dim=1))

        return x_out