import torch.nn as nn

from nets import Conv2d, Residual

# input res: 511 x 511
# output res: 128 x 128
# Note: The hourglass architecture uses global and local perceptive fields.
class Hourglass(nn.Module):
    def __init__(self, in_chs):
        super(Hourglass, self).__init__()
        
        # Note: Reduce feature map resolution 5 times
        self.enc = nn.Sequential(
            conv1 = Conv2d(in_chs, 256, 7),
            conv2 = Residual(256),

            pool1 = nn.MaxPool2d(2, 2),

            conv3 = Conv2d(256, 512, 7),
            conv4 = Residual(512),

            pool2 = nn.MaxPool2d(2, 2),

            conv5 = Conv2d(512, 512, 7),
            conv6 = Residual(512),

            pool3 = nn.MaxPool2d(2, 2),

            conv7 = Conv2d(512, 512, 7),
            conv8 = Residual(512),

            pool4 = nn.MaxPool2d(2, 2),

            conv9 = Conv2d(512, 512, 7),
            conv10 = Residual(512),

            pool5 = nn.MaxPool2d(2, 2)
        )
        
        self.dec = nn.Sequential(
            up5 = nn.Upsample(scaling_factor=2, mode='nearest'),
        
            conv10 = Residual(512),
            conv9 = Conv2d(512, 512, 7),

            up4 = nn.Upsample(scaling_factor=2, mode='nearest'),

            conv8 = Residual(512),
            conv7 = Conv2d(512, 512, 7),

            up3 = nn.Upsample(scaling_factor=2, mode='nearest'),

            conv6 = Residual(512),
            conv5 = Conv2d(512, 512, 7),

            up2 = nn.Upsample(scaling_factor=2, mode='nearest'),

            conv4 = Residual(512),
            conv3 = Conv2d(256, 512, 7),

            up1 = nn.Upsample(scaling_factor=2, mode='nearest'),

            conv2 = Residual(256),
            conv1 = Conv2d(in_chs, 256, 7),
        )
        
        self.skip1 = nn.Sequential(
            Residual(256),
            Residual(256)
        )
        
        self.skip2 = nn.Sequential(
            Residual(512),
            Residual(512)
        )
        
        self.skip3 = nn.Sequential(
            Residual(512),
            Residual(512)
        )
        
        self.skip4 = nn.Sequential(
            Residual(512),
            Residual(512)
        )
        
        self.skip5 = nn.Sequential(
            Residual(512),
            Residual(512)
        )
        
        
    def forward(self, x):
        skip_conns = []
        
        # Encoding
        out = x
        for idx in range(len(self.enc)):
            out = self.enc[idx]
            if idx != 0 and idx % 2 == 0:
                skip_conns.append(out)
        
        # Decoding
        for idx in range(len(self.dec)):
            out = self.dec(out)
            if idx != 0 and idx % 2 == 0:
                pass      
            
        out5 = self.up5(x)
        out = self.conv10(out5)
        out = self.conv9(out)
        
        out4 = self.up4(x)
        out = self.conv8(out4)
        out = self.conv7(out)
        
        out3 = self.up3(x)
        out = self.conv6(out3)
        out = self.conv5(out)
        
        out2 = self.up2(x)
        out = self.conv4(out2)
        out = self.conv3(out)
        
        out1 = self.up1(x)
        out = self.conv2(out1)
        out = self.conv1(out)
        
        return out
