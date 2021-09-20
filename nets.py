import torch
import torch.nn as nn

from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, in_chs):
        super(Residual, self).__init__()
        
        out_chs = in_chs
        out1 = out_chs // 4
        self.conv1 = Conv2d(in_chs, out1, 1)
        out2 = out1 // 4
        self.conv2 = Conv2d(out1, out2, 3)
        self.conv3 = Conv2d(out2, out_chs, 1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += x
        return out

class Conv2d(nn.Module):
    def __init__(self, in_chs, out_chs, ksize, stride=1, padding=0):
        super(Conv2d, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, ksize, stride, padding, bias=False),
            nn.BatchNorm2d(out_chs)
        )
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        return self.relu(self.conv(x))
    
class ReductionConv2d(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(ReductionConv2d, self).__init__()

        half_out_chs = out_chs // 2
        self.conv = nn.Sequential(
            Conv2d(in_chs, half_out_chs, 1),
            Conv2d(half_out_chs, out_chs, 3)
        )

    def forward(self, x):
        return self.conv(x)

'''
    grid: S x S
    cell: B bboxes + B scores ( Pr(obj) * IoU = IoU(y_pred, y_true) in ideal case )
    bbox: x, y, w, h, conf where x = gx + delta_x and y = gy + delta_y
    class: Pr(k class | obj) for k=1...K => Pr(k class) * IoU = Pr(k class | obj) * Pr(obj)
'''

class YoloNet(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YoloNet, self).__init__()

        self.S = S
        self.B = B
        self.C = C

        # 24 layers + 2 FCls
        # 448 x 448 x 3
        self.model = nn.Sequential(
            # Module 1
            # 448 x 448 x 3
            Conv2d(3, 32, 3, 2, 2),         # 1
            nn.MaxPool2d(2, 2),             # 2
            # 112 x 112 x 32
            
            # Module 2
            Conv2d(32, 64, 3, 1, 1),        # 3
            nn.MaxPool2d(2, 2),             # 4
            # 56 x 56 x 64

            # Module 3
            Conv2d(64, 512, 3, 1, 1),       # 5
            nn.MaxPool2d(2, 2),             # 6
            # 28 x 28 x 512
    
            # Module 4
            Conv2d(512, 512, 1, 1, 1),      # 7
            Conv2d(512, 1024, 3, 1, 1),     # 8
            Conv2d(1024, 1024, 3),          # 9
            nn.MaxPool2d(2, 2),             # 10
            # 14 x 14 x 1024

            # Module 5
            Conv2d(1024, 1024, 3, 1, 1),     # 11
            Conv2d(1024, 1024, 3, 1, 1),     # 12
            nn.MaxPool2d(2, 2),              # 13
            # 7 x 7 x 1024

            nn.AvgPool2d(7),                 # 14
        )

        # Note: S x S x 5 * B + C = 7 x 7 x 5 * 2 + 1 = 7 x 7 x 11
        self.bbox_predictions = self.B * 5
        prediction_size = self.S * self.S * (self.bbox_predictions + self.C)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, prediction_size)

    def forward(self, x):
        out = x
        if __debug__:
            print('Forward through:')
        for idx in range(0, len(self.model)):
            if __debug__:
                print(f'{idx + 1} :: {self.model[idx]._get_name()}')
            out = self.model[idx](out)

        out = out.view(-1, out.size(1) * out.size(2) * out.size(3))
        out = self.fc2(self.fc1(out))
        out = out.view(-1, self.S, self.S, self.bbox_predictions + self.C)

        if __debug__:
            print(f'\nTensor of Prediction: {out.size()}\n')

        return out

if __name__ == '__main__':
    net = YoloNet().cuda()

    summary(net, (3, 448, 448))
