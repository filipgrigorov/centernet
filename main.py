import multiprocessing as mp

from cv2 import transform
from matplotlib.colors import Normalize
from nets import YoloNet
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append('.')
from dataset import CocoDataset
from nets import YoloNet

class PersonDetection:
    def __init__(self, device, opts):
        self.device = device
        self.opts = opts

        self.net = YoloNet(self.opts['S'], self.opts['B'], self.opts['C']).to(self.device)

    def train(self):
        root = 'coco'
        img_paths = 'images'
        train_dataset = CocoDataset(
            root, 
            data_type=os.path.join(root, 'train2017'), 
            transforms=None, 
            S=self.opts['S'], 
            B=self.opts['B'], C=self.opts['C'], 
            in_memory=False, is_debug=False
        )

        transform_ops = transforms.Compose([
            transforms.Resize(448),
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            transforms.ToTensor()
        ])

        num_workers = (0.4 * mp.cpu_count())
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=1)

        L2_xy = nn.MSELoss()
        L2_wh = nn.MSELoss()
        L_obj_score = nn.
        L_class_score = nn.BCELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.opts['lr'], momentum=self.opts['momentum'])

        for epoch in range(self.opts['nepochs']):
            for idx, (img, top) in enumerate(train_dataloader):
                img = img.float().to(self.device)
                top = top.float().to(self.device)

                optimizer.zero_grad()
                preds = self.net(img)

                loss = self.opts['weight_coords'] * L_coords(preds, top) + \
                    self.opts['weight_coords'] * L_bbox_score(preds, top) + \
                        L_class_score(preds, top) + self.opts['weight_noobj'] * (1 - L_class_score(preds, top))
                print(f'\rLoss={loss.item()}', end='')
                loss.backward()

                optimizer.step()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts = {
        'nepochs': 120,
        'batch_size': 32,
        'lr': 1e-3,
        'momentum': 0.9,
        'S': 7,
        'B': 2,
        'C': 1,
        'weight_coords': 5.0,
        'weight_noobj': 0.5
    }
    det = PersonDetection(device, opts)
    det.train()