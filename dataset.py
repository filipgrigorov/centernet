import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
import torch

# from PIL import Image
from pycocotools.coco import COCO
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms as transforms

def display_image_annot(img, annts):
    img = io.imread(img['coco_url']) if isinstance(img, dict) else img
    if not annts:
        raise('Annotation is empty')
    for annt in annts:
        if 'bbox' in annt:
            bbox = np.ascontiguousarray(annt['bbox']).astype(np.int)
            img = cv2.rectangle(
                img, 
                (bbox[0], bbox[1]), 
                (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                (0, 255, 0), 3
            )
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
def display_bbox_annot(img, bboxes, S):
    img = io.imread(img['coco_url']) if isinstance(img, dict) else img
    for bbox in bboxes:
        bbox = bbox.astype(np.int)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        img = cv2.circle(
            img, 
            (int(0.5 * (bbox[2] - bbox[0])) + bbox[0], int(0.5 * (bbox[3] - bbox[1])) + bbox[1]), 
            1, (255, 0, 0), 3)
    
    fig = plt.figure(figsize=(10, 10))
    
    # Draw grid with numbers on top of image with cv2
    h, w, c = img.shape
    step_w = w // S
    step_h = h // S
    print(f'image size: {w} x {h}')
    print(f'steps: {step_w} x {step_h}')
    for y in range(0, S):
        row = y * step_h
        img = cv2.line(img, (0, row), (w - 1, row), (255, 0, 0), 1)
    for x in range(0, S):
        col = x * step_w
        img = cv2.line(img, (col, 0), (col, h - 1), (255, 0, 0), 1)

    idx = 0; half_step_w = step_w // 2; half_step_h = step_h // 2
    for row in range(1, S + 1):
        for col in range(0, S):
            img = cv2.putText(
                img, 
                str(idx + 1),
                (col * step_w, row * step_h), 
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (200, 205, 100))
            idx += 1

    plt.axis('off')
    plt.imshow(img)
    plt.show()

class CocoDataset(Dataset):
    def __init__(self, root, data_type, transforms, S, B, C, in_memory=False, is_debug=False):
        self.S = S
        self.B = B
        self.C = 1

        self.data_type = data_type
        self.transforms = transforms
        self.in_memory = in_memory
        self.is_debug = is_debug

        data_type = data_type.split('/')[-1]
        annts_file = f'{ root }/annotations/instances_{ data_type }.json'

        self.coco = COCO(annts_file)
        
        # Note: In theory, ensures persons presence.
        category_ids = self.coco.getCatIds(catNms=['person'])
        image_ids = self.coco.getImgIds(catIds=category_ids)
        self.image_meta = self.coco.loadImgs(image_ids)
        
        self.images = []; self.annts = []
        for image_meta_data in self.image_meta:
            annts_ids = self.coco.getAnnIds(
                imgIds=image_meta_data['id'], catIds=category_ids, iscrowd=False)
            img = io.imread(image_meta_data['coco_url']) if self.in_memory else image_meta_data
            self.images.append(img)
            self.annts.append(self.coco.loadAnns(annts_ids))
        
        #if self.is_debug and len(self.images) > 0 and len(self.annts) > 0:
        #    idx = np.random.randint(0, len(self.images) - 1)
        #    display_image_annot(self.images[idx], self.annts[idx])
            
        self.n = len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx] if self.in_memory else io.imread(self.images[idx]['coco_url'])
        bboxes = [ annt['bbox'] for annt in self.annts[idx] if 'bbox' in annt ]
        bboxes = None if len(bboxes) == 0 else np.array(bboxes)
        
        # Converting to tl and br from x, y, w, h
        for bbox in bboxes:
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
        
        if self.is_debug and bboxes.shape[0] > 0:
            print('\nDisplaying bboxes\n')
            display_bbox_annot(self.images[idx], bboxes, self.S)
            
        if not self.transforms:
            self.transforms = transforms.ToTensor()
            
        img_tensor = self.transforms(img).float()
        tensor_of_prediction = torch.zeros(size=(self.S, self.S, self.B * 5 + self.C))
        
        # Note: Check if center of bbox alings with any of the grid cells
        img_w, img_h = img.shape[: -1]
        for bbox in bboxes:
            #print(bbox)
            x, y, w, h = bbox[0] / img_w, bbox[1] / img_h, bbox[2] / img_w, bbox[3] / img_h
            # x_i, y_i
            cx, cy = x + w / 2, y + h / 2
            #print(f'{cx}x{cy}')
            cell_coords = [
                min(int(self.S * cy), self.S - 1),
                min(int(self.S * cx), self.S - 1),
            ]

            print(cell_coords)
            
            idx = 1
            if sum(tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx : idx + 4]) != 0:
                idx = 6
                if sum(tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx : idx + 4]) != 0:
                    continue

            tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx - 1] = 1.0
            if tensor_of_prediction[cell_coords[0]][cell_coords[1]][-1] == 0.0:
                tensor_of_prediction[cell_coords[0]][cell_coords[1]][-1] = 1.0

            tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx] = cx
            tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx + 1] = cy
            tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx + 2] = w
            tensor_of_prediction[cell_coords[0]][cell_coords[1]][idx + 3] = h

        if __debug__:
            print(f'\n Number of bboxes: {bboxes.shape}\n')
            print(f'\n Tensor Of prediction: size= {tensor_of_prediction.size()}\n{tensor_of_prediction}')
        
        return (img_tensor, tensor_of_prediction)
    
    def __len__(self):
        return self.n

if __name__ == '__main__':
    
    root = 'coco'
    img_paths = 'images'
    S, B, C = 7, 2, 1
    train_dataset = CocoDataset(root, data_type=os.path.join(root, 'train2017'), transforms=None, S=S, B=B, C=C, in_memory=False, is_debug=False)

    train_dataset[3]
