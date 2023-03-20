import os

from torch.utils.data import Dataset
from PIL import Image


class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None,train=True,rate=0.8):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = os.listdir(hr_dir)
        self.hr_images.sort(key= lambda x: int(x.replace('.png','')))
        self.lr_images = os.listdir(lr_dir)
        self.lr_images.sort(key=lambda x: int(x.replace('.png', '')))
        # img_num=len(self.hr_images)
        # if train:
        #     self.hr_images = self.hr_images[:int(img_num * rate)]
        #     self.lr_images = self.lr_images[:int(img_num * rate)]
        # else:
        #     self.hr_images = self.hr_images[int(img_num * rate):]
        #     self.lr_images = self.lr_images[int(img_num * rate):]
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        min_max = (-1, 1)
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        hr_image = hr_image * (min_max[1] - min_max[0]) + min_max[0]
        lr_image = lr_image * (min_max[1] - min_max[0]) + min_max[0]
        return lr_image, hr_image , lr_path ,hr_path
