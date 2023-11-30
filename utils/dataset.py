import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import glob
import sys
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import numpy as np

class ClassesColors:
    impervious_surfaces = (255, 255, 255)
    building = (0, 0, 255)
    low_vegetation = (0, 255, 255)
    tree = (0, 255, 0)
    car = (255, 255, 0)
    background = (255, 0, 0)

    def __len__(self):
        return 3


class PotsdamDataset(Dataset):
    def __init__(self, images_path, label_path, transform=None, load_tensor=False):
        self.images = glob.glob(os.path.join(images_path, '*.npy'))
        self.labels_path = label_path
        self.transform = transform
        self.load_tensor = load_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx >len(self.images):
            raise IndexError("Index out of bound")

        image_path = self.images[idx]
        image_name =  image_path.split('/')[-1]


        img = np.load(image_path)
        img = img.transpose(1,2,0) # data structure should be (img_dim[0], img_dim[1], num_classes)
        #img = Image.fromarray(img, "RGB")
        
        label_name = image_name.replace('RGB', 'label')
        #label_path = os.path.join(self.labels_path, label_name)
        label_path = image_path.replace('.npy', '-mask.npy')
        label_path = label_path.replace('imgs', 'masks')
        
        if self.load_tensor:
            label_path = image_path.replace('.npy', '-mask.npy')
            label = torch.load(label_path)
        else:
            label = np.load(label_path)
            #label = label.transpose(1,2,0) # data structure should be (img_dim[0], img_dim[1], num_classes)

        if self.transform:
            img, label = self.transform(img, label)
        else:
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            label = torch.tensor(label, dtype=torch.float32)

        return img, label

def compute_mean_std(images_path, label_path):
    dataset = PotsdamDataset(images_path, label_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for i, data in enumerate(loader):
        data = data[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(mean)
    print(std)

if __name__ == "__main__":
    train_image_path = './data/Potsdam_6k/training/imgs'
    train_label_path = './data/Potsdam_6k/training/masks'
    # compute_mean_std(sys.argv[1], sys.argv[2])
    compute_mean_std(train_image_path, train_label_path)