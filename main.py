import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms.functional as F
import numbers
import numpy as np
import datetime

from train import Trainer
from model import FastSCNN
from utils.dataset import PotsdamDataset
from metrics import pixel_accuracy

# python -m pip install setuptools tdqm matplotlib numpy torch torchvision rasterio opencv-python

# code heavily borrowed from https://github.com/Eladamar/fast_scnn/blob/master/main.py
num_epochs = 10 # changed epochs from 100 to 10 for time
batch_size = 32 # increased batch size (images are 256x256 instead of 6000x6000)
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

ClassesColors = {
    (255, 0, 0): 0, # background
    (255, 255, 255): 1, # impervious_surfaces
    (255, 255, 0): 2, # car
    (0, 0, 255): 3, # building
    (0, 255, 255): 4, # low_vegetation
    (0, 255, 0): 5 # tree
    }

def preprocessing(image, mask):
    mask_transformer = transforms.Compose([
        transforms.Lambda(lambda x: x)
    ])
    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3387, 0.3621, 0.3354], [0.1034, 0.1037, 0.1073])
    ])
    return image_transformer(image).float(), mask_transformer(mask)

train_image_path = './data/Potsdam_6k/training/imgs'
train_label_path = './data/Potsdam_6k/training/masks'
test_image_path = './data/Potsdam_6k/validation/imgs'
test_label_path = './data/Potsdam_6k/validation/masks'

ds_train = PotsdamDataset(train_image_path, train_label_path, transform=None)
ds_test = PotsdamDataset(test_image_path, test_label_path, transform=None)

dl_train = DataLoader(ds_train, batch_size, shuffle=False)
dl_test = DataLoader(ds_test, batch_size, shuffle=False)

model = FastSCNN(num_classes=6)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()
success_metric = pixel_accuracy
trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
fit_res = trainer.fit(dl_train,
                      dl_test,
                      num_epochs= num_epochs,
                      checkpoints='checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d"))

print(fit_res)
