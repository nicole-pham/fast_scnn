import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms.functional as F
import numbers
import numpy as np
import datetime

from utils.transforms import ToClassLabels
from utils.transforms import NewPad

from train import Trainer
from model import FastSCNN
from utils.dataset import PotsdamDataset
from metrics import pixel_accuracy

# python -m pip install setuptools tdqm matplotlib numpy torch torchvision rasterio opencv-python

num_epochs = 100
batch_size = 2
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

ClassesColors = {
    (255, 255, 255): 0, # impervious_surfaces
    (0, 0, 255): 1, # building
    (0, 255, 255): 2, # low_vegetation
    (0, 255, 0): 3, # tree
    (255, 255, 0): 4, # car
    (255, 0, 0): 5 # background
    }

def preprocessing(image, mask):
    mask_transformer = transforms.Compose([
        NewPad(fill=(255,0,0)),
        ToClassLabels()
#         transforms.Lambda(lambda x: to_class_labels(x))
    ])
    image_transformer = transforms.Compose([
        NewPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.3396, 0.3628, 0.3362], [0.1315, 0.1287, 0.1333])
    ])
    return image_transformer(image).float(), mask_transformer(mask)

'''
train_image_path = './data/UDD5/train/splitted/src/'
train_label_path = './fast_scnn/data/UDD5/train/splitted/gt/'
test_image_path = './fast_scnn/data/UDD5/val/splitted/src/'
test_label_path = './fast_scnn/data/UDD5/val/splitted/gt/'
'''

train_image_path = './data/Potsdam_6k/training/imgs'
train_label_path = './data/Potsdam_6k/training/masks'
test_image_path = './data/Potsdam_6k/validation/imgs'
test_label_path = './data/Potsdam_6k/validation/masks'

ds_train = PotsdamDataset(train_image_path, train_label_path, transform=None)
ds_test = PotsdamDataset(test_image_path, test_label_path, transform=None)

dl_train = DataLoader(ds_train, batch_size, shuffle=False)
dl_test = DataLoader(ds_test, batch_size, shuffle=False)

model = FastSCNN(num_classes=18)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()
success_metric = pixel_accuracy
trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
fit_res = trainer.fit(dl_train,
                      dl_test,
                      num_epochs= num_epochs,
                      checkpoints='checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d"))

print(fit_res)
