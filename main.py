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
from utils.dataset import PostdamDataset, UDD
from metrics import pixel_accuracy

num_epochs = 100
batch_size = 4
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def UDD_preprocessing(image, mask):
    mask = np.array(mask).astype('int64')
    mask = torch.torch.from_numpy(mask)
    image_transformer = transforms.Compose([
        NewPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.3967, 0.4193, 0.4018], [0.1837, 0.1673, 0.1833])
    ])
    return image_transformer(image).float(), mask

train_image_path = './data/UDD5/train/splitted/src/'
train_label_path = './fast_scnn/data/UDD5/train/splitted/gt/'
test_image_path = './fast_scnn/data/UDD5/val/splitted/src/'
test_label_path = './fast_scnn/data/UDD5/val/splitted/gt/'
ds_train = UDD(train_image_path, train_label_path, transform=UDD_preprocessing)
ds_test = UDD(test_image_path, test_label_path, transform=UDD_preprocessing)

dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_test = DataLoader(ds_test, batch_size, shuffle=False)

model = FastSCNN(num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()
success_metric = pixel_accuracy
trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
fit_res = trainer.fit(dl_train,
                      dl_test,
                      num_epochs= num_epochs,
                      checkpoints='checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d"))

print(fit_res)
