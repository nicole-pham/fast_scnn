import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from train import Trainer
from model import FastSCNN
from dataset import PostdamDataset
from metrics import pixel_accuracy

num_epochs = 2
batch_size = 1
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

def to_class_labels(segmented_image):
    w, h = segmented_image.size
    ret = torch.zeros(h, w)
    for i in range(w):
        for j in range(h):
            color = segmented_image.getpixel((i, j))
            try:
                ret[j, i] = ClassesColors[color]
            except KeyError:
                print("Error when converting lable image to class label {color} not im color mapping")
                ret[j, i] = 5
    return ret

def preprocessing(image, mask):
    mask_transformer = transforms.Compose([
        transforms.Lambda(lambda x: to_class_labels(x))
    ])
    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3396, 0.3628, 0.3362], [0.1315, 0.1287, 0.1333])
    ])
    return image_transformer(image).float(), mask_transformer(mask).float()

train_image_path = './data/2_Ortho_RGB_train/'
train_label_path = './data/labels_train/'
test_image_path = './data/2_Ortho_RGB_test/'
test_label_path = './data/labels_test/'


ds_train = PostdamDataset(train_image_path, train_label_path, transform=preprocessing)
ds_test = PostdamDataset(test_image_path, test_label_path, transform=preprocessing)
dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_test = DataLoader(ds_test, batch_size, shuffle=False)

model = FastSCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()
success_metric = pixel_accuracy
trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
fit_res = trainer.fit(dl_train,
                      dl_test,
                      num_epochs= num_epochs,
                      checkpoints='src/saved_models/' + model.__class__.__name__)
