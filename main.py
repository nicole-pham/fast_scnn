import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms.functional as F
import numbers
import numpy as np

from train import Trainer
from model import FastSCNN
from dataset import PostdamDataset
from metrics import pixel_accuracy

num_epochs = 2
batch_size = 2
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

def get_class_color(color):
    try:
        return ClassesColors[color]
    except KeyError:
        r, g, b = color
        if r > 200 and g > 200 and b > 200:
            return 0
        elif r < 50 and g < 50 and b > 200:
            return 1
        elif r < 50 and g > 200 and b > 200:
            return 2
        elif r < 50 and g > 200 and b < 50:
            return 3
        elif r > 200 and g > 200 and b < 50:
            return 4
        else:
            return 5

class ToClassLabels(object):
    def __call__(self, segmented_image):
        if torch.is_tensor(segmented_image):
            return segmented_image
        w, h = segmented_image.size
        ret = torch.zeros((h, w), dtype=torch.long)
        for i in range(w):
            for j in range(h):
                color = segmented_image.getpixel((i, j))
                ret[j, i] = get_class_color(color)
                # closest_color = min(list(ClassesColors.keys()), key=lambda x: np.linalg.norm(np.subtract(x, color)))
                # ret[j,i] = ClassesColors[closest_color]
                # ret[j, i] = ClassesColors[color]
        return ret

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): img to be padded.

        Returns:
            Padded image.
        """
        if torch.is_tensor(img):
            assert img.shape == (3, 1024, 2048)
            return img
        return transforms.functional.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.fill, self.padding_mode)


def preprocessing(image, mask):
    mask_transformer = transforms.Compose([
        NewPad(fill=(255, 0, 0)),
        ToClassLabels()
        # transforms.Lambda(lambda x: to_class_labels(x))
    ])
    image_transformer = transforms.Compose([
        NewPad(),
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

model = FastSCNN(num_classes=6)
print("Loaded model")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()
success_metric = pixel_accuracy
trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
print("Loaded trainer")
fit_res = trainer.fit(dl_train,
                      dl_test,
                      num_epochs= num_epochs,
                      checkpoints='src/saved_models/' + model.__class__.__name__)

print("Finish training")
