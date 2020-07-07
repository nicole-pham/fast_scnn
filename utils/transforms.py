import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import numbers
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Potsdam
ClassesColors = {
    (255, 255, 255): 0, # impervious_surfaces
    (0, 0, 255): 1, # building
    (0, 255, 255): 2, # low_vegetation
    (0, 255, 0): 3, # tree
    (255, 255, 0): 4, # car
    (255, 0, 0): 5 # background
    }

potsdam_map = {v: k for k,v in ClassesColors.items()}

UDD_colors = {
    (0, 255, 0): 0,  # vegetation
    (255, 0, 0): 1,  # building
    (0, 0, 255): 2,  # road
    (128, 128, 0): 3,  # car
    (128, 128, 128): 4,  # other
}

UDD_map = {v: k for k,v in UDD_colors.items()}


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

def get_padding(image):
    image_w, image_h = image.size
    width = 2048
    height= 1024
    w_padding = width - image_w
    h_padding = height - image_h
    l_pad = r_pad = w_padding // 2
    r_pad = width - (image_w + r_pad + l_pad)
    t_pad = b_pad = h_padding // 2
    b_pad = height - (image_h + t_pad + b_pad)
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

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
            assert img.shape == (1024, 2048)
            return img
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}'. \
            format(self.fill, self.padding_mode)

def pred_2_img(pred, output_loc='data/output/out.png', show=False):
    label_mask = pred.numpy()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(5):
        r[label_mask == ll] = UDD_map[ll][0]
        g[label_mask == ll] = UDD_map[ll][1]
        b[label_mask == ll] = UDD_map[ll][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if show:
        plt.imshow(rgb)
        plt.show()
    else:
        matplotlib.image.imsave(output_loc, rgb)