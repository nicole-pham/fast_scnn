import torch
import torchvision.transforms.functional as F
import numbers
import glob
import os
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


ClassesColors = {
    (255, 255, 255): 0, # impervious_surfaces
    (0, 0, 255): 1, # building
    (0, 255, 255): 2, # low_vegetation
    (0, 255, 0): 3, # tree
    (255, 255, 0): 4, # car
    (255, 0, 0): 5 # background
    }

color_2_rgb = {v: k for k,v in ClassesColors.items()}

def get_class_color(color):
    try:
        return ClassesColors[color]
    except KeyError:
        r, g, b = color
        if r > 200 and g > 200 and b > 200:
            return 0
        elif r < 50 and g < 50 and b > 200:
            return 1
        elif r < 50 and g > 50 and b > 200:
            return 2
        elif r < 50 and g > 200 and b < 50:
            return 3
        elif r > 200 and g > 200 and b < 50:
            return 4
        else:
            return 5


def to_class_labels(segmented_image):
    w, h = segmented_image.size
    ret = torch.zeros((h, w), dtype=torch.long)
    for i in range(w):
        for j in range(h):
            color = segmented_image.getpixel((i, j))
            try:
                ret[j, i] = get_class_color(color)
                # closest_color = min(list(ClassesColors.keys()), key=lambda x: np.linalg.norm(np.subtract(x, color)))
                # ret[j,i] = ClassesColors[closest_color]
                # ret[j, i] = ClassesColors[color]
            except KeyError:
                print(f"Error color {closest_color} not im color mapping on {j},{i}, originial color: {color}")
                ret[j, i] = 5
    return ret


def get_padding(image):
    image_w, image_h = image.size
    width = 2048
    height = 1024
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
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1})'. \
            format(self.fill, self.padding_mode)



train_image_path = './data/2_Ortho_RGB_train/'
train_label_path = './data/labels_train/'
test_image_path = './data/2_Ortho_RGB_test/'
test_label_path = './data/labels_test/'
batch_size = 1
new_label_train_folder = './data/labels_train_tensor/'
new_label_test_folder = './data/labels_test_tensor/'


if __name__ == "__main__":
    for labels_dir in [train_label_path, test_label_path]:
        all_labels = glob.glob(os.path.join(labels_dir, '*.tif'))
        for i, label in enumerate(all_labels):
            new_name = label.replace('.tif', '.pt')
            if os.path.exists(new_name):
                continue
            img = Image.open(label)
            img = NewPad(fill=(255, 0, 0))(img)
            img = to_class_labels(img)
            print(f"saved as {new_name}")
            #with open(new_name, 'w') as fd:
            torch.save(img, new_name)


    # all_tensor = glob.glob(os.path.join(train_label_path, '*.pt'))
    # for i, tensor in enumerate(all_tensor):
    #     t = torch.load(tensor)
    #     label_mask = t.numpy()
    #     r = label_mask.copy()
    #     g = label_mask.copy()
    #     b = label_mask.copy()
    #     for ll in range(6):
    #         r[label_mask == ll] = color_2_rgb[ll][0]
    #         g[label_mask == ll] = color_2_rgb[ll][1]
    #         b[label_mask == ll] = color_2_rgb[ll][2]
    #     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    #     rgb[:, :, 0] = r / 255.0
    #     rgb[:, :, 1] = g / 255.0
    #     rgb[:, :, 2] = b / 255.0
    #     matplotlib.image.imsave(tensor.replace('.pt', '.png'), rgb)
        # plt.imshow(rgb)
        # plt.show()

