import os
import argparse
import torch

from torchvision import transforms
from model import FastSCNN
from PIL import Image

from utils.transforms import NewPad
from utils.transforms import pred_2_img
from metrics import pixel_accuracy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--checkpoint', type=str, default='checkpoints/FastSCNN11_29',
                    help='which checkpoint to use')
parser.add_argument('--input_image', type=str,
                    help='path to the input picture')
parser.add_argument('--outdir', default='data/test_result', type=str,
                    help='path to save the predict result')
parser.add_argument('--num_classes', type=int, default=6,
                    help='num of classes in model')
parser.add_argument('--eval', default=None, type=str,
                    help='image label for evaluation score')


args = parser.parse_args()


def predict():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3396, 0.3628, 0.3362], [0.1315, 0.1287, 0.1333])
    ])
    #img = np.load(args.input_image)
    img = np.load('img-0a00f663-3528-4ac2-86f3-36ffbdf5e69b.npy')
    img = img.transpose(1,2,0)
    
    fig = plt.figure(1)
    canvas = FigureCanvas(fig)
    plt.imshow(img)
    canvas.print_figure('test-pic.png')
    
    image = Image.fromarray(img).convert('RGB')
    image = image_transformer(image).unsqueeze(0).to(device)
    model = FastSCNN(args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # model = torch.load(args.checkpoint, map_location=device)
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).to(device)
    #outname = os.path.splitext(os.path.split(args.input_image)[-1])[0] + '.png'
    outname = 'test-pred.png'
    #pred_2_img(pred, os.path.join(args.outdir, outname))
    pred_2_img(pred, os.path.join(outname))
    
    mask = np.load('img-0a00f663-3528-4ac2-86f3-36ffbdf5e69b-mask.npy')
    mask = np.argmax(mask, 0)
    
    ClassesColors = {
        (255,0,0):0,
        (255,255,255):1,
        (255,255,0):2,
        (0,0,255):3,
        (0,255,255):4,
        (0,255,0):5
    }
    
    Class1H2RGB =  dict([[str(val),key] for key,val in ClassesColors.items()])
    cmap = ListedColormap(np.array([Class1H2RGB[k] for k in sorted(Class1H2RGB.keys())])/255.0)
    plt.figure(2)
    plt.imshow(mask,cmap=cmap,rasterized=True)
    plt.figure(3)
    plt.imshow(pred, cmap=cmap, rasterized=True)
    plt.show(block=True)

    if not args.eval is None:
        # only one label for now, so add batch=1 dim
        label = torch.load(args.eval).unsqueeze(0)
        output = outputs[0]
        print("image score: ", pixel_accuracy(output, label))


if __name__ == '__main__':
    predict()

