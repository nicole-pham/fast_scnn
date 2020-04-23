import os
import argparse
import torch

from torchvision import transforms
from model import FastSCNN
from PIL import Image

from utils.transforms import NewPad
from utils.transforms import pred_2_img


parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--checkpoint', type=str, default='checkpoints/FastSCNN',
                    help='which checkpoint to use')
parser.add_argument('--input_image', type=str,
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')
parser.add_argument('--num_classes', type=int, default=6,
                    help='num of classes in model')


args = parser.parse_args()


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    image_transformer = transforms.Compose([
        NewPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.3396, 0.3628, 0.3362], [0.1315, 0.1287, 0.1333])
    ])
    image = Image.open(args.input_image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    # model = FastSCNN(args.num_classes).to(device)
    print('Finished loading model!')
    model = torch.load(args.checkpoint)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu()
    outname = os.path.splitext(os.path.split(args.input_image)[-1])[0] + '.png'
    pred_2_img(pred, os.path.join(args.outdir, outname))


if __name__ == '__main__':
    predict()

