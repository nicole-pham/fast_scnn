import torch

def pixel_accuracy(preds, label):
    acc_sum = torch.sum(preds.argmax(axis=1) == label).item()
    image_size = preds.shape[2] * preds.shape[3]
    acc = float(acc_sum) / (image_size + 1e-10)
    return acc

