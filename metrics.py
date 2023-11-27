import torch

def pixel_accuracy(preds, label):
    pred_argmax = preds.argmax(axis=1)
    acc_sum = torch.sum(preds == label).item()
    # divide by batch size
    acc_mean = acc_sum / preds.shape[0]
    image_size = preds.shape[2] * preds.shape[3]
    acc = float(acc_mean) / (image_size + 1e-10)
    return acc

