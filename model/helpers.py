import torch
import torch.nn as nn

def convSet( outFeature, inFeature = 3, numConv = 2, isStart = False):
    layers = []
    for i in range(numConv):
        layers.append(nn.Conv2d(in_channels = inFeature, out_channels= outFeature , kernel_size = 3, padding = 1, bias = False))
        inFeature = outFeature

    layers.append(nn.BatchNorm2d(outFeature))
    layers.append(nn.ReLU())
    if not isStart:
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    block = nn.Sequential(*layers)
    return block



def getFc(inFeature = 4096, outFeature = 4096):
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(inFeature, outFeature),
        nn.ReLU()
    )