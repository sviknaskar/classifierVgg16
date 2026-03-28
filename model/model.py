import torch
import torch.nn as nn
import numpy as np
from .helpers import convSet, getFc
def log(text, array = None):
    """logging functionality"""

    if array is not None:
        text = text.ljust(25)
        text += ("shape: {} , min:{}, max:{}").format(
            str(array.shape),
            str(array.min()) if array.size else "",
            str(array.max()) if array.size else ""
        )
    print(text)

class vgg16(nn.Module):
    def __init__(self, numClasses = 2):
        super().__init__()
        self.block1 = convSet(64, isStart = True)
        self.block2 = convSet(inFeature = 64, outFeature = 128)
        self.block3 = convSet(inFeature = 128, outFeature = 256)
        self.block4 = convSet(inFeature = 256, outFeature = 512)
        self.block5 = convSet(inFeature = 512, outFeature = 512)
        self.block6 = convSet(inFeature = 512, outFeature = 512)
        self.linear1 = getFc(7*7*512, 4096)
        self.linear2 = getFc(4096, 4096)
        self.linear3 = nn.Linear(4096, numClasses)


        # self.sm = nn.Softmax(4096, numClasses)

    def forward(self, x):
        r = self.block1(x)
        r = self.block2(r)
        r = self.block3(r)
        r = self.block4(r)
        r = self.block5(r)
        r = self.block6(r)
        r = r.reshape(r.size(0), -1)
        r = self.linear1(r)
        r = self.linear2(r)
        r = self.linear3(r)
        return r
