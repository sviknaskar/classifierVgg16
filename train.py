import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import TrainConfig


def saveCkpt(model, optimizer, epochs, path):
    state = {}
    state["model"] = model.state.dict()
    state["optimizer"] = optimizer.state_dict()
    state["epochs"] = epochs

    prefix, ext = os.path.splitext(path)
    ckptPath = "{}_{}{}".format(prefix, epochs, ext)
    torch.save(state, ckptPath)


def train(model, trainDataLoader, valDataLoader, optimizer, criterion):
    print("configuring model for gpu")
    model = torch.nn.DataParallel(model)
    model = model.to(TrainConfig.DEVICE_TYPE)
    print("\nStarting to train")
    trainingLoss = []
    valLoss = []
    for e in TrainConfig.TRAIN_EPOCHS:
        model.train()
        runTrainLoss = 0
        dataLen = len(trainDataLoader)
        print(f"Epoch{e}")
        with tqdm(total = dataLen) as bar:
            for i, l in trainDataLoader:
                img = i.to(TrainConfig.DEVICE_TYPE)
                lbl = i.to(TrainConfig.DEVICE_TYPE)
                pred = model(img)
                loss = criterion(pred, lbl)
                runTrainLoss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.update()

        valDataLen = len(valDataLoader)
        if valDataLen > 0:
            model.eval()
            runValLoss = 0
            with torch.no_grad():
                with tqdm(total = dataLen) as bar:
                    for i,l in valDataLoader:
                        img = i.to(TrainConfig.DEVICE_TYPE)
                        lbl = i.to(TrainConfig.DEVICE_TYPE)
                        pred = model(img)
                        loss = criterion(pred, lbl)
                        valLoss += loss.item()
                        bar.update()

        trainingLoss.append(runTrainLoss/dataLen)
        valLoss.append(runValLoss/valDataLen)

