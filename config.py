# import numpy as np

class Config:
    """
    Store all the configuration for data, models and training steps
    """
    TRAIN_EPOCHS = 100
    VAL_EPOCHS = 50
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    DATALOADER_BATCH_SIZE = 32
    DEVICE_TYPE = "gpu"


    def display(self):
        print("\nConfigurations:")
        for c in dir(self):
            if not c.startswith("__") and not callable(getattr(self,c)):
                print(c, getattr(self, c))
        print("\n")
