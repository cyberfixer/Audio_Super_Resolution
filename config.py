# This file will have the global variables for the model
# Any setting should be here
import torch
from torch import nn
# This just outline for the config file from the TUNet repo.

# TODO: at the end need to remove the sub classes that are not in use


class CONFIG:

    class TASK:
        pass

    class TRAIN:
        lr = 0.01

    class MODEL:
        class TRANSFORMER:
            pass

    class DATA:
        pass

    class LOG:
        pass

    class TEST:
        pass
