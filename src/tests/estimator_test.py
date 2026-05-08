

import sys
import os

sys.path.append(os.getcwd())

from logger import *


from src.entity.estimator import MyModel

def main():
    model=MyModel(load_pretrained_weights=True)


if __name__=="__main__":
    main()
