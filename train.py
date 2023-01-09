import random
import argparse
import os
import hydra

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def load_data():
    import json
  
    asdas = 'val/val_v1.0.json'
    with open() as f:
        data = json.load(f)


@hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
def main(config):
    # seed
    set_seed(config.seed)
  
    # load data
  
  
    # 전처리 asd
    # batch box 안에 있는 애들
    tokenizer = tokenizer()
  
  
    
    # 모델 불러오고
  
    # trainer
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main()
