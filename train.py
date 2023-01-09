import random
import argparse
import os

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



def main():
    parser = argparse.ArgumentParser()
  
    # data args
    parser.add_argument("--data_dir", default="/opt/ml/input/data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)

    # train args
    parser.add_argument("--lr", default="1e-3", type=float)
    parser.add_argument("--batch", default="128", type=int)
    parser.add_argument("--max_len", default="512", type=int)
    parser.add_argument("--epochs", default="10", type=int)
    parser.add_argument("--seed", default=42, type=int)

    # model args
    parser.add_argument("--model", default="layoutlmv2", type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p",)
    parser.add_argument("--hidden_dropout_prob", type=float,
                          default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_len", default=512, type=int)

    args = parser.parse_args()
  
    # seed
    set_seed()
  
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

train/
  - documents/
  - ocr_results/
  - train_v1.0.json
valid/
test/