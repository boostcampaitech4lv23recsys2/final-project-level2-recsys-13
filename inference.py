import argparse
import torch
from tqdm import tqdm
import hydra
from PIL import Image
import json
from data_loader.data_loaders import DataLoader
from transformers import AutoModelForQuestionAnswering


@hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
def main(config):
    model = AutoModelForQuestionAnswering.from_pretrained(config.checkpoint).to(config.device)
    model.load_state_dict(torch.load(config.model_name))
    data_loader = DataLoader(config)
    tokenizer = data_loader.tokenizer
    for idx, batch in enumerate(data_loader.test_data_loader):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        token_type_ids = batch["token_type_ids"].to(config.device)
        bbox = batch["bbox"].to(config.device)
        image = batch["image"].to(config.device)

        # forward + backward + optimize


        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                bbox=bbox, image=image)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        max_score = -float('inf')
        predicted_start_idx = 0
        predicted_end_idx = 0

        for start in range(len(start_logits[0])):
            for end in range(start, len(end_logits[0])):
                score = start_logits[0][start] + end_logits[0][end]
                if score > max_score:
                    max_score = score
                    predicted_start_idx = start
                    predicted_end_idx = end
        
        print(tokenizer.decode(batch['input_ids'].squeeze()[predicted_start_idx:predicted_end_idx+1]))


        
    
if __name__ == '__main__':
    main()