import argparse
import torch
from tqdm import tqdm
import hydra
import numpy as np
from PIL import Image
import json
from data_loader.data_loaders import DataLoader
from transformers import AutoModelForQuestionAnswering

from utils.util import predict


@hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
def main(config):
    model = AutoModelForQuestionAnswering.from_pretrained(config.checkpoint).to(config.device)
    # model.load_state_dict(torch.load(config.model_name))
    data_loader = DataLoader(config, 'test')
    tokenizer = data_loader.tokenizer
    for idx, batch in tqdm(enumerate(data_loader.test_data_loader)):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        token_type_ids = batch["token_type_ids"].to(config.device)
        bbox = batch["bbox"].to(config.device)
        image = batch["image"].to(config.device)

        # forward + backward + optimize

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                bbox=bbox, image=image)
        
        predicted_start_idx, predicted_end_idx = predict(outputs)
        
        for i in range(batch['input_ids'].shape[0]):
            print(tokenizer.decode(batch['input_ids'][i, predicted_start_idx[i]:predicted_end_idx[i]+1]))







def predict_vec(outputs):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_logits = np.array(start_logits)
    end_logits = np.array(end_logits)
    
    # Compute the max score and corresponding indices for each input
    max_scores = np.amax(start_logits[:,:,np.newaxis] + end_logits[:,np.newaxis,:], axis=(1,2))
    predicted_start_idx = np.argmax(start_logits, axis=1)
    predicted_end_idx = np.argmax(end_logits, axis=1)
    
    return predicted_start_idx, predicted_end_idx


        
    
if __name__ == '__main__':
    main()