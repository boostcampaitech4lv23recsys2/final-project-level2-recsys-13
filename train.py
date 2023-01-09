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
        
    return data


@hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
def main(config):
    # seed
    set_seed(config.seed)
  
    # load data
    import pandas as pd
    data = load_data()
    df = pd.DataFrame(data['data'])

    from datasets import Dataset
    dataset = Dataset.from_pandas(df.iloc[:50])

    feature_extractor = get_extractor(config)
    
  
  
    # 전처리 

    from datasets import Features, Sequence, Value, Array2D, Array3D

    # we need to define custom features
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'start_positions': Value(dtype='int64'),
        'end_positions': Value(dtype='int64'),
    })

    dataset_with_ocr = dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)

    encoded_dataset = dataset_with_ocr.map(encode_dataset, batched=True, batch_size=2, 
                                        remove_columns=dataset_with_ocr.column_names,
                                        features=features)


    import torch
    encoded_dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=4)
    batch = next(iter(dataloader))

    idx = 2

    tokenizer.decode(batch['input_ids'][2])

    start_position = batch['start_positions'][idx].item()
    end_position = batch['end_positions'][idx].item()

    tokenizer.decode(batch['input_ids'][idx][start_position:end_position+1])
  
  
    
    # 모델 불러오고
    

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(model, 
                      optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      device=device,)
    # trainer
    trainer = Trainer()
    
    trainer.train()
    #trainer.validate()
    #trainer.inference(idx)

if __name__ == '__main__':
    main()
    

def get_extractor(config):
    from transformers import LayoutLMv2FeatureExtractor
    if config.model == 'layoutlmv2':
        return LayoutLMv2FeatureExtractor()


def get_ocr_words_and_boxes(examples):
    from PIL import Image
    global config, feature_extractor
    
    root_dir = config['data_dir'] + 'val/'
    # get a batch of document images
    images = [Image.open(root_dir + image_file).convert("RGB") for image_file in examples['image']]
    
    # resize every image to 224x224 + apply tesseract to get words + normalized boxes
    encoded_inputs = feature_extractor(images)

    examples['image'] = encoded_inputs.pixel_values
    examples['words'] = encoded_inputs.words
    examples['boxes'] = encoded_inputs.boxes

    return examples

  

def subfinder(words_list, answer_list):  
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def encode_dataset(examples, max_length=512):
    global tokenizer

    # take a batch 
    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']

    # encode it
    encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

    # next, add start_positions and end_positions
    start_positions = []
    end_positions = []
    answers = examples['answers']
    # for every example in the batch:
    for batch_index in range(len(answers)):
        cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words[batch_index]]
        for answer in answers[batch_index]:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            if match:
                break
        # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        if not match:
            for answer in answers[batch_index]:
                for i in range(len(answer)):
                    # drop the ith character from the answer
                    answer_i = answer[:i] + answer[i+1:]
                    # check if we can find this one in the context
                    match, word_idx_start, word_idx_end = subfinder(words_example, answer_i.lower().split())
                    if match:
                        break
        # END OF EXPERIMENT 
    
        if match:
            sequence_ids = encoding.sequence_ids(batch_index)
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(encoding.input_ids[batch_index]) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
        
            word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]
            for id in word_ids:
                if id == word_idx_start:
                    start_positions.append(token_start_index)
                    break
                else:
                    token_start_index += 1

            for id in word_ids[::-1]:
                if id == word_idx_end:
                    end_positions.append(token_end_index)
                    break
                else:
                    token_end_index -= 1
        
            start_position = start_positions[batch_index]
            end_position = end_positions[batch_index]
            reconstructed_answer = tokenizer.decode(encoding.input_ids[batch_index][start_position:end_position+1])
        
        else:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
  
    encoding['image'] = examples['image']
    encoding['start_positions'] = start_positions
    encoding['end_positions'] = end_positions

    return encoding


    
class Trainer():
    def __init__(self, model, optimizer, config, train_data_loader, valid_data_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.epochs = config['epochs']
    
    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for idx, batch in enumerate(self.train_data_loader):
                # get the inputs;
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                bbox = batch["bbox"].to(self.device)
                image = batch["image"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                print("Loss:", loss.item())
                loss.backward()
                self.optimizer.step()
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_data_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                bbox = batch["bbox"].to(self.device)
                image = batch["image"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
    
                loss = outputs.loss
                print("Loss:", loss.item())
                
    def inference(self, idx):
        # step 1: pick a random example
        import json
        from PIL import Image
        
        # path = config['data_dir'] + val/val_v1.0.json
        # '/content/drive/MyDrive/LayoutLMv2/Tutorial notebooks/DocVQA/val/val_v1.0.json'
        with open(path) as f:
            data = json.load(f)
            
        example = data['data'][idx]
        question = example['question']
        image = Image.open(config['data_dir'] + example['image']).convert("RGB")

        # config['processor']: "microsoft/layoutlmv2-base-uncased"
        # processor = LayoutLMv2Processor.from_pretrained(config['processor'])
        processor = self.processor.from_pretrained(config['processor'])

        # prepare for the model
        encoding = processor(image, question, return_tensors="pt")

        """Note that you can also verify what the processor has created, by decoding the `input_ids` back to text:"""
        print(processor.tokenizer.decode(encoding.input_ids.squeeze()))

        # step 2: forward pass
        for k,v in encoding.items():
            encoding[k] = v.to(self.model.device)

        outputs = self.model(**encoding)

        # step 3: get start_logits and end_logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # step 4: get largest logit for both
        predicted_start_idx = start_logits.argmax(-1).item()
        predicted_end_idx = end_logits.argmax(-1).item()
        print("Predicted start idx:", predicted_start_idx)
        print("Predicted end idx:", predicted_end_idx)

        # step 5: decode the predicted answer
        return processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx:predicted_end_idx+1])
                
