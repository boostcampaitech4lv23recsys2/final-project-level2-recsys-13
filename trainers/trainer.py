import os
import torch

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
                
    def inference(self, idx, config):
        # step 1: pick a random example
        import json
        from PIL import Image
        
        path = os.path.join(config['data_dir'], 'val/val_v1.0.json')
        # '/content/drive/MyDrive/LayoutLMv2/Tutorial notebooks/DocVQA/val/val_v1.0.json'
        with open(path) as f:
            data = json.load(f)
            
        example = data['data'][idx]
        question = example['question']
        image = Image.open(config.data_dir + '/val/' + example['image']).convert("RGB")

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
                
