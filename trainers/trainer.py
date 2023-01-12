import os
import torch
from utils.metric import ANLS

answers = ['coca cola', 'coca cola company']
preds = ['the coca', 'cocacola', 'coca cola', 'cola', 'cat']

print(ANLS(answers, 'the coca'))

class Trainer():
    def __init__(self, model, optimizer, config, train_data_loader, valid_data_loader, device):
        self.model = model.to(device)
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

