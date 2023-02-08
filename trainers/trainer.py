# import os
# import torch
# from utils.util import (predict, predict_start_first)
# from utils.metric import ANLS
# from transformers import AutoTokenizer
# import wandb


# class Trainer():
#     def __init__(self, model, optimizer, config, data_loader,device):
#         self.model = model.to(device)
#         self.optimizer = optimizer
#         self.config = config
#         self.data_loader = data_loader
#         self.train_data_loader = data_loader.train_data_loader
#         self.valid_data_loader = data_loader.valid_data_loader
#         self.answers = data_loader.answers
#         self.device = device
#         self.epochs = config['epochs']
#         self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

#     def train(self):
#         self.model.train()
#         if not os.path.exists('/opt/ml/input/saved/'):
#             os.mkdir('/opt/ml/input/saved/')
#         patience = 0
#         best_loss = float('inf')
#         for epoch in range(self.epochs):
#             train_loss = 0
#             for idx, batch in enumerate(self.train_data_loader):
#                 self.optimizer.zero_grad()
                
#                 outputs = self.get_output(batch)
#                 if outputs is None:
#                     continue
                
#                 loss = outputs.loss
#                 train_loss += loss.item()
#                 loss.backward()
#                 self.optimizer.step()
#             train_loss /= len(self.train_data_loader)
                
#             valid_loss = 0
#             for idx, batch in enumerate(self.valid_data_loader):
#                 with torch.no_grad():
#                     outputs = self.get_output(batch)
#                     if outputs is None:
#                         continue
#                     loss = outputs.loss
#                     valid_loss += loss.item()
#             valid_loss /= len(self.valid_data_loader)
            
#             print(f"epoch: {epoch:3}  train: {train_loss:.4f}  valid: {valid_loss:.4f}", end='')
#             torch.save(self.model.state_dict(), '/opt/ml/input/saved/'+self.config.model+'_last.pt')
#             if valid_loss < best_loss:
#                 patience  = 0
#                 best_loss = valid_loss
#                 torch.save(self.model.state_dict(), '/opt/ml/input/saved/'+self.config.model+'_best.pt')
#                 print(f"  best: {best_loss:.4f}  new best model!")
            
#             else:
#                 patience += 1
#                 print(f"  best: {best_loss:.4f}")
#                 if patience >= self.config.early_stop:
#                     break
            
#             # wandb.log({
#             #     "train_loss": train_loss,
#             #     "valid_loss": valid_loss
#             # })
        
#     def validate(self):
#         self.model.eval()
#         with torch.no_grad():
#             for idx, batch in enumerate(self.valid_data_loader):
#                 outputs = self.get_output(batch)
#                 predicted_start_idx, predicted_end_idx = predict_start_first(outputs)
                
#                 for i in range(batch['input_ids'].shape[0]):
#                     # predicted answer
#                     pred = self.tokenizer.decode(batch['input_ids'][i, predicted_start_idx[i]:predicted_end_idx[i]+1])

#                     answers = self.answers[self.config.batch_data * idx + i]
#                     print("-" * 80)
#                     print(f"actual : {answers}")
#                     print(f"predicted : {pred}")
#                     print(f"ANLS : {ANLS(answers, pred)}")
#                     print("-" * 80)

#                 loss = outputs.loss
#                 print("Loss:", loss.item())


#     def get_output(self, batch):
#         input_ids       = batch["input_ids"].to(self.device)
#         attention_mask  = batch["attention_mask"].to(self.device)
#         token_type_ids  = batch["token_type_ids"].to(self.device)
#         bbox            = batch["bbox"].to(self.device)
#         image           = batch["image"].to(self.device)
#         start_positions = batch["start_positions"].to(self.device)
#         end_positions   = batch["end_positions"].to(self.device)
        
#         input_ids       = input_ids[start_positions != 0]
#         attention_mask  = attention_mask[start_positions != 0]
#         token_type_ids  = token_type_ids[start_positions != 0]
#         bbox            = bbox[start_positions != 0]
#         image           = image[start_positions != 0]
#         end_positions   = end_positions[start_positions != 0]
#         start_positions = start_positions[start_positions != 0]

#         # 유효한 input이 없으면 continue
#         if len(start_positions) == 0:
#             return None
        
#         outputs = self.model(input_ids=input_ids, 
#                              attention_mask=attention_mask, 
#                              token_type_ids=token_type_ids,
#                              bbox=bbox, image=image, 
#                              start_positions=start_positions, 
#                              end_positions=end_positions)
        
#         return outputs


# # import os
# # import torch
# # from utils.util import (predict, predict_start_first)
# # from utils.metric import ANLS
# # from transformers import AutoTokenizer


# # class Trainer():
# #     def __init__(self, model, optimizer, config, data_loader,device):
# #         self.model = model.to(device)
# #         self.optimizer = optimizer
# #         self.config = config
# #         self.data_loader = data_loader
# #         self.train_data_loader = data_loader.train_data_loader
# #         self.valid_data_loader = data_loader.valid_data_loader
# #         self.answers = data_loader.answers
# #         self.device = device
# #         self.epochs = config['epochs']
# #         self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

# #     def train(self):
# #         self.model.train()
# #         for epoch in range(self.epochs):
# #             total_loss = 0
# #             for idx, batch in enumerate(self.train_data_loader):
# #                 # get the inputs;
# #                 input_ids       = batch["input_ids"].to(self.device)
# #                 attention_mask  = batch["attention_mask"].to(self.device)
# #                 token_type_ids  = batch["token_type_ids"].to(self.device)
# #                 bbox            = batch["bbox"].to(self.device)
# #                 image           = batch["image"].to(self.device)
# #                 start_positions = batch["start_positions"].to(self.device)
# #                 end_positions   = batch["end_positions"].to(self.device)
                
# #                 # TODO 지우기
# #                 input_ids       =       input_ids[start_positions != 0]
# #                 attention_mask  =  attention_mask[start_positions != 0]
# #                 token_type_ids  =  token_type_ids[start_positions != 0]
# #                 bbox            =            bbox[start_positions != 0]
# #                 image           =           image[start_positions != 0]
# #                 end_positions   =   end_positions[start_positions != 0]
# #                 start_positions = start_positions[start_positions != 0]

# #                 # zero the parameter gradients
# #                 self.optimizer.zero_grad()

# #                 # forward + backward + optimize
# #                 outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
# #                                      bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
# #                 loss = outputs.loss
# #                 total_loss += loss.item()
# #                 loss.backward()
# #                 self.optimizer.step()
# #             print("Loss:", total_loss / len(self.train_data_loader))

# #     def validate(self):
# #         self.model.eval()
# #         with torch.no_grad():
# #             for idx, batch in enumerate(self.valid_data_loader):
# #                 input_ids = batch["input_ids"].to(self.device)
# #                 attention_mask = batch["attention_mask"].to(self.device)
# #                 token_type_ids = batch["token_type_ids"].to(self.device)
# #                 bbox = batch["bbox"].to(self.device)
# #                 image = batch["image"].to(self.device)
# #                 start_positions = batch["start_positions"].to(self.device)
# #                 end_positions = batch["end_positions"].to(self.device)

# #                 outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
# #                                      bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)
                
# #                 predicted_start_idx, predicted_end_idx = predict_start_first(outputs)
                
# #                 for i in range(batch['input_ids'].shape[0]):
# #                     # predicted answer
# #                     pred = self.tokenizer.decode(batch['input_ids'][i, predicted_start_idx[i]:predicted_end_idx[i]+1])

# #                     answers = self.answers[self.config.batch_data * idx + i]
# #                     print("-" * 80)
# #                     print(f"actual : {answers}")
# #                     print(f"predicted : {pred}")
# #                     print(f"ANLS : {ANLS(answers, pred)}")
# #                     print("-" * 80)

# #                 loss = outputs.loss
# #                 print("Loss:", loss.item())


import os
import torch
from utils.util import (predict, predict_start_first)
from utils.metric import ANLS
from transformers import AutoTokenizer
import wandb

class Trainer():
    def __init__(self, model, optimizer, config, data_loader,device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.data_loader = data_loader
        self.train_data_loader = data_loader.train_data_loader
        self.valid_data_loader = data_loader.valid_data_loader
        self.answers = data_loader.answers
        self.device = device
        self.epochs = config['epochs']
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

    def train(self):
        self.model.train()
        if not os.path.exists('/opt/ml/input/saved/'):
            os.mkdir('/opt/ml/input/saved/')
        patience = 0
        best_loss = float('inf')
        for epoch in range(self.epochs):
            train_loss = 0
            for idx, batch in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                
                outputs = self.get_output(batch)
                if outputs is None:
                    continue
                
                loss = outputs.loss
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss /= len(self.train_data_loader)
                
            valid_loss = 0
            for idx, batch in enumerate(self.valid_data_loader):
                with torch.no_grad():
                    outputs = self.get_output(batch)
                    if outputs is None:
                        continue
                    loss = outputs.loss
                    valid_loss += loss.item()
            valid_loss /= len(self.valid_data_loader)
            
            print(f"epoch: {epoch:3}  train: {train_loss:.4f}  valid: {valid_loss:.4f}", end='')
            torch.save(self.model.state_dict(), '/opt/ml/input/saved/'+self.config.model+'_last.pt')
            if valid_loss < best_loss:
                patience  = 0
                best_loss = valid_loss
                torch.save(self.model.state_dict(), '/opt/ml/input/saved/'+self.config.model+'_best.pt')
                print(f"  best: {best_loss:.4f}  new best model!")
            
            else:
                patience += 1
                print(f"  best: {best_loss:.4f}")
                if patience >= self.config.early_stop:
                    break
            
            # wandb.log({
            #     "train_loss": train_loss,
            #     "valid_loss": valid_loss
            # })


    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_data_loader):
                outputs = self.get_output(batch)
                predicted_start_idx, predicted_end_idx = predict_start_first(outputs)
                
                for i in range(batch['input_ids'].shape[0]):
                    # predicted answer
                    pred = self.tokenizer.decode(batch['input_ids'][i, predicted_start_idx[i]:predicted_end_idx[i]+1])

                    answers = self.answers[self.config.batch_data * idx + i]
                    print("-" * 80)
                    print(f"actual : {answers}")
                    print(f"predicted : {pred}")
                    print(f"ANLS : {ANLS(answers, pred)}")
                    print("-" * 80)

                loss = outputs.loss
                print("Loss:", loss.item())


    def get_output(self, batch):
        input_ids       = batch["input_ids"].to(self.device)
        attention_mask  = batch["attention_mask"].to(self.device)
        token_type_ids  = batch["token_type_ids"].to(self.device)
        bbox            = batch["bbox"].to(self.device)
        image           = batch["image"].to(self.device)
        start_positions = batch["start_positions"].to(self.device)
        end_positions   = batch["end_positions"].to(self.device)
        
        input_ids       = input_ids[start_positions != 0]
        attention_mask  = attention_mask[start_positions != 0]
        token_type_ids  = token_type_ids[start_positions != 0]
        bbox            = bbox[start_positions != 0]
        image           = image[start_positions != 0]
        end_positions   = end_positions[start_positions != 0]
        start_positions = start_positions[start_positions != 0]

        # 유효한 input이 없으면 continue
        if len(start_positions) == 0:
            return None
        
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             token_type_ids=token_type_ids,
                             bbox=bbox, image=image, 
                             start_positions=start_positions, 
                             end_positions=end_positions)
        
        return outputs