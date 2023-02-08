# import argparse
# import torch
# from tqdm import tqdm
# import hydra
# import numpy as np
# from PIL import Image
# import json
# from data_loader.data_loaders import DataLoader
# from transformers import AutoModelForQuestionAnswering

# from utils.util import predict, predict_start_first

# @hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
# def main(config):
#     model = AutoModelForQuestionAnswering.from_pretrained(config.checkpoint).to(config.device)
#     # model.load_state_dict(torch.load(config.model_name))
#     model.load_state_dict(torch.load('/opt/ml/input/saved/layoutlmv2_best.pt'))
#     data_loader = DataLoader(config, 'test')
#     tokenizer = data_loader.tokenizer
#     answers = []
#     for idx, batch in tqdm(enumerate(data_loader.test_data_loader)):
#         input_ids = batch["input_ids"].to(config.device)
#         word_ids = batch['word_ids'].to(config.device)
#         attention_mask = batch["attention_mask"].to(config.device)
#         token_type_ids = batch["token_type_ids"].to(config.device)
#         bbox = batch["bbox"].to(config.device)
#         image = batch["image"].to(config.device)

#         # forward + backward + optimize

#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                                 bbox=bbox, image=image)
        
#         # predicted_start_idx, predicted_end_idx = predict(outputs)
#         predicted_start_idx, predicted_end_idx = predict_start_first(outputs)
        
#         # for i in range(batch['input_ids'].shape[0]):
#             # answers.append(tokenizer.decode(batch['input_ids'][i, predicted_start_idx[i]:predicted_end_idx[i]+1]))

#         for batch_idx in range(batch['input_ids'].shape[0]):
#             answer     = ""
#             pred_start = predicted_start_idx[batch_idx]
#             pred_end   = predicted_end_idx[batch_idx]
#             word_id    = word_ids[batch_idx, pred_start]
#             for i in range(pred_start, pred_end + 1):
#                 if word_id == word_ids[batch_idx, i]:
#                     answer += tokenizer.decode(batch['input_ids'][batch_idx][i])
#                 else:
#                     answer += ' ' + tokenizer.decode(batch['input_ids'][batch_idx][i])
#                     word_id = word_ids[batch_idx, i]
            
#             answer = answer.replace('##', '')
            
#             answers.append(answer)

#     ret = [{'questionId': qid, 'answer': answer} for qid,answer in zip(data_loader.test_df['questionId'].tolist(), answers)]
#     with open('/opt/ml/input/submission/layoutlmv2_no_sort_epoch7.json', 'w') as f:
#         json.dump(ret, f)


# def predict_vec(outputs):
#     start_logits = outputs.start_logits
#     end_logits = outputs.end_logits

#     start_logits = np.array(start_logits)
#     end_logits = np.array(end_logits)
    
#     # Compute the max score and corresponding indices for each input
#     max_scores = np.amax(start_logits[:,:,np.newaxis] + end_logits[:,np.newaxis,:], axis=(1,2))
#     predicted_start_idx = np.argmax(start_logits, axis=1)
#     predicted_end_idx = np.argmax(end_logits, axis=1)
    
#     return predicted_start_idx, predicted_end_idx

    
# if __name__ == '__main__':
#     main()


import argparse
import torch
from tqdm import tqdm
import hydra
import numpy as np
from PIL import Image
import json
from data_loader.data_loaders import DataLoader
from transformers import AutoModelForQuestionAnswering

from utils.util import predict, predict_start_first


@hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
def main(config):
    model = AutoModelForQuestionAnswering.from_pretrained(config.checkpoint).to(config.device)
    model.load_state_dict(torch.load(config.model_name))
    data_loader = DataLoader(config, 'test')
    tokenizer = data_loader.tokenizer
    answers = []
    for idx, batch in tqdm(enumerate(data_loader.test_data_loader)):
        input_ids = batch["input_ids"].to(config.device)
        word_ids = batch['word_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        token_type_ids = batch["token_type_ids"].to(config.device)
        bbox = batch["bbox"].to(config.device)
        image = batch["image"].to(config.device)

        # forward + backward + optimize

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                bbox=bbox, image=image)
        
        predicted_start_idx, predicted_end_idx = predict_start_first(outputs)
        
        for batch_idx in range(batch['input_ids'].shape[0]):
            answer     = ""
            pred_start = predicted_start_idx[batch_idx]
            pred_end   = predicted_end_idx[batch_idx]
            word_id    = word_ids[batch_idx, pred_start]
            for i in range(pred_start, pred_end + 1):
                if word_id == word_ids[batch_idx, i]:
                    answer += tokenizer.decode(batch['input_ids'][batch_idx][i])
                else:
                    answer += ' ' + tokenizer.decode(batch['input_ids'][batch_idx][i])
                    word_id = word_ids[batch_idx, i]

            answer = answer.replace('##', '')

            answers.append(answer)

    ret = [{'questionId': qid, 'answer': answer} for qid,answer in zip(data_loader.test_df['questionId'].tolist(), answers)]
    with open('/opt/ml/input/submission/layoutlmv2_no_sort_epoch11.json', 'w') as f:
        json.dump(ret, f)


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