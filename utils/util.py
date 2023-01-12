import os
import json
from PIL import Image
import torch
import random
import numpy as np
from transformers import LayoutLMv2FeatureExtractor

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data(config, mode):
    file = mode+'/'+mode+'_v1.0.json'
    with open(os.path.join(config.data_dir, file)) as f:
        data = json.load(f)

    return data

def get_extractor(config):
    if config.model == 'layoutlmv2':
        return LayoutLMv2FeatureExtractor()

def get_ocr_results(examples, config, mode):
    ocr_root_dir = config.data_dir + "/" + mode + "/ocr_results/"
    image_root_dir = config.data_dir + "/" + mode + "/"
    ids = examples['ucsf_document_id']
    nums = examples['ucsf_document_page_no']

    images = [Image.open(image_root_dir + image_file).convert("RGB")
              for image_file in examples['image']]

    images = [image.resize(size=(224, 224), resample=Image.BILINEAR)
              for image in images]
    images = [np.array(image) for image in images]

    images = [image[::-1, :, :] for image in images]

    # text processing
    batch_words, batch_boxes = [], []
    for i in range(len(ids)):
        each_words, each_boxes = [], []
        path = ocr_root_dir + ids[i] + "_" + nums[i] + ".json"
        with open(path) as f:
            ocr = json.load(f)

        image_width, image_height = ocr['recognitionResults'][0]['width'], ocr['recognitionResults'][0]['height']
        lines: list[dict] = ocr['recognitionResults'][0]['lines']
        for line in lines:
            words: list[dict] = line['words']
            for word in words:
                boundingBox: list[int] = word['boundingBox']
                text: str = word['text']
                x1, y1, x2, y2, x3, y3, x4, y4 = boundingBox
                xs, ys = [x1, x2, x3, x4], [y1, y2, y3, y4]
                x_max, x_min, y_max, y_min = max(xs), min(xs), max(ys), min(ys)
                if x_max - x_min == 0 or y_max - y_min == 0:
                    continue
                left, upper, right, lower = normalize_bbox(
                    [x_min, y_min, x_max, y_max], image_width, image_height)
                assert all(0 <= (each) <= 1000 for each in [
                    left, upper, right, lower])

                each_words.append(text)
                each_boxes.append([left, upper, right, lower])

        batch_words.append(each_words)
        batch_boxes.append(each_boxes)

    examples['image'] = images
    examples['words'] = batch_words
    examples['boxes'] = batch_boxes

    return examples

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def get_ocr_words_and_boxes(examples, config, feature_extractor, mode):

    root_dir = os.path.join(config['data_dir'], mode)
    # get a batch of document images
    images = [Image.open(root_dir + '/' + image_file).convert("RGB")
              for image_file in examples['image']]

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
        if len(answer_list) == 0:
            continue
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def encode_dataset(examples, tokenizer, mode='train', max_length=512):
    # take a batch
    questions = examples['question']
    words = examples['words']
    boxes = examples['boxes']

    # encode it
    encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
    encoding['image'] = examples['image']
    
    # next, add start_positions and end_positions
    if mode == 'test':
        encoding['start_positions'] = [0] * len(examples['question'])
        encoding['end_positions']   = [0] * len(examples['question'])
        return encoding
    
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

            if batch_index >= len(start_positions) or \
                    batch_index >= len(end_positions):
                match = False

        if not match:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    encoding['start_positions'] = start_positions
    encoding['end_positions']   = end_positions

    return encoding


def predict(outputs):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    predicted_start_idx_list = []
    predicted_end_idx_list = []

    # TODO vectorized code로 바꾸기 
    for i in range(len(start_logits)):
        predicted_start_idx = 0
        predicted_end_idx = 0
        max_score = -float('inf')
        for start in range(len(start_logits[i])):
            for end in range(start, len(end_logits[i])):
                score = start_logits[i][start] + end_logits[i][end]
                if score > max_score:
                    max_score = score
                    predicted_start_idx = start
                    predicted_end_idx = end
        predicted_start_idx_list.append(predicted_start_idx)
        predicted_end_idx_list.append(predicted_end_idx)
    
    
    
    return predicted_start_idx_list, predicted_end_idx_list


def predict_start_first(outputs):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    predicted_start_idx_list = []
    predicted_end_idx_list = []
    
    start_position = start_logits.argmax(1)

    # TODO vectorized code로 바꾸기 
    for i in range(len(start_logits)):
        
        start = start_position[i]
        predicted_start_idx_list.append(start)
        max_score = -float('inf')
        predicted_end_idx = 0
        
        for end in range(start, len(end_logits[i])):
            score = end_logits[i][end]
            if score > max_score:
                max_score = score
                predicted_end_idx = end
                
        predicted_end_idx_list.append(predicted_end_idx)
    
    return predicted_start_idx_list, predicted_end_idx_list