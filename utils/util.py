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
    # data_dir: "/opt/ml/input/data"
    ocr_root_dir = config.data_dir + "/" + mode + "/ocr_results/"
    image_root_dir = config.data_dir + "/" + mode + "/"
    ids = examples['ucsf_document_id']
    nums = examples['ucsf_document_page_no']

    # image processing
    images = [Image.open(image_root_dir + image_file).convert("RGB")
              for image_file in examples['image']]  # 'documents/xnbl0037_1.png'

    images = [image.resize(size=(224, 224), resample=Image.BILINEAR)
              for image in images]
    images = [np.array(image) for image in images]
    # flip color channels from RGB to BGR (as Detectron2 requires this)
    images = [image[::-1, :, :] for image in images]

    # text processing
    batch_words, batch_boxes = [], []
    for i in range(len(ids)):
        each_words, each_boxes = [], []
        path = ocr_root_dir + ids[i] + "_" + nums[i] + ".json"
        with open(path) as f:
            ocr = json.load(f)
        # lines 마다
        image_width, image_height = ocr['recognitionResults'][0]['width'], ocr['recognitionResults'][0]['height']
        lines: list[dict] = ocr['recognitionResults'][0]['lines']
        for line in lines:
            words: list[dict] = line['words']
            for word in words:
                boundingBox: list[int] = word['boundingBox']
                text: str = word['text']
                # bounding box 전처리
                x1, y1, x2, y2, x3, y3, x4, y4 = boundingBox
                xs, ys = [x1, x2, x3, x4], [y1, y2, y3, y4]
                x_max, x_min, y_max, y_min = max(xs), min(xs), max(ys), min(ys)
                if x_max - x_min == 0 or y_max - y_min:
                    # 바운딩 박스가 의미없게 쳐진 경우. word랑 박스 추가 안하고 넘어감
                    continue
                # 위 if문 추가한 이후로 아래 except 실행 안될거임.
                try:
                    left, upper, right, lower = normalize_bounding_box(
                        x1, y1, x2, y2, x3, y3, x4, y4)
                    # 모델 바운딩박스 임베딩 입력 차원이 1024임. 근데 1000넘기면 에러 띄움. 깃헙 이슈 가보면 24는 안 쓴다고함. 결론은 range 0-1000에 맞춰줘야함.
                    # 그래서 normalize 해준거. 근데 박스가 [1463, 1660, 1463, 1660, 1463, 1691, 1463, 1691] 이런 애가 있음. 분모 0됨.
                    # 애초에 이런 애들은 박스 자체가 이상하게 쳐진거니까 그냥 이 케이스는 무시하고 데이터에 추가하지 않는 게 좋을 것 같음.
                    assert all(0 <= (each) <= 1000 for each in [
                        left, upper, right, lower])
                except:
                    print(boundingBox, word)
                    print([x1, y1, x2, y2], " at ", path)

                # model 코드보니까 아래처럼 bbox[:,:,0]이 left, ... 니까 위 코드를 이에 맞게 고쳐줬음.
                # left_position_embeddings  = self.x_position_embeddings(bbox[:, :, 0])
                # upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
                # right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
                # lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

                each_words.append(text)
                each_boxes.append([x1, y1, x2, y2])

        batch_words.append(each_words)
        batch_boxes.append(each_boxes)

    examples['image'] = images
    examples['words'] = batch_words
    examples['boxes'] = batch_boxes

    return examples


def normalize_bounding_box(x1, y1, x2, y2, x3, y3, x4, y4):
    # Find the min and max x and y coordinates
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    # Normalize the coordinates to the range of 0-1000
    x1_norm = (x1 - x_min) / (x_max - x_min) * 1000
    y1_norm = (y1 - y_min) / (y_max - y_min) * 1000
    x2_norm = (x2 - x_min) / (x_max - x_min) * 1000
    y2_norm = (y2 - y_min) / (y_max - y_min) * 1000
    x3_norm = (x3 - x_min) / (x_max - x_min) * 1000
    y3_norm = (y3 - y_min) / (y_max - y_min) * 1000
    x4_norm = (x4 - x_min) / (x_max - x_min) * 1000
    y4_norm = (y4 - y_min) / (y_max - y_min) * 1000

    # Find the left, upper, right, and lower coordinates
    left = min(x1_norm, x2_norm, x3_norm, x4_norm)
    upper = min(y1_norm, y2_norm, y3_norm, y4_norm)
    right = max(x1_norm, x2_norm, x3_norm, x4_norm)
    lower = max(y1_norm, y2_norm, y3_norm, y4_norm)

    return left, upper, right, lower

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
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def encode_dataset(examples, tokenizer, max_length=512):
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

            if batch_index >= len(start_positions) or \
                    batch_index >= len(end_positions):
                match = False

        if not match:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    encoding['image'] = examples['image']
    encoding['start_positions'] = start_positions
    encoding['end_positions'] = end_positions

    return encoding
