import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import Features, Sequence, Value, Array2D, Array3D
from utils.util import (
    load_data,
    get_extractor,
    get_ocr_results,
    get_ocr_words_and_boxes,
    encode_dataset,
)


class DataLoader():
    def __init__(self, config):
        self.config = config
        train_data = load_data(self.config, 'train')
        valid_data = load_data(self.config, 'val')
        train_df = pd.DataFrame(train_data['data'])
        valid_df = pd.DataFrame(valid_data['data'])

        train_len, valid_len = len(train_df), len(valid_df)
        
        debug = True
        
        if debug:
            train_len, valid_len = 1, 1
            
        train_dataset = Dataset.from_pandas(train_df.iloc[:train_len])
        valid_dataset = Dataset.from_pandas(valid_df.iloc[:valid_len])


        # we need to define custom features
        # TODO: shape 하드코딩 제거
        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'image': Array3D(dtype="int64", shape=(3, 224, 224)),
            'start_positions': Value(dtype='int64'),
            'end_positions': Value(dtype='int64'),
        })

        print("processing dataset...")
        use_ocr_library = False
        if use_ocr_library:
            feature_extractor = get_extractor(self.config)
            train_preprocess = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'train')
            valid_preprocess = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'val')
        else:
            train_preprocess = lambda x: get_ocr_results(x, self.config, 'train')
            valid_preprocess = lambda x: get_ocr_results(x, self.config, 'val')
        
        train_dataset_with_ocr = train_dataset.map(train_preprocess, batched=True, batch_size=2)
        valid_dataset_with_ocr = valid_dataset.map(valid_preprocess, batched=True, batch_size=2)

        tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)
        print("encoding dataset...")
        encode = lambda x: encode_dataset(x, tokenizer)
        # TODO: batch_size 하드코딩 제거
        train_encoded_dataset = train_dataset_with_ocr.map(encode, batched=True, batch_size=2,
                                                           remove_columns=train_dataset_with_ocr.column_names,
                                                           features=features)
        valid_encoded_dataset = valid_dataset_with_ocr.map(encode, batched=True, batch_size=2,
                                                           remove_columns=valid_dataset_with_ocr.column_names,
                                                           features=features)

        print("# of \"answer not found\" (train):",
              sum(x == 0 for x in train_encoded_dataset['start_positions']))
        print("# of \"answer not found\" (valid):",
              sum(x == 0 for x in valid_encoded_dataset['start_positions']))

        train_encoded_dataset.set_format(type="torch")
        valid_encoded_dataset.set_format(type="torch")

        # TODO: batch_size 하드코딩 제거
        self.train_data_loader = torch.utils.data.DataLoader(
            train_encoded_dataset, batch_size=4)
        self.valid_data_loader = torch.utils.data.DataLoader(
            valid_encoded_dataset, batch_size=4)
