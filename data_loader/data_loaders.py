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
    def __init__(self, config, mode='train'):
        self.config = config
        train_data  = load_data(self.config, 'train')
        valid_data  = load_data(self.config, 'val')
        test_data   = load_data(self.config, 'test')
        train_df    = pd.DataFrame(train_data['data'])
        valid_df    = pd.DataFrame(valid_data['data'])
        test_df     = pd.DataFrame(test_data['data'])
        
        if config.debug:
            train_dataset = Dataset.from_pandas(train_df.iloc[:10])
            valid_dataset = Dataset.from_pandas(valid_df.iloc[:10])
            test_dataset  = Dataset.from_pandas(test_df.iloc[:10])
        
        else:
            train_dataset = Dataset.from_pandas(train_df)
            valid_dataset = Dataset.from_pandas(valid_df)
            test_dataset  = Dataset.from_pandas(test_df)


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
        
        
        print("\nprocessing dataset...")
        if config.use_ocr_library:
            feature_extractor = get_extractor(self.config)
            train_preprocess  = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'train')
            valid_preprocess  = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'val')
            test_preprocess   = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'test')
        else:
            train_preprocess  = lambda x: get_ocr_results(x, self.config, 'train')
            valid_preprocess  = lambda x: get_ocr_results(x, self.config, 'val')
            test_preprocess   = lambda x: get_ocr_results(x, self.config, 'test')
        
        if mode == 'train':
            train_dataset_with_ocr = train_dataset.map(train_preprocess, batched=True, batch_size=config.batch_data, num_proc = config.num_proc)
            valid_dataset_with_ocr = valid_dataset.map(valid_preprocess, batched=True, batch_size=config.batch_data, num_proc = config.num_proc)
        
        elif mode == 'test':
            test_dataset_with_ocr  = test_dataset.map(test_preprocess, batched=True, batch_size=config.batch_data, num_proc = config.num_proc)

        
        print("\nencoding dataset...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)
        encode = lambda x: encode_dataset(x, self.tokenizer, config.mode)
        # TODO: batch_size 하드코딩 제거
        if mode == 'train':
            train_encoded_dataset  = train_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,
                                                            remove_columns=train_dataset_with_ocr.column_names,
                                                            features=features, num_proc = config.num_proc)
            valid_encoded_dataset  = valid_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,
                                                            remove_columns=valid_dataset_with_ocr.column_names,
                                                            features=features, num_proc = config.num_proc)
            
            print("# of \"answer not found\" (train):",
                sum(x == 0 for x in train_encoded_dataset['start_positions']))
            print("# of \"answer not found\" (valid):",
                sum(x == 0 for x in valid_encoded_dataset['start_positions']))
            
            train_encoded_dataset.set_format(type="torch")
            valid_encoded_dataset.set_format(type="torch")
            
            self.train_data_loader = torch.utils.data.DataLoader(
                train_encoded_dataset, batch_size=config.batch_data)
            self.valid_data_loader = torch.utils.data.DataLoader(
                valid_encoded_dataset, batch_size=config.batch_data)
            
        elif mode == 'test':
            test_encoded_dataset = test_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,
                                                            remove_columns=test_dataset_with_ocr.column_names,
                                                            features=features, num_proc = config.num_proc)
            
            test_encoded_dataset.set_format(type="torch")
            
            self.test_data_loader = torch.utils.data.DataLoader(
                test_encoded_dataset, batch_size=config.batch_data)
