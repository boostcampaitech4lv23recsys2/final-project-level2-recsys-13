# import pandas as pd
# import torch
# import json
# from transformers import AutoTokenizer
# from datasets import Dataset, concatenate_datasets
# from datasets import Features, Sequence, Value, Array2D, Array3D
# import datasets
# from utils.util import (
#     load_data,
#     get_extractor,
#     get_ocr_results,
#     get_ocr_words_and_boxes,
#     encode_dataset,
# )


# class DataLoader():
#     def __init__(self, config, mode='train'):
#         self.config = config
#         train_data  = load_data(self.config, 'train')
#         valid_data  = load_data(self.config, 'val')
#         test_data   = load_data(self.config, 'test')
#         train_df    = pd.DataFrame(train_data['data'])
#         valid_df    = pd.DataFrame(valid_data['data'])
#         self.test_df = pd.DataFrame(test_data['data'])
#         self.answers = valid_df['answers']
        
        
#         if config.debug:
#             train_dataset = Dataset.from_pandas(train_df.iloc[:5000])
#             valid_dataset = Dataset.from_pandas(valid_df.iloc[:1500])
#             test_dataset  = Dataset.from_pandas(self.test_df)
        
#         else:
#             train_dataset = Dataset.from_pandas(train_df)
#             valid_dataset = Dataset.from_pandas(valid_df)
#             test_dataset  = Dataset.from_pandas(self.test_df)


#         # we need to define custom features
#         # TODO: shape 하드코딩 제거
#         features = Features({
#             'input_ids': Sequence(feature=Value(dtype='int64')),
#             'bbox': Array2D(dtype="int64", shape=(512, 4)),
#             'attention_mask': Sequence(Value(dtype='int64')),
#             'token_type_ids': Sequence(Value(dtype='int64')),
#             'image': Array3D(dtype="int64", shape=(3, 224, 224)),
#             'start_positions': Value(dtype='int64'),
#             'end_positions': Value(dtype='int64'),
#         })
        
        
#         print("\nprocessing dataset...")
#         if config.use_ocr_library:
#             feature_extractor = get_extractor(self.config)
#             train_preprocess  = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'train')
#             valid_preprocess  = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'val')
#             test_preprocess   = lambda x: get_ocr_words_and_boxes(x, self.config, feature_extractor, 'test')
#         else:
#             train_preprocess  = lambda x: get_ocr_results(x, self.config, 'train')
#             valid_preprocess  = lambda x: get_ocr_results(x, self.config, 'val')
#             test_preprocess   = lambda x: get_ocr_results(x, self.config, 'test')
        
#         # num_proc = config.num_proc
#         if mode == 'train':
#             train_dataset_with_ocr = train_dataset.map(train_preprocess, batched=True, batch_size=config.batch_data, num_proc = config.num_proc)
#             valid_dataset_with_ocr = valid_dataset.map(valid_preprocess, batched=True, batch_size=config.batch_data, num_proc = config.num_proc)
        
#         elif mode == 'test':
#             test_dataset_with_ocr  = test_dataset.map(test_preprocess, batched=True, batch_size=config.batch_data, num_proc = config.num_proc)

        
#         print("\nencoding dataset...")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)
#         encode = lambda x: encode_dataset(x, self.tokenizer, mode)
        
#         # num_proc = config.num_proc
#         if mode == 'train':
#             train_encoded_dataset = train_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,remove_columns=train_dataset_with_ocr.column_names,
#                                                             features=features, num_proc = config.num_proc)

#             valid_encoded_dataset = valid_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data, remove_columns=valid_dataset_with_ocr.column_names,
#                                                             features=features, num_proc = config.num_proc)


#             # # save data
#             # train_encoded_dataset.save_to_disk('/opt/ml/input/data/train_encoded_sebest')
#             # valid_encoded_dataset.save_to_disk('/opt/ml/input/data/valid_encoded_sebest')
#             # print('end of saving data')

#             # # load data
#             # train_encoded0 = datasets.load_from_disk('/opt/ml/input/data/ocr_train_encoded0')
#             # train_encoded1 = datasets.load_from_disk('/opt/ml/input/data/ocr_train_encoded1')
#             # train_encoded2 = datasets.load_from_disk('/opt/ml/input/data/ocr_train_encoded2')
#             # train_encoded3 = datasets.load_from_disk('/opt/ml/input/data/ocr_train_encoded3')
#             # train_encoded_dataset = concatenate_datasets([train_encoded0, train_encoded1, train_encoded2, train_encoded3])
#             # valid_encoded_dataset = datasets.load_from_disk('/opt/ml/input/data/ocr_valid_encoded')
#             # print('end of loading encoded dataset')

#             # train_encoded_dataset = datasets.load_from_disk('/opt/ml/input/data/train_paragraph_encoded')
#             # valid_encoded_dataset = datasets.load_from_disk('/opt/ml/input/data/ocr_valid_encoded')
            
#             print("# of \"answer not found\" (train):",
#                 sum(x == 0 for x in train_encoded_dataset['start_positions']))
#             print("# of \"answer not found\" (valid):",
#                 sum(x == 0 for x in valid_encoded_dataset['start_positions']))
            
#             train_encoded_dataset.set_format(type="torch")
#             valid_encoded_dataset.set_format(type="torch")
            
#             self.train_data_loader = torch.utils.data.DataLoader(
#                 train_encoded_dataset, batch_size=config.batch_data, shuffle=True)
#             self.valid_data_loader = torch.utils.data.DataLoader(
#                 valid_encoded_dataset, batch_size=config.batch_data, shuffle=True)
        
#         # num_proc = config.num_proc
#         elif mode == 'test':
#             features = Features({
#                     'input_ids': Sequence(feature=Value(dtype='int64')),
#                     'bbox': Array2D(dtype="int64", shape=(512, 4)),
#                     'attention_mask': Sequence(Value(dtype='int64')),
#                     'token_type_ids': Sequence(Value(dtype='int64')),
#                     'image': Array3D(dtype="int64", shape=(3, 224, 224)),
#                     'word_ids': Sequence(feature=Value(dtype='int64')),
#                     'start_positions': Value(dtype='int64'),
#                     'end_positions': Value(dtype='int64'),
#                 })

#             test_encoded_dataset = test_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,
#                                                             remove_columns=test_dataset_with_ocr.column_names,
#                                                             features=features, num_proc = config.num_proc)
#             # test_encoded_dataset = datasets.load_from_disk('/opt/ml/input/data/ocr_test_encoded_post')
#             # test_encoded_dataset.save_to_disk('/opt/ml/input/data/ocr_test_encoded_post')

#             test_encoded_dataset.set_format(type="torch")
            
#             self.test_data_loader = torch.utils.data.DataLoader(test_encoded_dataset, batch_size=config.batch_data)


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
    encode_with_stride,
    load_textract_result,
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
        self.answers = valid_df['answers']
        if mode == 'test':
            self.test_df = test_df
        
        if config.debug:
            train_dataset = Dataset.from_pandas(train_df.iloc[:5000])
            valid_dataset = Dataset.from_pandas(valid_df.iloc[:1500])
            test_dataset  = Dataset.from_pandas(test_df)
        
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
            # test_preprocess   = lambda x: load_textract_result(x, self.config, 'test')
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
        encode = lambda x: encode_dataset(x, self.tokenizer, mode)
        # encode = lambda x: encode_with_stride(x, self.tokenizer, mode)
        
        if mode == 'train':
            train_encoded_dataset  = train_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,remove_columns=train_dataset_with_ocr.column_names,
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
                train_encoded_dataset, batch_size=config.batch_size, shuffle=True)
            self.valid_data_loader = torch.utils.data.DataLoader(
                valid_encoded_dataset, batch_size=config.batch_size, shuffle=True)
            
        elif mode == 'test':
            # features for test
            features = Features({
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                'word_ids': Sequence(feature=Value(dtype='int64')),
                'start_positions': Value(dtype='int64'),
                'end_positions': Value(dtype='int64'),
            })
            test_encoded_dataset = test_dataset_with_ocr.map(encode, batched=True, batch_size=config.batch_data,
                                                            remove_columns=test_dataset_with_ocr.column_names,
                                                            features=features, num_proc = config.num_proc)
            
            test_encoded_dataset.set_format(type="torch")
            
            self.test_data_loader = torch.utils.data.DataLoader(
                test_encoded_dataset, batch_size=config.batch_size)