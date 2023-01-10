import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import Features, Sequence, Value, Array2D, Array3D
from utils.util import load_data, get_extractor, get_ocr_words_and_boxes, encode_dataset

class DataLoader():
    def __init__(self, config):
        self.config = config
        train_data = load_data(self.config, 'train')
        valid_data = load_data(self.config, 'val')
        train_df = pd.DataFrame(train_data['data'])
        valid_df = pd.DataFrame(valid_data['data'])

        train_dataset = Dataset.from_pandas(train_df.iloc[:50])
        valid_dataset = Dataset.from_pandas(valid_df.iloc[:50])

        feature_extractor = get_extractor(self.config)


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

        train_dataset_with_ocr = train_dataset.map(lambda x: get_ocr_words_and_boxes(x,self.config, feature_extractor, 'train'), batched=True, batch_size=2)
        valid_dataset_with_ocr = valid_dataset.map(lambda x: get_ocr_words_and_boxes(x,self.config, feature_extractor, 'val'), batched=True, batch_size=2)


        tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)

        train_encoded_dataset = train_dataset_with_ocr.map(lambda x:encode_dataset(x, tokenizer), batched=True, batch_size=2, 
                                            remove_columns=train_dataset_with_ocr.column_names,
                                            features=features)

        valid_encoded_dataset = valid_dataset_with_ocr.map(lambda x:encode_dataset(x, tokenizer), batched=True, batch_size=2, 
                                            remove_columns=valid_dataset_with_ocr.column_names,
                                            features=features)


        train_encoded_dataset.set_format(type="torch")
        valid_encoded_dataset.set_format(type="torch")

        train_data_loader = torch.utils.data.DataLoader(train_encoded_dataset, batch_size=4)
        valid_data_loader = torch.utils.data.DataLoader(valid_encoded_dataset, batch_size=4)