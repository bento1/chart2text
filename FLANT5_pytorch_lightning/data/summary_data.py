import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
import numpy as np
import copy
added_tokens = []

class SummaryChartDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        input_max_length: int,
        output_max_length: int,
        tokenizer,
        split: str = "train",
        ignore_id: int = -100,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.split = split
        self.ignore_id = ignore_id

        self.sort_json_key = sort_json_key

        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.tokenizer = tokenizer


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]
        
        if 'origin_text' in sample:
            cap_path=sample['origin_text']
            with open(cap_path) as f:
                json_object = json.load(f)
            if 'description' in json_object:
                if "This scatterplot is a chart showing the distribution of values over time" in json_object["description"]:
                    query="describe this data distribution about trend and its abnormalities"
                elif "This scatterplot is a graph that represents the relationship between" in json_object["description"]:
                    query="describe this data distribution about correlation and its abnormalities"
                elif "This scatterplot shows the clustering results of groups between" in json_object["description"]:
                    query="describe this data distribution about differences and clustering and its abnormalities"
                else:
                    query="describe this data distribution"
            else:
                query="describe this data distribution"
        else:
            

            query="describe this data distribution"


        processed_parse = str(json_object['features']) + "Look at the data table above, " + query
        
        input_token = self.tokenizer(
            processed_parse ,
            add_special_tokens=False,
            max_length=self.input_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output_token = self.tokenizer(
            sample['text'] ,
            add_special_tokens=False,
            max_length=self.output_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        
        source_ids=input_token["input_ids"].squeeze(0)
        target_ids=output_token["input_ids"].squeeze(0)
        src_mask = input_token["attention_mask"].squeeze(0)
        target_mask = output_token["attention_mask"].squeeze(0)  
        labels = copy.deepcopy(target_ids)
        labels [labels==0] = -100 # Indicating T5 to ignore the padding tokens
        if self.split == "train":
            return source_ids, target_ids,src_mask,target_mask,labels
        else:
            return source_ids, target_ids,src_mask,target_mask,labels,str(json_object['features'])