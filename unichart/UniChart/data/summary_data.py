import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor
import numpy as np

added_tokens = []

class SummaryChartDataset(Dataset):
    """
    """

    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        task_prefix: str = '<summarize_chart>',
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key


        self.dataset = dataset
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]
        img_path = os.path.join(sample['image'])
        while not os.path.isfile(img_path):#없으면 True 있으면 False 가 되서 빠져나옴
          idx+=1
          if (idx-1)>=len(self.dataset):
            idx=0
          sample = self.dataset[idx]
          img_path = os.path.join(sample['image'])

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        w_scale= np.random.randint(-20,21)
        h_scale= np.random.randint(-20,21)
        new_width = int(width * (1+w_scale/100))
        new_height= int(height * (1+h_scale/100))
        img = img.resize((new_width, new_height), Image.LANCZOS)

        pixel_values = self.processor(img, random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()
        
        if 'origin_text' in sample:
            cap_path=sample['origin_text']
            with open(cap_path) as f:
                json_object = json.load(f)
            if 'description' in json_object:
                if "This scatterplot is a chart showing the distribution of values over time" in json_object["description"]:
                    query="describe this chart about trend and its abnormalities"
                elif "This scatterplot shows the clustering results of groups between" in json_object["description"]:
                    query="describe this chart about clustering and its abnormalities"
                elif "This scatterplot is a graph that represents the relationship between" in json_object["description"]:
                    query="describe this chart about correlation and its abnormalities"
                else:
                    query="describe this chart"
            else:
                query="describe this chart"
        else:
          query="describe this chart"

        # processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.processor.tokenizer.eos_token 
        processed_parse = self.task_prefix + " " + query + " " + self.prompt_end_token + " " + sample['text'] + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt 
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse