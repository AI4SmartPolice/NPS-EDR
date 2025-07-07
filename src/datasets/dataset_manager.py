# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from rdkit import Chem
import os
import csv
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import torch.nn as nn
import re
import numpy as np
import json

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        #print(mol)
        if mol is not None:
            return True
        else:
            return False
    except:
        return False


class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self._load_data()
        
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return self.data_shape


class MolDataset(BaseDataset):
    def __init__(self, split=None, tokenizer_org = None, args = None, transform=None):
        data_path = f"{args.dataset_path}/{args.dataset_name}"
        if data_path.endswith('.pkl'):
            self.data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.txt'):
            self.data = pd.read_table(data_path)
        elif data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.split = split
        self.tokenizer_org = tokenizer_org
        self.task = args.task
        self.args = args

        super(MolDataset, self).__init__(config=None)

    def _load_data(self):
        self.input = []
        self.input_selfies = []
        self.output = []
        self.tasks = []
        self.labels = []
        self.ids = []

        filtered_data = self.data[self.data['split'] == self.split]

        

        self.data  = filtered_data
        self.data_shape = len(filtered_data)
        
        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.input.append(row["smiles"])
            self.output.append(row["think"])
            if(row["NPS"] == "Yes"):
                self.labels.append("A")
            else:
                self.labels.append("B")
            
            self.ids.append(row["id"])
                
    def return_df(self):  
        return self.data
        
    def __getitem__(self, i):
        mol_data = {}
        mol_data['id'] = self.ids[i]
        mol_data['truth'] = self.labels[i]  #

        
        class_label = 1 if self.labels[i] == 'A' else 0
        NPS_label = 'Yes' if self.labels[i] == 'A' else 'No'
        
        
        input_str_1 = self.input[i]
        prompt_str_1 = input_str_1  
        
        
        hypothesis = 'is a Novel Psychoactive Substance (NPS)' if self.labels[i] == 'A' else 'is not a Novel Psychoactive Substance (NPS)'
        prompt_template_2 = """
        You think the molecule <smiles>{input_data}</smiles> {hypothesis_answer}. Please provide a detailed explanation of why.
        Given the potential risks, lean toward suspecting the molecule is an NPS molecule unless clear evidence suggests otherwise.
        Please answer me strictly in the following format:
        <think> Detailed reasoning process </think>
        <answer> Yes / No </answer>
        """
        prompt_2 = prompt_template_2.format(input_data=self.input[i], hypothesis_answer=hypothesis)
        answer_2 = f"<think>{self.output[i]}</think>\n<answer>{NPS_label}</answer>"
        
        
        input_str_2 = f"<｜User｜>{prompt_1}<｜Assistant｜><answer>{NPS_label}</answer>\n<｜User｜>{prompt_2}<｜Assistant｜>{answer_2}"
        prompt_str_2 = f"<｜User｜>{prompt_1}<｜Assistant｜><answer>{NPS_label}</answer>\n<｜User｜>{prompt_2}<｜Assistant｜>"
        
        
        input_encoded_1 = self.tokenizer_org(
            input_str_1,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        input_encoded_2 = self.tokenizer_org(
            input_str_2,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        
        
        prompt_encoded_1 = self.tokenizer_org(
            prompt_str_1,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        prompt_encoded_2 = self.tokenizer_org(
            prompt_str_2,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        
        
        mol_data['stage_1'] = {
            'input': input_encoded_1,
            'prompt': prompt_encoded_1,
            'class_label': torch.tensor(class_label, dtype=torch.long)  
        }
        
        
        labels_2 = input_encoded_2['input_ids'].clone()  # [1, seq_len]
        prompt_len_2 = prompt_encoded_2['input_ids'].size(1)
        if prompt_len_2 < labels_2.size(1):
            labels_2[:, :prompt_len_2] = -100  
        else:
            print(f"Warning: prompt_len_2 {prompt_len_2} >= input_len {labels_2.size(1)} for id {self.ids[i]}")
        
        mol_data['stage_2'] = {
            'input': input_encoded_2,
            'prompt': prompt_encoded_2,
            'labels': labels_2
        }
        
        
        mol_data['prompt_org_1'] = prompt_str_1
        mol_data['prompt_org_2'] = prompt_str_2
        
        return mol_data