import os
import numpy as np
import random
import torch

import datetime

import torch
from torch.nn.utils.rnn import pad_sequence

import logging
logger = logging.getLogger(__name__)


def ToDevice(obj, device):
    if isinstance(obj, (str)):
        return obj
    if isinstance(obj, (int)):
        return obj
    if isinstance(obj, (float)):
        return obj
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple):
        # Convert tuple to list, modify the list, then convert back to tuple
        obj = list(obj)
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return tuple(obj)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    else:
        return obj.to(device)

def print_model_info(model, level=0, prefix=''):
    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        total_params_module = sum(p.numel() for p in module.parameters())
        trainable_params_module = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += total_params_module
        trainable_params += trainable_params_module

        print(f"{prefix}Module: {name} | Total parameters: {total_params_module} | Trainable parameters: {trainable_params_module}")

        if level > 0:
            print_model_info(module, level=level-1, prefix=prefix + '  ')

    if prefix == '':
        print(f"Total parameters: {total_params} | Trainable parameters: {trainable_params} | Trainable ratio: {trainable_params / total_params:.2%}")


def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    max_length = 2048  # max length

    for key in elem_keys:
        if key in ['stage_1', 'stage_2']:
            
            stage_data = [elem[key] for elem in batch]
            collated_stage = {}

            
            input_ids_tensors = [item['input']['input_ids'].squeeze(0) for item in stage_data]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            if padded_input_ids.size(1) > max_length:
                padded_input_ids = padded_input_ids[:, :max_length]

            attention_mask_tensors = [item['input']['attention_mask'].squeeze(0) for item in stage_data]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True)
            if padded_attention_mask.size(1) > max_length:
                padded_attention_mask = padded_attention_mask[:, :max_length]

            collated_stage['input'] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }

            
            prompt_input_ids_tensors = [item['prompt']['input_ids'].squeeze(0) for item in stage_data]
            padded_prompt_input_ids = pad_sequence(prompt_input_ids_tensors, batch_first=True)
            if padded_prompt_input_ids.size(1) > max_length:
                padded_prompt_input_ids = padded_prompt_input_ids[:, :max_length]

            prompt_attention_mask_tensors = [item['prompt']['attention_mask'].squeeze(0) for item in stage_data]
            padded_prompt_attention_mask = pad_sequence(prompt_attention_mask_tensors, batch_first=True)
            if padded_prompt_attention_mask.size(1) > max_length:
                padded_prompt_attention_mask = padded_prompt_attention_mask[:, :max_length]

            collated_stage['prompt'] = {
                'input_ids': padded_prompt_input_ids,
                'attention_mask': padded_prompt_attention_mask
            }

            
            if key == 'stage_1':
                
                class_labels = [item['class_label'] for item in stage_data]
                collated_stage['class_label'] = torch.tensor(class_labels, dtype=torch.long)
            else:
                
                labels = [item['labels'].squeeze(0) for item in stage_data]
                max_dim1 = max(label.size(0) for label in labels)
                padded_labels = [torch.nn.functional.pad(label, (0, max_dim1 - label.size(0)), value=-100) for label in labels]
                padded_labels = pad_sequence(padded_labels, batch_first=True, padding_value=-100)
                if padded_labels.size(1) > max_length:
                    padded_labels = padded_labels[:, :max_length]
                collated_stage['labels'] = padded_labels

            collated_batch[key] = collated_stage

        elif key in ['instruction','input','output','prompt']:
            input_ids_tensors = [elem[key].input_ids for elem in batch]
            input_ids_tensors = [tensor.squeeze(0) for tensor in input_ids_tensors]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            
            
            if padded_input_ids.size(1) > max_length:
                padded_input_ids = padded_input_ids[:, :max_length]

            attention_mask_tensors = [elem[key].attention_mask for elem in batch]
            attention_mask_tensors = [tensor.squeeze(0) for tensor in attention_mask_tensors]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True)

            # if length > 512, stop at 512
            if padded_attention_mask.size(1) > max_length:
                padded_attention_mask = padded_attention_mask[:, :max_length]

            collated_batch[key] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }
        elif(key in ['graph']):
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ['truth','task','id','prompt_org_1','prompt_org_2']:
            collated_batch[key] = [item[key] for item in batch]
        elif key == 'labels':
            labels = [elem[key] for elem in batch]

            
            max_dim1 = max(label.size(1) for label in labels)

            
            padded_labels = [torch.nn.functional.pad(label, (0, max_dim1 - label.size(1)), value=-100) for label in labels]

           
            padded_labels = pad_sequence(padded_labels, batch_first=True, padding_value=-100)
            collated_batch[key] = padded_labels
        else:
            #print(key)
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data
              

    return collated_batch

def extract_answer(text):
    
    try:
        matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        else:
            return text.strip()
                
    except Exception:
        return None
        
def extract_think(text):
    
    try:
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        end_match = re.search(r'^(.*?)<｜end▁of▁sentence｜>', text, re.DOTALL)
        if end_match:
            return end_match.group(1).strip()
        return text.strip()
    except Exception:
        return text.strip()
        
def compute_reward(pred_answer, true_answer, think=None, text = None):
   
    try:
        y_true = 1 if true_answer.strip().lower() == "yes" else 0
        y_pred = 1 if pred_answer.strip().lower() == "yes" else 0
        answer_reward = 1.0 if y_true == y_pred else 0.0
     
        think_reward = 0.0
        if think:
            
            think_len = len(think.strip())
            length_reward = np.exp(-((think_len - 670) ** 2) / (2 * 150 ** 2))
            
        
            words = think.strip().split()
            diversity_reward = len(set(words)) / len(words) if words else 0.5
            

            think_reward = 0.5 * length_reward + 0.5 * diversity_reward


        reward = 0.5 * answer_reward + 0.5 * think_reward
        return reward
    except Exception as e:
        logger.warning(f"{e}")
        return 0.0