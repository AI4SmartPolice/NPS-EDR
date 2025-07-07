# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import GenerationConfig

import torch.nn.functional as F
from utils.xutils import print_model_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, Qwen2ForCausalLM, AutoConfig
from transformers import T5ForConditionalGeneration

from peft import get_peft_model, LoraConfig, PeftModelForCausalLM, get_peft_config
from copy import deepcopy
import numpy as np


import logging
from torch.nn.functional import cosine_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainModel(nn.Module):
    def __init__(self, config=None):
        super(MainModel, self).__init__()
        self.config = config
        self.task2id = {
            "stage_1": 0,  
            "stage_2": 1   
        }
        self.current_task = "stage_1"  
        
        
        self.model = Qwen2ForCausalLM.from_pretrained(self.config.model_pretrain)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_pretrain)
        self.model_config = AutoConfig.from_pretrained(self.config.model_pretrain)
        
        

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
            lora_dropout=0.2,
            bias="lora_only",
            task_type="CAUSAL_LM"
        )
      
        self.model = get_peft_model(self.model, lora_config)
        
        
        self.lora_weights = nn.ParameterDict()
        self.initial_weights = {}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        for task in self.task2id:
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    safe_name = f"{task}_{name.replace('.', '_')}"
                    noise = torch.randn_like(param.data) * 0.01
                    self.lora_weights[safe_name] = nn.Parameter((param.data + noise).to(device), requires_grad=(task == "stage_1"))
                    self.initial_weights[safe_name] = param.data.clone().to(device)
                    
        
        self.classifier = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 2)  
        ).to(device)
        
        
        self.apply_lora("stage_1")
        
        
        self.global_step = 0
        
        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def apply_lora(self, task):
        
        for name, param in self.model.named_parameters():
            if "lora" in name:
                safe_name = f"{task}_{name.replace('.', '_')}"
                if safe_name in self.lora_weights:
                    param.data = self.lora_weights[safe_name].data
                    param.requires_grad = (task == self.current_task)
            else:
                param.requires_grad = False
    
    def set_training_task(self, task):
        
        self.current_task = task
        self.apply_lora(task)
        logger.info(f"task: {task}")
    
    def forward_hidden_states(self, inputs):
        self.global_step += 1
        self.model.train()
        
        
        task = self.current_task
        if task not in inputs:
            return {"loss": torch.tensor(0.0, device=next(self.model.parameters()).device, requires_grad=True)}
        
        
        self.apply_lora(task)
        
        
        task_inputs = inputs[task]
        input_ids = task_inputs['input']['input_ids']
        attention_mask = task_inputs['input']['attention_mask']
        

        
        if task == "stage_1":

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

            return hidden_states
            
    def forward(self, inputs):
        self.global_step += 1
        self.model.train()
        
        
        task = self.current_task
        if task not in inputs:
            return {"loss": torch.tensor(0.0, device=next(self.model.parameters()).device, requires_grad=True)}
        
        
        self.apply_lora(task)
        
        
        task_inputs = inputs[task]
        input_ids = task_inputs['input']['input_ids']
        attention_mask = task_inputs['input']['attention_mask']
        

        
        if task == "stage_1":

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, hidden_size]
            logits = self.classifier(pooled)  # [batch_size, 2]
            class_labels = task_inputs['class_label'].to(logits.device)  # [batch_size]
            loss = nn.CrossEntropyLoss()(logits, class_labels)
        else:
            
            labels = task_inputs['labels'].to(input_ids.device)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output.loss if output.loss is not None else torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        return {"loss": loss}
    
    def generate(self, inputs, max_new_tokens=768):
        self.model.eval()
        output_text = []
        
        prompt_key = self.current_task
        
        if prompt_key not in inputs:
            logger.warning(f"can not find {prompt_key}")
            return output_text
        
        self.apply_lora(self.current_task)
        
        input_ids = inputs[prompt_key]['prompt']['input_ids']
        attention_mask = inputs[prompt_key]['prompt']['attention_mask']
        
        if self.current_task == "stage_1":
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
                logits = self.classifier(pooled)  # [batch_size, 2]
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)  # [batch_size]
                output_text = ["A" if pred == 1 else "B" for pred in predictions]
        else:
            
            prompt_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            prompt_lengths = [len(self.tokenizer.encode(pt, add_special_tokens=True)) for pt in prompt_texts]
            
            generation_config = GenerationConfig(
                pad_token_id=self.model.config.pad_token_id,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                num_beams=4,
                do_sample=True,
                top_p=0.85,
                temperature=1.1,
                repetition_penalty=2.5
            )
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens
                )
            
            sequences = generation_output.sequences
            self.tokenizer.padding_side = 'left'
            outputs = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)
            
            for i, output in enumerate(outputs):
                prompt_len = prompt_lengths[i]
                generated_tokens = self.tokenizer.encode(output, add_special_tokens=True)[prompt_len:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                output_text.append(generated_text)
        
        return output_text