# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import warnings
#warnings.filterwarnings("ignore", message="A decoder-only architecture is being used, but right-padding was detected!")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset_manager import MolDataset

from models.model_manager import MainModel

from torch.optim.lr_scheduler import StepLR
import datetime
from transformers import AutoTokenizer
import time

#from xutils import print_model_info, custom_collate_fn, ToDevice
from utils import *

from accelerate import Accelerator
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import gc
import re


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_epochs(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss=None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss_stage_1": [], "val_loss_stage_1": [], "train_loss_stage_2": [], "val_loss_stage_2": []}
    last_ckpt_file = None
    patience = 0
    device = args.device
    best_loss = best_loss if best_loss is not None else float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"========Epoch {epoch + 1}========")
        logger.info(f"training {model.current_task}...")
        
        train_loss = []
        train_loader = tqdm(train_loader, desc=f"training {model.current_task}")
        for mol in train_loader:
            mol = ToDevice(mol, args.device)
            outputs = model(mol)
            loss = outputs['loss']
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info(f"Steps={step} training loss ({model.current_task})=%.4lf" % running_loss.get_average())
                train_loss.append(running_loss.get_average())
                running_loss.reset()
        
        
        loss_values[f"train_loss_{model.current_task}"].append(np.mean(train_loss))
        
        
        val_loss = val_epochs(valid_loader, model, device)
        loss_values[f"val_loss_{model.current_task}"].append(val_loss)
        
        
        if val_loss < best_loss:
            patience = 0
            best_loss = val_loss
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            os.makedirs(f"{args.ckpt_output_path}/{args.task}/{model.current_task}", exist_ok=True)
            ckpt_file = f"{epoch}_{timestamp}_{model.current_task}.pth"
            ckpt_path = os.path.join(f"{args.ckpt_output_path}/{args.task}/{model.current_task}", ckpt_file)
            
            if last_ckpt_file is not None and os.path.exists(last_ckpt_file):
                os.remove(last_ckpt_file)
                logger.info(f"delete: {last_ckpt_file}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': loss_values
            }, ckpt_path)
            
            logger.info(f"Epoch: {epoch}, task: {model.current_task}, best loss: {val_loss}, ckpt: {ckpt_file} ")
            last_ckpt_file = ckpt_path
        else:
            patience += 1
            scheduler.step()
            if last_ckpt_file is not None:
                state_dict = torch.load(last_ckpt_file, map_location='cpu')["model_state_dict"]
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"ckpt: {last_ckpt_file}")
        
        logger.info(f"loss: {loss_values}")
        
        
        if model.current_task == "stage_1" and patience > args.patience:
            id_list, truth_list, result_list = test_epochs(test_loader, model, args.device)
            output_df = pd.DataFrame({
                'id': id_list,
                'truth': truth_list,
                'result': result_list
            })
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_df.to_csv(f'infer_result/results_{timestamp}.csv', index=False)
            
            
            model.set_training_task("stage_2")
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
            patience = 0
            best_loss = float('inf')  
            last_ckpt_file = None  
            continue
        
        
        if model.current_task == "stage_2" and patience > args.patience:
            logger.info(f"early stopping {patience} epochs ")
            break
            
            
            
    args.patience = 0
    patience = 0
    last_ckpt_file = None
    loss_values = {"reason_loss": [], "valid_loss": []}
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    args.lr = 1e-5
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    best_loss = None  

    
    logger.info("REINFORCE...")
    for epoch in range(args.rl_epochs):  
        logger.info(f"========REINFORCE Epoch {epoch + 1}========")
        epoch_loss = []
        reason_loader_iter = tqdm(train_loader, desc="REINFORCE")
        model.train()
        for mol in reason_loader_iter:
            mol = ToDevice(mol, device)
            input_ids = mol['prompt']['input_ids']
            attention_mask = mol['prompt']['attention_mask']
            truths = mol['truth']

            outputs = model.generate(
                {'prompt': {'input_ids': input_ids, 'attention_mask': attention_mask}}
            )

            rewards = []
            for j, output in enumerate(outputs):
                pred_answer = extract_answer(output)
                think = extract_think(output)
                true_answer = truths[j]  
                reward = compute_reward(pred_answer, true_answer, think, output)
                
                rewards.append(reward)

            reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float)
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-6)
            outputs = model(mol)
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)
            labels = mol['labels']  # 
            action_log_probs = []
            for i in range(labels.size(0)):
                seq_log_prob = 0.0
                for t in range(labels.size(1)):
                    if labels[i, t] != -100:
                        seq_log_prob += log_probs[i, t, labels[i, t]]
                action_log_probs.append(seq_log_prob)
            action_log_probs = torch.stack(action_log_probs)  # [batch]
            rl_loss = -torch.mean(action_log_probs * reward_tensor)
            if torch.isfinite(rl_loss):
                rl_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss.update(rl_loss.detach().cpu().item())
            else:
                running_loss.update(0.0)

            step += 1
            if step % args.logging_steps == 0:
                logger.info(f"Steps={step} REINFORCE Loss={running_loss.get_average():.4f}")
                epoch_loss.append(running_loss.get_average())
                running_loss.reset()

        loss_values["reason_loss"].append(np.mean(epoch_loss))
        valid_loss = val_epochs(valid_loader, model, device, stage='inference')
        loss_values["valid_loss"].append(valid_loss)


        if best_loss is None or valid_loss < best_loss:
            patience = 0
            best_loss = valid_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(args.ckpt_output_path):
                os.makedirs(args.ckpt_output_path)

            if last_ckpt_file and os.path.exists(last_ckpt_file):
                os.remove(last_ckpt_file)

            ckpt_file = f"{epoch}_{timestamp}_{model.current_task}.pth"
            last_ckpt_path = os.path.join(args.ckpt_output_path, ckpt_file)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': loss_values
            }, last_ckpt_path)

            message = f"REINFORCE: epoch={epoch}, best_loss={best_loss}, valid_loss={valid_loss}, {ckpt_file}"
            last_ckpt_file = last_ckpt_path
            scheduler.step()
        else:
            patience += 1
            
        print(loss_values)
        if patience > args.patience:
            break
            
    return loss_values

def val_epochs(valid_loader, model, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            #print(mol)
            truth = mol['truth']
            del mol['truth']
            mol = ToDevice(mol, device)
            loss = model(mol)['loss']
            if(i==1):
                result = model.generate(mol)
                temp_result = result[0]
                print(f"truth:{truth[0]} | Result : {temp_result}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)


def test_epochs(test_loader, model, device, message = None):
    model.eval()
    test_loss = 0
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logger.info("Testing...")
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        id_list = []
        truth_list = []
        result_list = []
        for data in test_loader:
            id = data['id']
            prompt_org = data['prompt_org_1']
            truth = data['truth']
            truth_list = truth_list + truth
            id_list = id_list + id
            del data['truth']
            data = ToDevice(data, device)
            result = model.generate(data)
            if(i==1):
                temp_result = result[0]
                print(f"truth:{truth[0]} | Prompt: {prompt_org[0]} | Result : {temp_result}")
            i=i+1
            for r in result:
                result_list.append(r)
        
        return id_list, truth_list, result_list


#toy_mol_train_rot_data.csv
def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="qwen_7b_all")
    parser.add_argument("--dataset_path", type=str, default='../data/cot_data')
    parser.add_argument("--dataset_name", type=str, default='nps_cot_data.csv')
    #parser.add_argument("--dataset_name", type=str, default='toy_nps_cot_data.csv')
    parser.add_argument("--ckpt_output_path", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_name", type=str, default='qwen_7b')
    parser.add_argument("--model_output_path", type=str, default="output")
    parser.add_argument("--log_save_path", type=str, default="log")
    parser.add_argument("--result_save_path", type=str, default="../result")
    parser.add_argument("--latest_checkpoint", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_pretrain", type=str, default="../ckpts/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--peft", type=str, default='lora')



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    print(args)

        
    if(args.mode == 'train'):
        
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)
        print_model_info(model,level=2)
        tokenizer_org = model.tokenizer
        
        train_dataset = MolDataset(split = "train",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        valid_dataset = MolDataset(split = "valid",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        test_dataset = MolDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        # Load checkpoint if available
        if args.latest_checkpoint is not None:
            state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)["model_state_dict"]
            model.load_state_dict(state_dict, strict=True)
            #best_loss = checkpoint['best_loss']
            #best_loss = None
            #model.current_task = checkpoint.get('current_task', 'stage_1')
            model.current_task = 'stage_2'
            model.set_training_task(model.current_task)
            logger.info(f"Loaded checkpoint: {args.latest_checkpoint}, Task: {model.current_task}")
        
        model.to(args.device)
        logger.info(f"Model moved to device: {args.device}")
        
        # Train
        loss_values = train_epochs(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
        logger.info(f"Training completed. Final loss values: {loss_values}")


        
    if(args.mode == 'infer_predict'):
        # Initialize lists to store results
        id_list, truth_list, predict_result_list, description_result_list = [], [], [], []
        csv_path = "../data/cot_data/nps_cot_data.csv"
        # Load CSV file
        df = pd.read_csv(csv_path)
        df = df[df['split']=='test']
        df = df.reset_index(drop=True)
        
        args.latest_checkpoint = "../ckpts/finetune_ckpts/qwen_7b_all/stage_1/best_ckpt.pth"
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)
        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
        model.to(args.device)
        tokenizer_org = model.tokenizer
        
        print_model_info(model,level=2)
    
        id_list = []
        truth_list = []
        predict_result_list = []
        # Process each row
        model.eval()
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            mol_data = {}
            id_val = row['id']
            input_data = row['smiles']  # Assuming SMILES is in 'smiles' column
            truth_label = row['NPS']  # Assuming truth label is in 'label' column ('A' or 'B')
            if(row["NPS"] == "Yes"):
                truth_nps = "A"
            else:
                truth_nps = "B"
            
            model.current_task = 'stage_1'
            # Stage 1: Predict NPS Classification
            prompt_1 = input_data
            inputs_1 = tokenizer_org(prompt_1, return_tensors="pt", truncation=True, padding=False).to(args.device)
            mol_data['stage_1'] = {
                'input': inputs_1,
                'prompt': inputs_1
            }
            with torch.no_grad():
                outputs_1 = model.generate(mol_data)  # Adjust max_length as needed
            predict_output_1 = outputs_1[0]
            id_list.append(id_val)
            truth_list.append(truth_nps)
            predict_result_list.append(predict_output_1)

    
        output_df = pd.DataFrame({
            'id': id_list,
            'truth': truth_list,
            'predict_result': predict_result_list
        })
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_df.to_csv(f'infer_result/results_{timestamp}.csv', index=False)
        
    if(args.mode == 'infer_description'):
        # Initialize lists to store results
        id_list, truth_list, predict_result_list, description_result_list = [], [], [], []
        csv_path = "../data/cot_data/nps_cot_data.csv"
        # Load CSV file
        df = pd.read_csv(csv_path)
        df = df[df['split']=='test']
 
        df = df.reset_index(drop=True)
        
        args.latest_checkpoint = "../ckpts/finetune_ckpts/qwen_7b_all/stage_2/best_ckpt.pth"
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)
        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
        model.to(args.device)
        tokenizer_org = model.tokenizer
        
        print_model_info(model,level=2)
    
        prompt_template_2 = """
        You think the molecule <smiles>{input_data}</smiles> {hypothesis_answer}. Please provide a detailed explanation of why.
        Given the potential risks, lean toward suspecting the molecule is an NPS molecule unless clear evidence suggests otherwise.
        Please answer me strictly in the following format:
        <think> Detailed reasoning process </think>
        <answer> Yes / No </answer>
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'infer_result/results_{timestamp}.txt'
        
        # Process each row
        model.eval()
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            mol_data = {}
            id_val = row['id']
            input_data = row['smiles']  # Assuming SMILES is in 'smiles' column
            truth_label = row['NPS']  # Assuming truth label is in 'label' column ('A' or 'B')
            if(row["NPS"] == "Yes"):
                truth_nps = "A"
            else:
                truth_nps = "B"
            
            model.current_task = 'stage_1'
            # Stage 1: Predict NPS Classification
            prompt_1 = input_data
            inputs_1 = tokenizer_org(prompt_1, return_tensors="pt", truncation=True, padding=False).to(args.device)
            mol_data['stage_1'] = {
                'input': inputs_1,
                'prompt': inputs_1
            }
            with torch.no_grad():
                outputs_1 = model.generate(mol_data)  # Adjust max_length as needed
            predict_output_1 = outputs_1
            
            # Extract Yes/No from <answer>Yes / No</answer>
            predict_nps = predict_output_1
            print(f"truth: {truth_nps}; predict: {predict_nps[0]}")

            model.current_task = 'stage_2'
            # Stage 2: Generate Explanation Based on Stage 1 Prediction
            hypothesis = 'is a Novel Psychoactive Substance (NPS)' if predict_nps[0] == 'A' else 'is not a Novel Psychoactive Substance (NPS)'
            truth_text = 'Yes, please provide a detailed explanation of why.' if predict_nps[0] == 'A' else 'No, you can keep the answer or give the potential risks of suspecting the molecule is an NPS molecule.'
            prompt_2 = prompt_template_2.format(input_data=input_data, hypothesis_answer=hypothesis, truth_text = truth_text, truth_label = truth_label)
            print(prompt_2)
            input_str_2 = f"<｜User｜>{prompt_1}<｜Assistant｜><answer>{predict_nps}</answer>\n<｜User｜>{prompt_2}<｜Assistant｜>"
            inputs_2 = tokenizer_org(input_str_2, return_tensors="pt", truncation=True, padding=False).to(args.device)
            mol_data['stage_2'] = {
                'input': inputs_2,
                'prompt': inputs_2
            }
            with torch.no_grad():
                outputs_2 = model.generate(mol_data)  # Longer for detailed explanation
            description_output = outputs_2[0]
            print(description_output)
            # Append results
            result = {
                'id': id_val,
                'truth': truth_nps,
                'predict_result': predict_nps[0],
                'description_result': description_output
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')  
           
