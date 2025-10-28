# NPS-EDR: Explainable Analysis of New Psychoactive Substances

## Overview
The NPS-EDR (New Psychoactive Substances - Explainable Deep Reasoning) project introduces an innovative AI model designed to identify novel psychoactive substances (NPS) with transparent, chemically grounded reasoning. Utilizing a Cooperative Experts Framework, NPS-EDR combines a classifier and an explainer to deliver accurate NPS predictions and clear explanations.

## Code Documentation
This section outlines the structure and purpose of the code repository for the NPS-EDR project.

1. **data/cot_data**  
   - **toy_nps_cot_data.xlsx**: Contains example data for the Chain-of-Thought (COT) dataset, including SMILES strings, NPS labels, and reasoning annotations for over 2,900 molecules. This file serves as a reference for dataset format and content.
   - For the full dataset, refer to https://zenodo.org/records/16778503; access is available upon request to the corresponding author for legitimate research purposes.
2. **src/datasets**  
   - Contains scripts for dataset construction and preprocessing.

3. **src/models**  
   - Includes code for building the NPS-EDR model, implementing the Cooperative Experts Framework with dual LoRA modules (classifier and explainer). The classifier performs binary NPS/non-NPS prediction, while the explainer generates chemically grounded reasoning, leveraging a Transformer-based architecture.

4. **src/utils**  
   - Provides utilities for reinforcement learning (RL) training, including reward functions for the REINFORCE algorithm (balancing answer accuracy, reasoning length, and diversity). Additional tools support SMILES processing, data visualization, and model evaluation.

5. **src/task_manager.py**  
   - Contains the main execution code for training and inference. It orchestrates the two-stage training process (Supervised Fine-Tuning and RL) and manages inference tasks, integrating classifier predictions with explainer reasoning for NPS identification and analysis.
