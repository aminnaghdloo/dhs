import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

import pandas as pd
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
from torch.utils.data import Dataset
import argparse
import yaml

from peft import (
	LoraConfig,
	get_peft_model,
	get_peft_model_state_dict,
)


def train(args):
	"Function to fine-tune a huggingface model on DNA sequence data"

	# set device name
	device = torch.device(args.device)
	torch.cuda.set_device(0)

	# load tokenizer
	tokenizer = transformers.AutoTokenizer.from_pretrained(
		args.model_name,
		cache_dir=args.cache_dir,
		model_max_length=205,
		padding_side="right",
		use_fast=True,
		device_map={'':args.device},
		trust_remote_code=True,
	)

	# define datasets and data collator
	train_dataset = DHSDataset(tokenizer=tokenizer, data_path=args.train_data)
	val_dataset = DHSDataset(tokenizer=tokenizer, data_path=args.valid_data)
	data_collator = DHSDataCollator(tokenizer=tokenizer)

	# load model
	model = transformers.AutoModelForSequenceClassification.from_pretrained(
		args.model_name,
		cache_dir=args.cache_dir,
		num_labels=args.n_labels,
		device_map={'':args.device},
		trust_remote_code=True,
	)

	# load training arguments
	training_args = transformers.TrainingArguments(
		output_dir=args.output_dir,
		evaluation_strategy="steps",
		learning_rate=args.lr,
		num_train_epochs=args.n_epochs,
		weight_decay=args.weight_decay,
		warmup_steps=args.warmup_steps,
		logging_dir=args.logging_dir,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		save_total_limit=args.save_limit,
		per_device_train_batch_size=args.train_batch_size_per_device,
		per_device_eval_batch_size=args.eval_batch_size_per_device,
	)

	# define trainer
	trainer = transformers.Trainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		preprocess_logits_for_metrics=preprocess_logits_for_metrics,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		data_collator=data_collator
	)

	# start training
	trainer.train()

	# save model
	if args.save_model:
		trainer.save_state()
		safe_save_model_for_hf_trainer(
			trainer=trainer,
			output_dir=training_args.output_dir
		)


def parse_args():



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Fine-tuning CLI")
	parser.add_argument(
		"--config", type=str, help="Path to the config file (.yml)")
	args = parser.parse_args()
	
	if args.config:
		with open(args.config, "r") as f:
			config = yaml.safe_load(f)
		args = config
	else:
		ValueError("configuration file (.yml) is not provided!")

	train(args)
