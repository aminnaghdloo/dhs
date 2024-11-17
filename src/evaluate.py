import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from dataclass import DHSDataset, DHSDataCollator
import utils
import pandas as pd
import torch
import argparse
import yaml


def evaluate(config):
	# Load the fine-tuned model and tokenizer
	model = AutoModelForSequenceClassification.from_pretrained(
		config['model_path'],
		num_labels=config['num_labels'],
		trust_remote_code=True
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		config['model_path'],
		model_max_length=205,
		padding_side="right",
		use_fast=True,
		device_map={'':config['device']},
		trust_remote_code=True,
	)
	# Load model weights from disk
	model.load_state_dict(torch.load(config['model_weights_path']))

	# Tokenize and prepare the test dataset
	test_dataset = DHSDataset(tokenizer=tokenizer, data_path=config['input_path'])
	data_collator = DHSDataCollator(tokenizer=tokenizer)

	# Evaluate the model on the test dataset
	training_args = TrainingArguments("test-trainer",
		evaluation_strategy="steps",
		per_device_eval_batch_size=config['batch_size']
		)
	
	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		preprocess_logits_for_metrics=utils.preprocess_logits_for_metrics,
		compute_metrics=utils.compute_metrics,
		train_dataset=None,
		eval_dataset=test_dataset,
		data_collator=data_collator
	)


	trainer = Trainer(model=model)

	results = trainer.evaluate(eval_dataset=test_dataset)
	print(results)

	# Save the evaluation results to a CSV file
	results = pd.DataFrame(results, index=[0])
	results.to_csv(config['output_path'], index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Model evaluation CLI")
	parser.add_argument(
		"--config", type=str, help="Path to the config file (.yml)")
	args = parser.parse_args()
	
	if args.config:
		with open(args.config, "r") as f:
			config = yaml.safe_load(f)
	else:
		ValueError("configuration file (.yml) is not provided!")

	evaluate(config)