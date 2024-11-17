import numpy as np
import torch
import sklearn
import transformers
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
	"""Calculate the following performance metrics with sklearn:
		- F1-score,
		- Matthews_correlation,
		- Precision,
		- Recall
	"""
	# Exclude padding tokens (padding token ID: -100)
	valid_mask = labels != -100  
	valid_predictions = predictions[valid_mask]
	valid_labels = labels[valid_mask]

	# Calculate metrics
	accuracy = sklearn.metrics.accuracy_score(valid_labels, valid_predictions)
	f1 = sklearn.metrics.f1_score(
			valid_labels, valid_predictions, average="macro", zero_division=0
		)
	mc = sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions)
	precision = sklearn.metrics.precision_score(
			valid_labels, valid_predictions, average="macro", zero_division=0
		)
	recall = sklearn.metrics.recall_score(
			valid_labels, valid_predictions, average="macro", zero_division=0
		)
		
	return {
		"accuracy":accuracy,
		"f1": f1,
		"matthews_correlation": mc,
		"precision": precision,
		"recall": recall,
	}


def preprocess_logits_for_metrics(
	logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):

	# Unpack logits if it's a tuple
	if isinstance(logits, tuple):
		logits = logits[0]
	
	# Reshape logits to 2D if needed
	if logits.ndim == 3:
		logits = logits.reshape(-1, logits.shape[-1])

	return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
	"Compute metrics used for huggingface trainer."
	predictions, labels = eval_pred
	return calculate_metric_with_sklearn(predictions, labels)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
	"""Collects the state dict and dump to disk."""
	state_dict = trainer.model.state_dict()
	if trainer.args.should_save:
		cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
		del state_dict
		trainer._save(output_dir, state_dict=cpu_state_dict)