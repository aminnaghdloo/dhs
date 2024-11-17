from torch.utils.data import Dataset
import transformers
import pandas as pd
from typing import Dict
import torch
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class DHSDataset(Dataset):
	"""
	Dataset class for supervised multiclass classification of celltype-specific
	DHS sequences.
	"""
	def __init__(self,
				 data_path: str,
				 tokenizer: transformers.PreTrainedTokenizer):
		super(DHSDataset, self).__init__()

		# load data from pandas dataframe
		self.df = pd.read_feather(data_path)
		output = tokenizer(
			self.df['sequence'].to_list(),
			return_tensors='pt',
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True
		)
		
		self.input_ids = output['input_ids']
		self.attention_masks = output['attention_mask']
		self.labels = self.df['label'].to_list()
		self.num_labels = len(set(self.labels))

	def __len__(self) -> int:
		return (len(self.input_ids))

	def __getitem__(self, i) -> Dict[str, torch.Tensor]:
		return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DHSDataCollator(object):
	"""Collate examples for supervised fine-tuning."""

	tokenizer: transformers.PreTrainedTokenizer

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

		input_ids, labels = tuple(
			[instance[key] for instance in instances] 
				for key in ("input_ids", "labels")
			)

		input_ids = torch.nn.utils.rnn.pad_sequence(
			input_ids, batch_first=True,
			padding_value=self.tokenizer.pad_token_id
			)

		labels = torch.Tensor(labels).long()

		return dict(
			input_ids=input_ids,
			labels=labels,
			attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
		)