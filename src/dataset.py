import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class CertDataset(Dataset):

	@staticmethod
	def prepare_dataset(pkl_file, answers_csv, min_length, max_length, dataset_version='4.2'):
		# TODO: drop weekends and holidays

		df = pd.read_pickle(pkl_file)
		df = df.reset_index().dropna()

		main_df = pd.read_csv(answers_csv)
		main_df = main_df[main_df['dataset'].astype(str) == str(dataset_version)]\
			.drop(['dataset', 'details'], axis=1)

		main_df['start'] = pd.to_datetime(main_df['start'], format='%m/%d/%Y %H:%M:%S')
		main_df['end'] = pd.to_datetime(main_df['end'], format='%m/%d/%Y %H:%M:%S')

		df = df.merge(main_df, left_on='user', right_on='user', how='left')
		df['malicious'] = (df.day >= df.start) & (df.day <= df.end)
		df = df.drop(['start', 'end', 'day', 'user'], axis=1)

		df['action_length'] = df.action_id.apply(len)

		df = df[df.action_length < min_length]

		df['action_id'] = df.action_id.apply(lambda x: x[:max_length])
		df['action_id'] = df.action_id.apply(lambda x: x + [0] * (max_length - len(x)))

		actions = np.vstack(df.action_id.values)
		targets = df.malicious.values

		return actions, targets

	def __init__(self, actions, targets, transform=None):

		self.actions = actions
		self.targets = targets.astype(int)
		self.transform = transform

	def __len__(self):
		return len(self.actions)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = {'actions': self.actions[idx], 'targets': self.targets[idx]}

		if self.transform:
			sample = self.transform(sample)

		return sample


def create_data_loaders(dataset, shuffle_dataset=True, validation_split=0.3, batch_size=16, random_seed=0):

	# Creating data indices for training and validation splits:
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
											   sampler=train_sampler)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=valid_sampler)
	
	return train_loader, validation_loader