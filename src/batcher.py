import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class CertDataset(Dataset):

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