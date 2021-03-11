import numpy as np
import pandas as pd
import itertools

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from scipy.sparse import csc_matrix


class CertDataset(Dataset):

    @staticmethod
    def pad_to_length(series, max_length=200, padding_id=0):
        series = series.apply(lambda x: x[:max_length])
        series = series.apply(
            lambda x: x + [padding_id] * (max_length - len(x)))
        return series
    
    @staticmethod
    def pad_to_length_numpy(series, max_length=200, padding_id=0):
        series = series.apply(lambda x: x[:max_length])
        series = series.apply(
            lambda x: np.concatenate([x, np.ones(max_length - len(x)) * padding_id])
        )
        return series

    @staticmethod
    def pad_topic_matricies(topic_matricies_array, max_length=200):

        # FIXME: not sure if it's correct to do inplace operations inside of
        # function
        for idx, a in enumerate(topic_matricies_array):
            topic_matricies_array[idx] = csc_matrix(
                (a.data, a.indices, a.indptr),
                shape=(max_length, 100)
            )
        return topic_matricies_array

    def __init__(self, actions, targets, content_topics=None, transform=None,
                 conditional_vecs=None, positions=None):

        self.actions = actions
        self.targets = targets.astype(int)
        self.transform = transform
        self.content_topics = content_topics
        self.conditional_vecs = conditional_vecs
        self.positions = positions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'actions': self.actions[idx], 'targets': self.targets[idx]}
        

        if self.content_topics is not None:
            sample['content_topics'] = self.content_topics[idx].toarray()
            
        if self.conditional_vecs is not None:
            sample['conditional_vecs'] = self.conditional_vecs[idx]

        if self.positions is not None:
            sample['positions'] = self.positions[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_data_loaders(
        dataset,
        shuffle_dataset=True,
        validation_split=0.3,
        batch_size=16,
        random_seed=0):

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

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader


class SkipGramDataset(Dataset):

    @staticmethod
    def prepare_dataset(pkl_file, min_length, dataset_version='4.2'):
        """
        Returns simple flat list of action ids
        """

        df = pd.read_pickle(pkl_file)
        df = df.reset_index().dropna()
        df = df.sort_values(['user', 'day'])

        # filter out small sequences
        df['action_length'] = df.action_id.apply(len)
        df = df[df.action_length < min_length]

        actions = df['action_id'].values.tolist()
        # flatten list of lists
        actions = list(itertools.chain.from_iterable(actions))

        return actions

    def __init__(self, actions, window_size=3, padding_id=0, transform=None):

        self.actions = actions
        self.transform = transform
        self.window_size = window_size
        self.padding_id = padding_id

        padding_tail = [padding_id] * window_size
        self.actions = padding_tail + self.actions + padding_tail * 2

    def __len__(self):
        return len(self.actions) - self.window_size * 3

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        contexts = np.array(self.actions[idx:idx + 2 * self.window_size])
        centers = contexts[self.window_size]  # get the center of each window
        # remove center indecies
        contexts = np.delete(contexts, self.window_size)

        return {'context': contexts, 'center': centers}
