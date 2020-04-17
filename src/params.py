from pathlib import Path
import torch.nn as nn
import numpy as np


def get_params():

	params = {

		'model': {
			'cnn_classifier': {
				'conv1_filters': 32,
				'conv1_kernel_size': 5,

				'conv2_filters': 64,
				'conv2_kernel_size': 5,
				# FIXME:
				'max_seq_length': 200, # implied from seq_len
				'lstm_hidden_size': 40, 

				'activation': nn.ReLU(), # None is linear
				'apply_batch_norm': False,
				'dropout': 0,
			},

			'lstm_encoder': {
				'input_size': 64, # gets from num_classes
				'hidden_size': 40,
				'embedding_size': 40, # None or negative value to disable embedding layer
				'num_layers': 3,
				'dropout': 0.5,
			},
		},

		'train': {

			'lstm_encoder': {
				'num_epochs': 200,
				'learning_rate': 1e-3,
				'batch_size': 1024,

				'manual_seed': 0,
			},
			'cnn_classifier': {
				'num_epochs': 500,
				'learning_rate': 1e-3,
				'batch_size': 1024,

				'manual_seed': 0,

				'weighting_beta': np.inf # if np.inf means equal weights
										 # 0.1 will have 1:10.9603 weights
										 # 1 - 1:1.996
										 # 10 - 1:1.0906
										 # ...etc
				'weight_decay': 0,
				'lr_annealing': True,
				'lr': 1e-3,
			}
		},

		'min_seq_length': 50,
		'seq_length': 200,

	}

	return params
