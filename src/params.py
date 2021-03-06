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
                'max_seq_length': 200,  # implied from seq_len
                'lstm_hidden_size': 40,

                'activation': 'relu',  # None is linear
                'apply_batch_norm': False,
                'dropout': 0,
            },

            'lstm_encoder': {
                'input_size': 64,  # gets from num_classes
                'hidden_size': 40,

                #  None or negative value to disable embedding layer
                'embedding_size': 40,

                'num_layers': 3,
                'dropout': 0.5,
            },
        },

        'train': {

            'lstm_encoder': {
                'num_epochs': 200,
                'lr': 1e-3,
                'batch_size': 1024,

                'manual_seed': 0,
            },
            'cnn_classifier': {
                # since we eploy the early
                # stopping we apply large number of epochs
                'num_epochs': 1000,
                'batch_size': 1024,

                'manual_seed': 0,

                'weighting_beta': 1e3,  # if np.inf means equal weights
                # 0.1 will have 1:10.9603 weights
                # 1 - 1:1.996
                # 10 - 1:1.0906
                # ...etc
                'weight_decay': 0,
                'lr_annealing': True,  # It's hardcoded between 1e-2 and 1e-4
                'lr': 1e-3,
            }
        },

        'min_seq_length': 50,
        'seq_length': 200,

    }

    return params
