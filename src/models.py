import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
from torch import autograd


class LSTM_Encoder(nn.Module):
	
	def __init__(self, params):
		super(LSTM_Encoder, self).__init__()

		self.lstm_encoder = nn.LSTM(
			params['input_size'], params['hidden_size'],
			num_layers=params['num_layers'], dropout=params['dropout'], batch_first=True)
		self.dropout = nn.Dropout(params['dropout'])
		self.decoder = nn.Linear(params['hidden_size'], params['input_size'])
		self.softmax = nn.Softmax(dim=2)
		# self.loss = nn.BCELoss()

	def forward(self, sequence):
	

		# x = self.one_hot_encoder(sequence).float()
		x, _ = self.lstm_encoder(sequence)

		if self.training:
			x = self.dropout(x)
			x = self.decoder(x)
			x = self.softmax(x)

			# target = self.one_hot_encoder(sequence[:,:-1]).float()
			# loss = self.loss(x[:,1:], target)
			return x
		else:
			return x
		

class CNN_Classifier(nn.Module):

	def __init__(self, params):
		super(CNN_Classifier, self).__init__()
		# TODO: use sigmoid on matrix element-wise

		self.conv1 = nn.Conv2d(1, params['conv1_filters'], kernel_size=params['conv1_kernel_size'])
		self.maxpool1 = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(params['conv1_filters'], params['conv2_filters'], kernel_size=params['conv2_kernel_size'])
		self.maxpool2 = nn.MaxPool2d(2, stride=2)

		self.flatten = nn.Flatten()
		self.linear = nn.Linear(params['conv2_filters']*params['max_seq_length']*params['lstm_hidden_size']//16, params, 2)
		self.softmax = nn.Softmax()

	def forward(self, x):

		# TODO
		assert(x.shape)

		x = self.conv1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.maxpool2(x)

		x = self.flatten(x)
		x = self.linear(x)
		x = self.softmax(x)

class InsiderClassifier(nn.Module):

	def __init__(self, params):
		super(InsiderClassifier, self).__init__()

		self.lstm_encoder = LSTM_Encoder(self, params['lstm_encoder'])
		self.cnn_classifier = CNN_Classifier(self, params['cnn_classifier'])

	def forward(self, x):
		"""
		x : batch of sequences of action tokens. All of them should be greater than minimum length and truncated
		"""
		# TODO: throw out small sequences
		# TODO: truncate big sequences
		hidden_state = self.lstm_encoder(x)
		scores = self.cnn_classifier(hidden_state)
