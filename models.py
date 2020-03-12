import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
from torch import autograd


class LSTM_Endoder(nn.Module):
	
	def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.5):


		self.one_hot_encoder = None
		self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
		self.sigmoid = nn.Sigmoid()
		self.padding = nn.ZeroPad2d()

	def forward(self, sequence):
	
		output = self.lstm_encoder(sequence)
		output = self.sigmoid(output)
		# TODO: pad to zeroes medium sequences
		# TODO: use sigmoid on matrix element-wise
		return output

class CNN_Classifier(nn.Module):

	def __init__(self, params):

		self.conv1 = nn.Conv2d(1, params['conv1_filters'], kernel_size=params['conv1_kernel_size'])
		self.maxpool1 = nn.MaxPool2d(2,stride=2)
		self.conv2 = nn.Conv2d(params['conv1_filters'], params['conv2_filters'], kernel_size=params['conv2_kernel_size'])
		self.maxpool2 = nn.MaxPool2d(2,stride=2)

		self.flatten = nn.Flatten()
		self.linear = nn.Linear(params['conv2_filters']*params['max_seq_length']*params['lstm_hidden_size']//16, 2)
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

class InsiderClassifier(nn.Module)

	def __init__(self, params):

		self.lstm_encoder = LSTM_Endoder(self, params)
		self.cnn_classifier = CNN_Classifier(self, params)

	def forward(self, x):
		"""
		x : batch of sequences of action tokens. All of them should be greater than minimum length and truncated
		"""
		# TODO: throw out small sequences
		# TODO: truncate big sequences
		hidden_state = self.lstm_encoder(x)
		scores = self.cnn_classifier(hidden_state)
