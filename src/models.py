import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
from torch import autograd


class LSTM_Encoder(nn.Module):
	
	def __init__(self, params, padding_idx=None):
		super(LSTM_Encoder, self).__init__()

		self.input_size = params['input_size']

		self.embedding = None
		lstm_input_size = params['input_size']
		if params['embedding_size'] and params['embedding_size'] > 0:
			self.embedding = nn.Embedding(params['input_size'],
				params['embedding_size'],
				padding_idx=padding_idx)
			lstm_input_size = params['embedding_size']

		self.lstm_encoder = nn.LSTM(
			lstm_input_size, params['hidden_size'],
			num_layers=params['num_layers'], dropout=params['dropout'], batch_first=True)
		self.dropout = nn.Dropout(params['dropout'])
		self.decoder = nn.Linear(params['hidden_size'], params['input_size'])
		self.log_softmax = nn.LogSoftmax(dim=2)
		# self.loss = nn.NLLLoss()

	def forward(self, sequence):
		"""
		Input Args:
		* sequence - tensor of indecies with shape (batch_size, seq_len)
		Output:
		If model in train mode:
			tensor in shape (batch_size, seq_len-1, input_size)
		If model in eval mode:
			tensor in shape (batch_size, seq_len, hidden size)
		"""

		if self.embedding:
			x = self.embedding(sequence)
		else:
			x = self.one_hot_encoder(sequence, num_classes=self.input_size).float()
		x, _ = self.lstm_encoder(x)

		if self.training:
			x = self.dropout(x)
			x = self.decoder(x)
			x = self.log_softmax(x)

			# target = self.one_hot_encoder(sequence[:,:-1]).float()
			# loss = self.loss(x[:,1:], target)
			return x
		else:
			return x
		

class CNN_Classifier(nn.Module):

	def __init__(self, params):
		super(CNN_Classifier, self).__init__()

		self.seq_length = params['max_seq_length']
		self.lstm_hidden_size = params['lstm_hidden_size']

		self.conv1 = nn.Conv2d(1, params['conv1_filters'], kernel_size=params['conv1_kernel_size'], padding=params['conv1_kernel_size']//2)
		self.maxpool1 = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(params['conv1_filters'], params['conv2_filters'],
			kernel_size=params['conv2_kernel_size'], padding=params['conv2_kernel_size']//2)
		self.maxpool2 = nn.MaxPool2d(2, stride=2)

		self.flatten = lambda x: x.view(x.size(0),-1) # not nn.Flatten because of compatability issue
		self.linear = nn.Linear(params['conv2_filters']*self.seq_length*self.lstm_hidden_size//16, 2)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):

		# assert(len(x.shape)==4)
		assert(x.shape[2] == self.seq_length)
		assert(x.shape[3] == self.lstm_hidden_size)

		x = self.conv1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.maxpool2(x)

		x = self.flatten(x)
		x = self.linear(x)
		# x = self.softmax(x)

		return x

class InsiderClassifier(nn.Module):

	def __init__(self, params, lstm_checkpoint):
		super(InsiderClassifier, self).__init__()

		self.lstm_encoder = LSTM_Encoder(params['lstm_encoder'])
		self.lstm_encoder.requires_grad = False
		self.lstm_encoder.eval()
		self.load_encoder(lstm_checkpoint)

		self.sigmoid = nn.Sigmoid()
		self.cnn_classifier = CNN_Classifier(params['cnn_classifier'])

	def train(self, mode=True):
		"""
		Customized train method. It restricts setting
		lstm_encoder to train mode
		"""
		self.training = mode
		self.sigmoid.train(mode)
		self.cnn_classifier.train(mode)
		return self

	# FIXME: device
	def load_encoder(self, checkpoint, device='cpu'):
		self.lstm_encoder.load_state_dict(
			torch.load(checkpoint, map_location=torch.device(device)), strict=True
			)
		return self

	def forward(self, x):
		"""
		x : batch of sequences of action tokens. All of them
		should be greater than minimum length and truncated
		"""
		with torch.no_grad():
			hidden_state = self.lstm_encoder(x)
			hidden_state = self.sigmoid(hidden_state)
		scores = self.cnn_classifier(hidden_state[:,None])

		return scores
