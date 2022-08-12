import random
import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Function
from networks.pytorch_i3d import InceptionI3d

class MISA(nn.Module):
	def __init__(self, config):
		super(MISA, self).__init__()
		self.config = config
		self.size_1 = self.config.d_1
		self.size_2 = self.config.d_2

		self.input_sizes = input_sizes = [self.size_1, self.size_2]
		self.hidden_sizes = hidden_sizes = [self.size_1, self.size_2]
		self.output_size = output_size = config.num_classes
		self.dropout_rate = dropout_rate = config.dropout
		self.activation = nn.ReLU()

		# defining modules - two layer bidirectional LSTM with layer norm in between
		self.rnn_11 = nn.LSTM(input_sizes[0], hidden_sizes[0], bidirectional=True)
		self.rnn_12 = nn.LSTM(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)

		self.rnn_21 = nn.LSTM(input_sizes[1], hidden_sizes[1], bidirectional=True)
		self.rnn_22 = nn.LSTM(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)

		##########################################
		# mapping modalities to same sized space
		##########################################
		self.project_1 = nn.Sequential(
			nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size),
			self.activation,
			nn.LayerNorm(config.hidden_size),
		)

		self.project_2 = nn.Sequential(
			nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size),
			self.activation,
			nn.LayerNorm(config.hidden_size),
		)

		##########################################
		# private encoders
		##########################################
		self.private_1 = nn.Sequential(
			nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size),
			nn.Sigmoid(),
		)

		self.private_2 = nn.Sequential(
			nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size),
			nn.Sigmoid(),
		)

		##########################################
		# shared encoder
		##########################################
		self.shared = nn.Sequential(
			nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size),
			nn.Sigmoid(),
		)

		##########################################
		# reconstruct
		##########################################
		self.recon_1 = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
		self.recon_2 = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

		##########################################
		# fusion
		##########################################
		self.fusion = nn.Sequential(
			nn.Linear(in_features=self.config.hidden_size*4, out_features=self.config.hidden_size*2),
			nn.Dropout(dropout_rate),
			self.activation,
			nn.Linear(in_features=self.config.hidden_size*2, out_features=output_size),
		)

		self.layer_norm_1 = nn.LayerNorm((hidden_sizes[0]*2,))
		self.layer_norm_2 = nn.LayerNorm((hidden_sizes[1]*2,))

		encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

		# Feature extraction
		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		for p in self.rgb_enc.parameters():
			p.requires_grad = False

		self.flow_enc = InceptionI3d(400, in_channels=2, dropout_rate=config.dropout)
		self.flow_enc.load_state_dict(torch.load('checkpoints/flow_imagenet.pt'))
		for p in self.flow_enc.parameters():
			p.requires_grad = False

	def extract_features(self, sequence, rnn1, rnn2, layer_norm):
		sequence = sequence.permute(2, 0, 1)
		h, (final_h1, _) = rnn1(sequence)
		normed_h = layer_norm(h)
		_, (final_h2, _) = rnn2(normed_h)

		return final_h1, final_h2

	def get_features(self, x_1, x_2):
		for endpoint in self.rgb_enc.VALID_ENDPOINTS:
			if endpoint in self.rgb_enc.end_points:
				layer1 = self.rgb_enc._modules[endpoint]
				layer2 = self.flow_enc._modules[endpoint]
				x_1, x_2 = layer1(x_1), layer2(x_2)
				if endpoint == 'Logits':
					break

		return x_1, x_2

	def run(self, x_1, x_2):
		x_1, x_2 = self.get_features(x_1, x_2)
		B, C, T, H, W = x_1.size()
		x_1 = x_1.view(B, C, -1)
		x_2 = x_2.view(B, C, -1)

		batch_size = x_1.size(0)

		h11, h12 = self.extract_features(x_1, self.rnn_11, self.rnn_12, self.layer_norm_1)
		x_1 = torch.cat((h11, h12), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

		h21, h22 = self.extract_features(x_2, self.rnn_21, self.rnn_22, self.layer_norm_2)
		x_2 = torch.cat((h21, h22), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

		# Shared-private encoders
		self.shared_private(x_1, x_2)

		# For reconstruction
		self.reconstruct()

		# 1-LAYER TRANSFORMER FUSION
		h = torch.stack((self.x_1_private, self.x_2_private, self.x_1_shared, self.x_2_shared), dim=0)
		h = self.transformer_encoder(h)
		h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
		o = self.fusion(h)
		return o

	def reconstruct(self):
		x_1 = (self.x_1_private + self.x_1_shared)
		x_2 = (self.x_2_private + self.x_2_shared)

		self.x_1_recon = self.recon_1(x_1)
		self.x_2_recon = self.recon_2(x_2)

	def shared_private(self, x_1, x_2):
		# Projecting to same sized space
		self.x_1_orig = x_1 = self.project_1(x_1)
		self.x_2_orig = x_2 = self.project_2(x_2)

		# Private-shared components
		self.x_1_private = self.private_1(x_1)
		self.x_2_private = self.private_2(x_2)

		self.x_1_shared = self.shared(x_1)
		self.x_2_shared = self.shared(x_2)

	def forward(self, x_1, x_2):
		return self.run(x_1, x_2)
