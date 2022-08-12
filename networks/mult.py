import torch
import torch.nn.functional as F

from torch import nn
from networks.pytorch_i3d import InceptionI3d
from networks.transformer import TransformerEncoder


class MULTModel(nn.Module):
	def __init__(self, config):
		'''
		Construct a MulT model.
		'''
		super(MULTModel, self).__init__()
		self.config = config

		self.orig_d_1, self.orig_d_2 = self.config.d_1, self.config.d_2

		self.d_1, self.d_2 = 30, 30

		self.num_heads = 5
		self.layers = 5
		self.attn_dropout = 0.1
		self.relu_dropout = 0.1
		self.res_dropout = 0.1
		self.out_dropout = 0.0
		self.embed_dropout = 0.25
		self.attn_mask = True

		combined_dim = self.d_1 + self.d_2
		output_dim = self.config.num_classes

		# 1. Temporal convolutional layers
		self.proj_1 = nn.Conv1d(self.orig_d_1, self.d_1, kernel_size=1, padding=0, bias=False)
		self.proj_2 = nn.Conv1d(self.orig_d_2, self.d_2, kernel_size=1, padding=0, bias=False)

		# 2. Crossmodal Attentions
		self.trans_1_with_2 = self.get_network(self_type='12')
		self.trans_2_with_1 = self.get_network(self_type='21')

		# 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
		self.trans_1_mem = self.get_network(self_type='1_mem', layers=3)
		self.trans_2_mem = self.get_network(self_type='2_mem', layers=3)

		# Projection layers
		self.proj1 = nn.Linear(combined_dim, combined_dim)
		self.proj2 = nn.Linear(combined_dim, combined_dim)
		self.out_layer = nn.Linear(combined_dim, output_dim)

		# Feature extraction
		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		for p in self.rgb_enc.parameters():
			p.requires_grad = False

		self.flow_enc = InceptionI3d(400, in_channels=2, dropout_rate=config.dropout)
		self.flow_enc.load_state_dict(torch.load('checkpoints/flow_imagenet.pt'))
		for p in self.flow_enc.parameters():
			p.requires_grad = False

	def get_network(self, self_type, layers=-1):
		if self_type == '12':
			embed_dim, attn_dropout = self.d_2, self.attn_dropout
		elif self_type == '21':
			embed_dim, attn_dropout = self.d_1, self.attn_dropout
		elif self_type == '1_mem':
			embed_dim, attn_dropout = self.d_1, self.attn_dropout
		elif self_type == '2_mem':
			embed_dim, attn_dropout = self.d_2, self.attn_dropout

		return TransformerEncoder(embed_dim=embed_dim,
								  num_heads=self.num_heads,
								  layers=max(self.layers, layers),
								  attn_dropout=attn_dropout,
								  relu_dropout=self.relu_dropout,
								  res_dropout=self.res_dropout,
								  embed_dropout=self.embed_dropout,
								  attn_mask=self.attn_mask)

	def get_features(self, x_1, x_2):
		for endpoint in self.rgb_enc.VALID_ENDPOINTS:
			if endpoint in self.rgb_enc.end_points:
				layer1 = self.rgb_enc._modules[endpoint]
				layer2 = self.flow_enc._modules[endpoint]
				x_1, x_2 = layer1(x_1), layer2(x_2)
				if endpoint == 'Logits':
					break

		return x_1, x_2

	def forward(self, x_1, x_2):
		x_1, x_2 = self.get_features(x_1, x_2)
		B, C, T, H, W = x_1.size()
		x_1 = x_1.view(B, C, -1)
		x_2 = x_2.view(B, C, -1)

		proj_x_1 = self.proj_1(x_1).permute(2, 0, 1)
		proj_x_2 = self.proj_2(x_2).permute(2, 0, 1)

		h_1 = self.trans_1_with_2(proj_x_1, proj_x_2, proj_x_2)
		h_1 = self.trans_1_mem(h_1)[0]

		h_2 = self.trans_2_with_1(proj_x_2, proj_x_1, proj_x_1)
		h_2 = self.trans_2_mem(h_2)[0]

		last_hs = torch.cat([h_1, h_2], dim=1)

		# A residual block
		last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout))
		last_hs_proj += last_hs

		output = self.out_layer(last_hs_proj)
		return output
