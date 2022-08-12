import os
import sys
import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.model import GB


class AR_GB_solver(nn.Module):
	def __init__(self, config):
		super(AR_GB_solver, self).__init__()
		self.config = config

		# Initiate the networks
		self.model = GB(config)

		# Setup the optimizers and loss function
		opt_params = list(self.model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=config.when, factor=0.5, verbose=False)
		self.criterion = nn.CrossEntropyLoss()

		# Select the best ckpt
		self.best_val_metric = 0.

	def update(self, rgb_frames, flow_frames, labels, weights):
		self.train()
		self.optimizer.zero_grad()

		rgb_frames, flow_frames, labels = rgb_frames.cuda(), flow_frames.cuda(), labels.cuda()
		pred = self.model(rgb_frames, flow_frames)

		loss_0 = self.criterion(pred[0], labels)
		loss_1 = self.criterion(pred[1], labels)
		loss_2 = self.criterion(pred[2], labels)
		loss = loss_0*weights[0]+loss_1*weights[1]+loss_2*weights[2]
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

		self.optimizer.step()

	def val(self, val_loader, index):
		val_loss, val_acc = self.test(val_loader, index)
		self.save_best_ckpt(val_acc)
		self.scheduler.step(val_loss)

		return val_loss, val_acc

	def test(self, test_loader, index):
		with torch.no_grad():
			self.eval()
			preds, gt = [], []
			total_loss, total_samples = 0.0, 0
			for (rgb_frames, flow_frames, labels) in test_loader:
				rgb_frames, flow_frames, labels = rgb_frames.cuda(), flow_frames.cuda(), labels.cuda()
				pred = self.model(rgb_frames, flow_frames)
				pred = pred[index]
				loss = self.criterion(pred, labels)
				_, pred = torch.max(pred, 1)
				preds.append(pred)
				gt.append(labels)

				total_loss += loss.item()*labels.size(0)
				total_samples += labels.size(0)

			preds, gt = torch.cat(preds).cpu(), torch.cat(gt).cpu()
			acc = accuracy_score(np.array(gt), np.array(preds))
			loss = total_loss / total_samples

			self.print_metric([loss, acc])

			return loss, acc

	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.fusion+'_'+str(self.config.seed)+'.pt')
		state_dict = torch.load(ckpt_name)

		self.model.load_state_dict(state_dict['model'])

	def save_best_ckpt(self, val_metric):
		def update_metric(val_metric):
			if val_metric > self.best_val_metric:
				self.best_val_metric = val_metric
				return True
			return False

		if update_metric(val_metric):
			ckpt_name = os.path.join(self.config.ckpt_path, self.config.fusion+'_'+str(self.config.seed)+'.pt')
			torch.save({'model': self.model.state_dict()}, ckpt_name)

	def print_metric(self, metric):
		print('Loss: %.4f Acc: %.3f'%(metric[0], metric[1]))

	def gb_val(self, model, loader, index):
		with torch.no_grad():
			model.eval()
			total_loss, total_samples = 0.0, 0
			for (rgb_frames, flow_frames, labels) in loader:
				rgb_frames, flow_frames, labels = rgb_frames.cuda(), flow_frames.cuda(), labels.cuda()
				pred = model(rgb_frames, flow_frames)
				pred = pred[index]
				loss = self.criterion(pred, labels)
				_, pred = torch.max(pred, 1)

				total_loss += loss.item()*labels.size(0)
				total_samples += labels.size(0)

			loss = total_loss / total_samples

			return loss

	def gb_train(self, model, optimizer, idx):
		model.train()
		for epoch in range(self.config.num_gb_epochs):
			print('Epoch: %d/%d' % (epoch+1, self.config.num_gb_epochs))
			for _, (rgb_frames, flow_frames, labels) in tqdm(enumerate(self.tt_loader), total=len(self.tt_loader)):
				optimizer.zero_grad()
				rgb_frames, flow_frames, labels = rgb_frames.cuda(), flow_frames.cuda(), labels.cuda()

				preds = model(rgb_frames, flow_frames)
				loss = self.criterion(preds[idx], labels)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip)
				optimizer.step()

		ltNn = self.gb_val(model, loader=self.tt_loader, index=idx)
		lvNn = self.gb_val(model, loader=self.tv_loader, index=idx)

		oNn = lvNn-ltNn
		if oNn < 0:
			oNn = 0.0001

		return abs(lvNn/(oNn**2))

	def gb_estimate(self, model):
		weights = []
		for modal_idx in range(2):
			print("At gb_estimate unimodal "+str(modal_idx))
			uni_model = copy.deepcopy(model).cuda()
			uni_params = list(uni_model.parameters())
			uni_optim = torch.optim.AdamW(uni_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
			w = self.gb_train(uni_model, uni_optim, modal_idx)
			weights.append(w)

		print("At gb_estimate multimodal ")
		tri_model = copy.deepcopy(model).cuda()
		tri_params = list(tri_model.parameters())
		tri_optim = torch.optim.AdamW(tri_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
		w = self.gb_train(tri_model, tri_optim, 2)
		weights.append(w)

		return weights/np.sum(np.array(weights))

	def run(self, train_loader, val_loader, test_loader):
		v_rate = 0.1
		train_datas = train_loader.dataset
		splitloc = int(len(train_datas)*v_rate)
		inds = list(range(len(train_datas)))
		t_inds = inds[splitloc:]
		v_inds = inds[:splitloc]
		tt_data = Subset(train_datas, t_inds)
		tv_data = Subset(train_datas, v_inds)

		self.tt_loader = DataLoader(
			dataset=tt_data,
			shuffle=True,
			drop_last=True,
			batch_size=train_loader.batch_size,
			num_workers=train_loader.num_workers)

		self.tv_loader = DataLoader(
			dataset=tv_data,
			shuffle=False,
			drop_last=True,
			batch_size=train_loader.batch_size,
			num_workers=train_loader.num_workers)

		#weights = self.gb_estimate(self.model)
		#print("weights: " + str(weights))
		if self.config.seed == 1:
			weights = [0.02495882,0.95270437,0.02233681]
		elif self.config.seed == 2:
			weights = [0.08169739,0.84437539,0.07392722]
		else:
			weights = self.gb_estimate(self.model)
		print("weights: " + str(weights))
		self.model = self.model.cuda()

		best_val_loss = 1e8
		patience = self.config.patience
		for epochs in range(1, self.config.num_epochs+1):
			print('Epoch: %d/%d' % (epochs, self.config.num_epochs))
			for _, (rgb_frames, flow_frames, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
				self.update(rgb_frames, flow_frames, labels, weights)

			# Validate model
			val_loss, val_acc = self.val(val_loader, index=2)

			if val_loss < best_val_loss:
				patience = self.config.patience
				best_val_loss = val_loss
			else:
				patience -= 1
				if patience == 0:
					break

		# Test model
		self.load_best_ckpt()
		self.test(test_loader, index=2)
