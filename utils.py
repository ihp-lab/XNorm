import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision.utils as vutils
from torchvision import transforms

from data import EPIC_Kitchens


def set_seed(seed):
	# Reproducibility
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

	random.seed(seed)
	np.random.seed(seed)


def get_data_loaders(config):
	csv_root = config.csv_root
	train_csv = os.path.join(csv_root, 'train.csv')
	val_csv = os.path.join(csv_root, 'val.csv')
	test_csv = os.path.join(csv_root, 'test.csv')

	data_root = config.data_root
	batch_size = config.batch_size
	num_frames = config.num_frames
	num_workers = config.num_workers

	train_loader = get_data_loader(train_csv, data_root, batch_size, True, num_frames, num_workers)
	val_loader = get_data_loader(val_csv, data_root, batch_size, False, num_frames, num_workers)
	test_loader = get_data_loader(test_csv, data_root, batch_size, False, num_frames, num_workers)

	return train_loader, val_loader, test_loader


def get_data_loader(csv_path, data_root, batch_size, train, num_frames=8, num_workers=4):
	dataset = EPIC_Kitchens(csv_path, data_root, train, num_frames)
	loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train,
						drop_last=True, num_workers=num_workers)

	return loader
