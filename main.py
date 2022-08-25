import os
import sys
import torch
import argparse

from solver import AR_solver
from solver_gb import AR_GB_solver
from solver_mult import AR_MulT_solver
from solver_misa import AR_MISA_solver
from utils import set_seed
from utils import get_data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

# data path
parser.add_argument('--csv_root', type=str, default='./dataset/epic_kitchens')
parser.add_argument('--data_root', type=str, default='./data/epic_kitchens')
parser.add_argument('--ckpt_path', type=str, default='./checkpoints')

# data
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--num_frames', type=int, default=16)

# architecture
parser.add_argument('--fusion', type=str)

# model
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--d_1', type=int, default=1024)
parser.add_argument('--d_2', type=int, default=1024)

# training
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--when', type=int, default=10, help='when to decay learning rate')
parser.add_argument('--patience', type=int, default=5, help='early stopping')

# MISA
parser.add_argument('--diff_weight', type=float, default=0.3)
parser.add_argument('--sim_weight', type=float, default=1.0)
parser.add_argument('--recon_weight', type=float, default=1.0)

# G-Blend
parser.add_argument('--num_gb_epochs', type=int, default=5)

# X-Norm
parser.add_argument('--weight', type=float, default=0.5)

opts = parser.parse_args()

# Fix random seed
set_seed(opts.seed)

# Setup model and data loader
if opts.fusion == 'gb':
	solver = AR_GB_solver(opts)
elif opts.fusion == 'mult':
	solver = AR_MulT_solver(opts).cuda()
elif opts.fusion == 'misa':
	solver = AR_MISA_solver(opts).cuda()
else:
	solver = AR_solver(opts).cuda()
train_loader, val_loader, test_loader = get_data_loaders(opts)

# Start training
solver.run(train_loader, val_loader, test_loader)
