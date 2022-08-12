import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

import videotransforms

action_list = ['put-down','pick-up','open','rinse','take','close','turn-on','move','turn-off','get']


def sample_frames(start_frame, stop_frame, num_frames=8):
	frame_list = range(start_frame, stop_frame+1)
	return [frame_list[i] for i in np.linspace(0, len(frame_list)-1, num_frames).astype('int')]


def load_rgb_frames(rgb_root, start_frame, stop_frame, num_frames):
	frames = []
	frame_list = sample_frames(start_frame, stop_frame, num_frames)
	for frame_idx in frame_list:
		frame_name = 'frame_'+str(frame_idx).zfill(10)+'.jpg'
		img = cv2.imread(os.path.join(rgb_root, frame_name))[:, :, [2, 1, 0]]
		w,h,c = img.shape
		if w < 226 or h < 226:
			d = 226.-min(w,h)
			sc = 1+d/min(w,h)
			img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
		img = (img / 255.)*2 - 1
		frames.append(img)

	return np.asarray(frames, dtype=np.float32)


def load_flow_frames(flow_u_root, flow_v_root, start_frame, stop_frame, num_frames):
	frames = []
	frame_list = sample_frames(start_frame, stop_frame, num_frames)
	for frame_idx in frame_list:
		frame_name = 'frame_'+str(frame_idx).zfill(10)+'.jpg'
		img_u = cv2.imread(os.path.join(flow_u_root, frame_name), cv2.IMREAD_GRAYSCALE)
		img_v = cv2.imread(os.path.join(flow_v_root, frame_name), cv2.IMREAD_GRAYSCALE)
		w,h = img_u.shape
		if w < 224 or h < 224:
			d = 224.-min(w,h)
			sc = 1+d/min(w,h)
			img_u = cv2.resize(img_u,dsize=(0,0),fx=sc,fy=sc)
			img_v = cv2.resize(img_v,dsize=(0,0),fx=sc,fy=sc)

		img_u = (img_u / 255.) * 2 - 1
		img_v = (img_v / 255.) * 2 - 1
		img = np.asarray([img_u, img_v]).transpose([1,2,0])
		frames.append(img)

	return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
	"""Convert a ``numpy.ndarray`` to tensor.
	Converts a numpy.ndarray (T x H x W x C)
	to a torch.FloatTensor of shape (C x T x H x W)
	
	Args:
		 pic (numpy.ndarray): Video to be converted to tensor.
	Returns:
		 Tensor: Converted video.
	"""
	return torch.from_numpy(pic.transpose([3,0,1,2]))


class EPIC_Kitchens(data.Dataset):
	def __init__(self, csv_path, data_root, train, num_frames):
		self.dataset = pd.read_csv(csv_path)
		self.data_root = data_root
		self.num_frames = num_frames

		if train:
			self.transform = transforms.Compose([
				videotransforms.RandomCrop(224),
				videotransforms.RandomHorizontalFlip()])
		else:
			self.transform = transforms.Compose([videotransforms.CenterCrop(224)])

	def __getitem__(self, index):
		participant_id = self.dataset['participant_id'][index]
		video_id = self.dataset['video_id'][index]
		start_frame = int(self.dataset['start_frame'][index])
		stop_frame = int(self.dataset['stop_frame'][index])

		rgb_root = os.path.join(self.data_root, participant_id, 'rgb_frames', video_id)
		rgb_frames = load_rgb_frames(rgb_root, start_frame, stop_frame, self.num_frames)
		rgb_frames = video_to_tensor(self.transform(rgb_frames))

		flow_u_root = os.path.join(self.data_root, participant_id, 'flow_frames', video_id, 'u')
		flow_v_root = os.path.join(self.data_root, participant_id, 'flow_frames', video_id, 'v')
		flow_frames = load_flow_frames(flow_u_root, flow_v_root, (start_frame+1)//2, (stop_frame+1)//2, self.num_frames)
		flow_frames = video_to_tensor(self.transform(flow_frames))

		label = int(self.dataset['verb'][index])

		return rgb_frames, flow_frames, label

	def __len__(self):
		return len(self.dataset)
