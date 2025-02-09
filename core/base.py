import sys
sys.path.append('..')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
from tools import os_walk, CrossEntropyLabelSmooth, TripletLoss, CBLoss, RankedListLoss
from .densenet import *
from .resnet_factory import *
from .resnet_SA import *


class Base:

	def __init__(self, config, loaders):

		self.config = config
		self.loaders = loaders

		# Data Configuration
		self.pid_num = config.pid_num
		self.samples_per_class = loaders.samples_per_class

		# Loss Configuration
		self.margin = config.margin

		# Logger Configuration
		self.max_save_model_num = config.max_save_model_num
		self.output_path = config.output_path
		self.model_name = config.model_name
		self.save_dist_path = os.path.join(self.output_path, 'dists/')
		self.save_json_path = os.path.join(self.output_path, 'jsons/')
		self.save_model_path = os.path.join(self.output_path, os.path.join(self.model_name, 'models/'))
		self.save_log_path = os.path.join(self.output_path, os.path.join(self.model_name, 'logs/'))

		# Train Configuration
		self.base_learning_rate = config.base_learning_rate
		self.weight_decay = config.weight_decay
		self.milestones = config.milestones

		# Init Model
		self._init_device()

		if self.model_name == "densenet161_CBL":
			self._init_model1()
		if self.model_name == "resnet101a_RLL":
			self._init_model2()
		if self.model_name == "resnet101a_SA":
			self._init_model3()

		self._init_criterion()
		self._init_optimizer()


	def _init_device(self):
		self.device = torch.device('cuda')


	def _init_model1(self):
		# For densenet161_CBL
		self.model = densenet161(num_classes=self.pid_num, pretrained=True)
		self.model = nn.DataParallel(self.model).to(self.device)


	def _init_model2(self):
		# For resnet101a_RLL
		self.model = ResNet('101a', num_classes=self.pid_num)
		self.model = nn.DataParallel(self.model).to(self.device)


	def _init_model3(self):
		# For resnet101a_SA
		self.model = ResNet_SA('101a', num_classes=self.pid_num)
		self.model = nn.DataParallel(self.model).to(self.device)


	def _init_criterion(self):
		self.ide_criterion = CrossEntropyLabelSmooth(self.pid_num)
		self.triplet_criterion = TripletLoss(self.margin, 'euclidean')
		self.cb_criterion = CBLoss(self.pid_num, self.samples_per_class, gamma=2)
		self.ranked_criterion = RankedListLoss(margin=1.3, alpha=2.0, tval=1.0)


	def _init_optimizer(self):

		params = []
		for key, value in self.model.named_parameters():
			if not value.requires_grad:
				continue
			lr = self.base_learning_rate
			weight_decay = self.weight_decay
			params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
		self.optimizer = optim.Adam(params)

		self.lr_scheduler = WarmupMultiStepLR(self.optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)


	# save model as save_epoch
	def save_model(self, save_epoch):

		# save model
		file_path = os.path.join(self.save_model_path, 'model_{}.pkl'.format(save_epoch))
		torch.save(self.model.state_dict(), file_path)

		# if saved model is more than max num, delete the model with smallest iter
		if self.max_save_model_num > 0:
			root, _, files = os_walk(self.save_model_path)
			if len(files) > self.max_save_model_num:
				file_iters = sorted([int(file.replace('.pkl', '').split('_')[1]) for file in files], reverse=False)
				file_path = os.path.join(root, 'model_{}.pkl'.format(file_iters[0]))
				os.remove(file_path)


	# resume model from resume_epoch
	def resume_model(self, resume_epoch):
		model_path = os.path.join(self.save_model_path, 'model_{}.pkl'.format(resume_epoch))
		self.model.load_state_dict(torch.load(model_path))
		print(('Successfully resume model from {}'.format(model_path)))


	# set model as train mode
	def set_train(self):
		self.model = self.model.train()


	# set model as eval mode
	def set_eval(self):
		self.model = self.model.eval()


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear", last_epoch=-1):
		if not list(milestones) == sorted(milestones):
			raise ValueError(
				"Milestones should be a list of" " increasing integers. Got {}",
				milestones,
			)

		if warmup_method not in ("constant", "linear"):
			raise ValueError(
				"Only 'constant' or 'linear' warmup_method accepted"
				"got {}".format(warmup_method)
			)
		self.milestones = milestones
		self.gamma = gamma
		self.warmup_factor = warmup_factor
		self.warmup_iters = warmup_iters
		self.warmup_method = warmup_method
		super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		warmup_factor = 1
		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == "constant":
				warmup_factor = self.warmup_factor
			elif self.warmup_method == "linear":
				alpha = float(self.last_epoch) / float(self.warmup_iters)
				warmup_factor = self.warmup_factor * (1 - alpha) + alpha
		return [
			base_lr
			* warmup_factor
			* self.gamma ** bisect_right(self.milestones, self.last_epoch)
			for base_lr in self.base_lrs
		]
