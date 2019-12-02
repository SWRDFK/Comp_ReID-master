import torch
from tools import *


def train_an_epoch(config, base, loaders):

	base.set_train()
	meter = MultiItemAverageMeter()

	batch_size = config.p * config.k
	num_batch = int(loaders.num_train / batch_size)

	for _ in range(num_batch):

		# load a batch data
		imgs, pids = loaders.train_iter.next_one()
		imgs, pids = imgs.to(base.device), pids.to(base.device)

		# forward
		# features, cls_score = base.model(imgs, config.model_name)
		features, cls_score = base.model(imgs)

		# CBL: use cb_loss and triplet_loss.
		if config.model_name == 'densenet161_CBL':
			ide_loss = base.cb_criterion(cls_score, pids)
			triplet_loss = base.triplet_criterion(features, features, features, pids, pids, pids)

			loss = ide_loss + triplet_loss
			acc = accuracy(cls_score, pids, [1])[0]

		# RLL: use ide_loss and ranked_list_loss.
		elif config.model_name == 'resnet101a_RLL':
			ide_loss = base.ide_criterion(cls_score, pids)
			triplet_loss = base.ranked_criterion(features, pids)

			loss = ide_loss + triplet_loss
			acc = accuracy(cls_score, pids, [1])[0]

		# SA: use ide_loss and triplet_loss.
		elif config.model_name == 'resnet101a_SA':
			ide_loss1 = base.ide_criterion(cls_score[0], pids)
			ide_loss2 = base.ide_criterion(cls_score[1], pids)
			ide_loss3 = base.ide_criterion(cls_score[2], pids)
			ide_loss4 = base.ide_criterion(cls_score[3], pids)

			ide_loss = ide_loss1 + ide_loss2 + ide_loss3 + ide_loss4
			triplet_loss = base.triplet_criterion(features, features, features, pids, pids, pids)

			loss = ide_loss + triplet_loss
			acc = accuracy(cls_score[3], pids, [1])[0]

		# optimize
		base.optimizer.zero_grad()
		loss.backward()
		base.optimizer.step()

		# record: ide_loss and triplet_loss
		meter.update({'ide_loss': ide_loss, 'triplet_loss': triplet_loss, 'acc': acc})

	return meter.get_val(), meter.get_str()