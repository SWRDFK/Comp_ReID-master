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
		features, cls_score = base.model(imgs)

		# use ide_loss and triplet_loss at the same time, but can replace them with cb_loss and ranked_loss, respectively.

		# # Baseline
		# ide_loss = base.ide_criterion(cls_score, pids)
		# triplet_loss = base.triplet_criterion(features, features, features, pids, pids, pids)
        #
		# # ide_loss = base.cb_criterion(cls_score, pids)
		# # triplet_loss = base.ranked_criterion(features, pids)
        #
		# loss = ide_loss + triplet_loss
		# acc = accuracy(cls_score, pids, [1])[0]


		# Spatial Attention
		ide_loss1 = base.ide_criterion(cls_score[0], pids)
		ide_loss2 = base.ide_criterion(cls_score[1], pids)
		ide_loss3 = base.ide_criterion(cls_score[2], pids)
		ide_loss4 = base.ide_criterion(cls_score[3], pids)

		ide_loss = ide_loss1 + ide_loss2 + ide_loss3 + ide_loss4
		triplet_loss = base.triplet_criterion(features, features, features, pids, pids, pids)

		# cb_loss1 = base.cb_criterion(cls_score[0], pids)
		# cb_loss2 = base.cb_criterion(cls_score[1], pids)
		# cb_loss3 = base.cb_criterion(cls_score[2], pids)
		# cb_loss4 = base.cb_criterion(cls_score[3], pids)
		# ide_loss = cb_loss1 + cb_loss2 + cb_loss3 + cb_loss4
		# triplet_loss = base.ranked_criterion(features, pids)

		loss = ide_loss + triplet_loss
		acc = accuracy(cls_score[3], pids, [1])[0]


		# optimize
		base.optimizer.zero_grad()
		loss.backward()
		base.optimizer.step()

		# record: ide_loss and triplet_loss
		meter.update({'ide_loss': ide_loss, 'triplet_loss': triplet_loss, 'acc': acc})

	return meter.get_val(), meter.get_str()