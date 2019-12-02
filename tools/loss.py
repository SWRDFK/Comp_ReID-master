import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math
import numpy as np
from .metric import *


class CrossEntropyLabelSmooth(nn.Module):
	'''
	Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	'''

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		'''
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		'''
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
		if self.use_gpu:
			targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()

		return loss


class RankingLoss:

	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
												descending=True)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
												descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n


class TripletLoss(RankingLoss):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, metric):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.metric = metric

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''
		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''
		if self.metric == 'cosine':
			mat_dist = cosine_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			mat_dist = cosine_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			margin_label = -torch.ones_like(hard_p)

		elif self.metric == 'euclidean':
			mat_dist = euclidean_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			mat_dist = euclidean_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			margin_label = torch.ones_like(hard_p)

		return self.margin_loss(hard_n, hard_p, margin_label)


class CBLoss(nn.Module):

	def __init__(self, num_classes, samples_per_class, gamma=2, alpha=None, size_average=True):
		super(CBLoss, self).__init__()
		self.num_classes = num_classes
		self.samples_per_class = samples_per_class
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)):
			self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list):
			self.alpha = torch.Tensor(alpha)
		self.size_average = size_average
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		if inputs.dim() > 2:
			inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  	# N,C,H,W => N,C,H*W
			inputs = inputs.transpose(1, 2)  							# N,C,H*W => N,H*W,C
			inputs = inputs.contiguous().view(-1, inputs.size(2))		# N,H*W,C => N*H*W,C

		targets = targets.view(-1, 1)

		logpt = self.logsoftmax(inputs)
		logpt = logpt.gather(1, targets)
		logpt = logpt.view(-1)

		pt = logpt.exp()

		if self.alpha is not None:
			if self.alpha.type() != inputs.data.type():
				self.alpha = self.alpha.type_as(inputs.data)
			at = self.alpha.gather(0, targets.data.view(-1))
			logpt = logpt * at

		# compute weights by the number of each class
		beta = 1.0 - 1.0 / np.array(self.samples_per_class)
		effective_num = 1.0 - np.power(beta, self.samples_per_class)
		weights = (1.0 - beta) / np.array(effective_num)
		weights = weights / np.sum(weights) * self.num_classes

		batch_weights = torch.Tensor([weights[i] for i in targets]).cuda()
		loss = -1 * (1 - pt) ** self.gamma * batch_weights * logpt

		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()


class RankedListLoss(nn.Module):

	def __init__(self, margin=None, alpha=None, tval=None):
		self.margin = margin
		self.alpha = alpha
		self.tval = tval

	def normalize_rank(self, x, axis=-1):
		"""
		Normalizing to unit length along the specified dimension.
		Args:
		  x: pytorch Variable
		Returns:
		  x: pytorch Variable, same shape as input
		"""
		x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
		return x

	def euclidean_dist_rank(self, x, y):
		"""
		Args:
		  x: pytorch Variable, with shape [m, d]
		  y: pytorch Variable, with shape [n, d]
		Returns:
		  dist: pytorch Variable, with shape [m, n]
		"""
		m, n = x.size(0), y.size(0)
		xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
		yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
		dist = xx + yy
		dist.addmm_(1, -2, x, y.t())
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
		return dist

	def rank_loss(self, dist_mat, labels, margin, alpha, tval):
		"""
		Args:
		  dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
		  labels: pytorch LongTensor, with shape [N]
		"""
		assert len(dist_mat.size()) == 2
		assert dist_mat.size(0) == dist_mat.size(1)
		N = dist_mat.size(0)

		total_loss = 0.0
		for ind in range(N):
			is_pos = labels.eq(labels[ind])
			is_pos[ind] = 0
			is_neg = labels.ne(labels[ind])

			dist_ap = dist_mat[ind][is_pos]
			dist_an = dist_mat[ind][is_neg]

			ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
			ap_pos_num = ap_is_pos.size(0) + 1e-5
			ap_pos_val_sum = torch.sum(ap_is_pos)
			loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

			an_is_pos = torch.lt(dist_an, alpha)
			an_less_alpha = dist_an[an_is_pos]
			an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
			an_weight_sum = torch.sum(an_weight) + 1e-5
			an_dist_lm = alpha - an_less_alpha
			an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
			loss_an = torch.div(an_ln_sum, an_weight_sum)

			total_loss = total_loss + loss_ap + loss_an
		total_loss = total_loss * 1.0 / N
		return total_loss

	def __call__(self, global_feat, labels, normalize_feature=True):
		if normalize_feature:
			global_feat = self.normalize_rank(global_feat, axis=-1)
		dist_mat = self.euclidean_dist_rank(global_feat, global_feat)
		total_loss = self.rank_loss(dist_mat, labels, self.margin, self.alpha, self.tval)

		return total_loss