import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import math

def ModelConstructor(config):
	dim = config['feat_dim'] * (config['num_parts']+1)
	cnn = timm.create_model(config['model'], pretrained = config['pretrained'], num_classes = dim)
	model = ContrastiveWrapper(cnn, dim, config)
	return model


# linear mapping that enforces same-sign weights
class SameSignLinear(nn.Module):
	def __init__(self, in_features, out_features, sign = 1, weight = None, bias = 0.0):
		super(SameSignLinear, self).__init__()
		if weight is None:
			self.weight = nn.Parameter(torch.empty(out_features, in_features))
			nn.init.kaiming_uniform_(self.weight)
		elif type(weight) in [float, int]:
			assert in_features == out_features
			assert in_features == 1
			self.weight = nn.Parameter(torch.Tensor([[float(weight)]]))
		else:
			assert type(weight) == torch.Tensor
			self.weight = nn.Parameter(weight)
		self.bias = nn.Parameter(torch.Tensor([bias]))
		self.sign = sign

	def forward(self, x):
		return F.linear(x, self.sign*self.weight.abs(), self.bias)


class ContrastiveWrapper(nn.Module):
	def __init__(self, cnn, cnn_dim, config):
		super(ContrastiveWrapper, self).__init__()
		self.cnn = cnn
		self.feat_dim = config['feat_dim']
		
		self.cnn_dim = cnn_dim
		self.num_heads = config['num_parts']+1
		
		if config['metric'] == 'L2':
			self.calc_similarity_scores = _calc_pairwise_L2_per_part
		elif config['metric'] == 'cos':
			self.calc_similarity_scores = _calc_pairwise_cos_per_part
		else:
			raise ValueError(config['metric'])
		
		if config['metric_scaler'] == 'batchnorm':
			self.metric_scaler = nn.BatchNorm1d(1)
			self.get_metric_scale = self._get_metric_scale_bn
		elif config['metric_scaler'] == 'linear':
			# this covers learnt temperature ("thermostat")
			self.metric_scaler = SameSignLinear(1, 1)
			self.get_metric_scale = self._get_metric_scale_linear
		elif type(config['metric_scaler']) in [int, float]:
			# this covers fixed temperature hyperparameters
			self.metric_scaler = lambda x : x*config['metric_scaler']
			self.get_metric_scale = lambda : config['metric_scaler']
		elif config['metric_scaler'] == 'none':
			self.metric_scaler = nn.Identity()
			self.get_metric_scale = self._get_metric_scale_none
		else:
			raise ValueError(config['metric_scaler'])
		
	
	# input: A x D*
	# output: H x A x D
	def forward(self, x):
		# A x H*D -> A x H x D -> H x A x D
		features = self.cnn(x).reshape(len(x), self.num_heads,-1).transpose(0,1)
		return features
	
	# for scaling margins
	def _get_metric_scale_bn(self):
		return self.metric_scaler.weight.detach()
	
	def _get_metric_scale_linear(self):
		return self.metric_scaler.weight.abs().detach()
	
	def _get_metric_scale_none(self):
		return 1


# for calculating (pairwise) similarity metrics
# features: A x D
# returns: A x A
def _calc_pairwise_L2(batch_dict):
	return -torch.cdist(batch_dict['features'], batch_dict['features'])

# features: A x D
# returns: A x A
def _calc_pairwise_cos(batch_dict):
	return F.cosine_similarity(	batch_dict['features'].unsqueeze(-3),
								batch_dict['features'].unsqueeze(-2), dim=-1)


# this extracts the similarity from the relevant head
# H x A x A -> A x A
def __extract_scores_from_heads(id2part, scores):
	num_heads = len(scores)
	# make mask to select appropriate pairs from each head
	# 1 if valid, 0 otherwise
	# each item should only be valid in exactly 1 head
	# A x A
	Xpart_mask = (id2part.unsqueeze(-1) != id2part.unsqueeze(0))
	# (H-1) x A x 1
	part_masks = [(id2part == i).unsqueeze(-1) for i in range(num_heads-1)]
	# H x A x A
	head_masks = [(part_mask * part_mask.T) for part_mask in part_masks] + [Xpart_mask]
	head_masks = torch.stack(head_masks, 0).float()
	# A x A
	scores = (scores * head_masks).sum(0)
	return scores

# features: H x A x D
# returns: A x A
def _calc_pairwise_L2_per_part(features, id2part):
	id2part = id2part.to(features.device, non_blocking=True)
	
	# H x A x A
	scores = -torch.cdist(features, features)
	
	# A x A
	scores = __extract_scores_from_heads(id2part, scores)
	return scores

def _calc_pairwise_cos_per_part(features, id2part):
	id2part = id2part.to(features.device, non_blocking=True)
	
	# H x A x A
	scores = F.cosine_similarity(	features.unsqueeze(-3),
									features.unsqueeze(-2), dim=-1)
	
	# A x A
	scores = __extract_scores_from_heads(id2part, scores)
	return scores
