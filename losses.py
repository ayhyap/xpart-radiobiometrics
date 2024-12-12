import torch
import torch.nn as nn
import torch.nn.functional as F

def all_pairs_CE_loss( 
		scores,	# A x A
		batch_dict,
		margin_scale,
		config
	):
	device = scores.device
	A = len(scores)
	
	if config['margin']:
		# apply margin to positive scores
		# eq. to Large Margin Cosine Loss for cosine metric
		# A	
		id2patient = batch_dict['id2patient'].to(device)
		# A x A
		labels = (id2patient.unsqueeze(0) == id2patient.unsqueeze(1)).float()
		scores = scores - config['margin']*margin_scale*labels
	else:
		# A
		id2patient = batch_dict['id2patient'].to(device, non_blocking=True)
	
	## generate positives+negatives
	index = torch.arange(len(id2patient), device=device)
	
	# A x var(G)
	positives = [index[id2patient==patient] for patient in id2patient]
	negatives = [index[id2patient!=patient] for patient in id2patient]
	
	# A x G_max
	positives = nn.utils.rnn.pad_sequence(positives, batch_first=True, padding_value=-1)+1
	negatives = nn.utils.rnn.pad_sequence(negatives, batch_first=True, padding_value=-1)+1
	
	# concat padding for indexing
	# A x A+1
	scores = F.pad(scores, (1,0,0,0))
	
	# separate positives/negatives by indexing
	temp = torch.arange(len(scores), device=device).unsqueeze(-1)
	
	pos_mask = positives.bool()
	neg_mask = negatives.bool()
	
	_,P = positives.shape
	_,N = negatives.shape
	
	# A x G_max
	pos = scores[temp,positives]
	neg = scores[temp,negatives]
	
	# mask out padding values so they don't contribute to loss
	# masking with inf or -inf makes nan loss, so mask with a big number instead
	pos = pos.masked_fill(~pos_mask, 1e4)
	neg = neg.masked_fill(~neg_mask, -1e4)
	
	# (A,P) -> (A,P,1)
	# (A,N) -> (A,P,N)
	pos = pos.unsqueeze(-1)
	neg = neg.unsqueeze(-2).expand(-1,P,-1)
	
	# (A,P,1+N)
	# positive labels are 0-th entry along last dimension
	out = torch.cat((pos,neg), dim=-1)
	# (A*P,1+N)
	out = out.reshape(A*P, 1+N)
	
	# (A*P)
	labels = torch.zeros(len(out), dtype=int, device=device)
	# A x P
	loss = F.cross_entropy(out, labels, reduction='none').reshape(A,P)
	loss = loss.sum() / pos_mask.sum()
	return loss


def all_pairs_BCE_loss(
		scores,	# A x A
		batch_dict,
		margin_scale,
		config
	):
	device = scores.device
	
	# 1 x A
	id2patient = batch_dict['id2patient'].unsqueeze(0).to(device)
	
	# A x A
	labels = (id2patient == id2patient.T).float()
	
	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
	
	return loss


def all_pairs_margin_loss( 
		scores,	# A x A
		triplet_dict,
		margin_scale,
		config
	):
	device = scores.device
	
	# 1 x A
	id2patient = triplet_dict['id2patient'].unsqueeze(0).to(device)
	
	# A x A
	labels = (id2patient == id2patient.T).float()
	
	# binary labels to +/-1 mask
	# [0,1] -> [-1,1]
	labels_mp = (labels*2-1)
	
	
	## negate positive pairs
	# neg: s(a,b)
	# pos:-s(a,b)
	loss = -labels_mp * scores
	
	## apply margin
	# margin is the weight of the metric scaler, or 1
	# this is to prevent the metric scaler's weight parameter from overriding the margin 
	# neg: m+s(a,b)
	# pos: m-s(a,b)
	loss = margin_scale + loss
	
	## relu
	# neg: relu(m+s(a,b))
	# pos: relu(m-s(a,b))
	loss = torch.relu(loss)
	# naiive hard negative mining
	counts = (loss > 0).sum()
	loss = loss.sum() / counts.clamp(min=1)
	return loss


def all_triplet_loss( 
		scores,	# A x A
		triplet_dict,
		margin_scale,
		config
	):
	device = scores.device
	
	## generate positives+negatives
	id2patient = batch_dict['id2patient'].to(device)
	index = torch.arange(len(id2patient), device=device)
	
	# A x var(G)
	positives = [index[id2patient==patient] for patient in id2patient]
	negatives = [index[id2patient!=patient] for patient in id2patient]
	
	# A x G_max
	positives = nn.utils.rnn.pad_sequence(positives, batch_first=True, padding_value=-1)+1
	negatives = nn.utils.rnn.pad_sequence(negatives, batch_first=True, padding_value=-1)+1
	
	# concat padding for indexing
	# A x A+1
	scores = F.pad(scores, (1,0,0,0))
	
	# separate positives/negatives by indexing
	temp = torch.arange(len(scores), device=device).unsqueeze(-1)
	
	pos_mask = positives.bool()
	neg_mask = negatives.bool()
	
	A,P = positives.shape
	_,N = negatives.shape
	
	# A x G_max
	pos = scores[temp,positives]
	neg = scores[temp,negatives]
	
	# (A,P) -> (A,P,1)
	# (A,N) -> (A,1,N)
	pos = pos.unsqueeze(-1)
	neg = neg.unsqueeze(-2)
	
	# A x P x N
	scores = -pos + neg + margin_scale
	
	loss = torch.relu(scores)
	
	# A x P x N
	mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
	
	# mask out padding values so they don't contribute to loss
	loss = loss.masked_fill(~mask, 0)
	
	# naiive hard negative mining
	mask = mask & (loss > 0)
	
	loss = loss[mask].sum() / mask.sum().clamp(min=1)
	return loss
