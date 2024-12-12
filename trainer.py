import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle

from sklearn.metrics import roc_curve, auc
from collections import defaultdict

from losses import *

class Trainer():
	def __init__(self, model, config, globals):
		## training stuff
		self.model = model
		self.device = config['cuda_device']
		
		self.batch_size = config['batch_size']
		self.iters_per_val = globals['iterations_per_validation']
		# linear learning rate warmup is used
		self.warmup_iters = config['warmup_iterations']
		
		# loss function
		# these are defined in losses.py because this file is too long already
		self.loss = loss_mapping[config['loss']]
		
		# optimizer
		model_params = self.model.parameters()
		if config['optimizer'] == 'sgd':
			model.optimizer = torch.optim.SGD(
								model_params,
								lr = config['start_lr'],
								momentum = 0.9,
								weight_decay = config['weight_decay'],
								nesterov = True)
		elif config['optimizer'] == 'adam':
			model.optimizer = torch.optim.AdamW(
								model_params, 
								lr = config['start_lr'],
								weight_decay = config['weight_decay']
								)
		else:
			raise ValueError(config['optimizer'])
		self.optimizer = model.optimizer
		
		## validation and performance stuff
		self.model.best = False # variable to keep track of whether to update best scores
		self.best_val_AUC = 0.5
		
		self.config = config
		self.globals = globals
	
	# this automates whole training process
	def train(self, train_loader, val_loader, test_loader):
		# loop stuff
		loop = True
		iter_per_epoch = len(train_loader)
		epoch = 0
		checkpoint = 1
		
		# automatic mixed-precision
		scaler = torch.cuda.amp.GradScaler()
		
		# performance tracking
		total_loss = 0
		
		# initialize warmup
		for group in self.optimizer.param_groups:
			group['lr'] /= self.warmup_iters
		
		self.model.to(self.device)
		self.model.train()
		
		print('Training Begins...')
		training_start_time = time.time()
		while loop:
			epoch_start_time = time.time()
			for i, batch_dict in enumerate(train_loader):
				iteration = epoch*iter_per_epoch + i
				## train iteration
				with torch.enable_grad():
					self.optimizer.zero_grad()
					with torch.cuda.amp.autocast():
						# head per part:	heads x total_images x D
						features = self.model(batch_dict['images'].to(self.device))
						# total_images x total_images
						scores = self.model.calc_similarity_scores(features, batch_dict['id2part']).reshape(-1,1)
						scores = self.model.metric_scaler(scores).reshape(features.shape[-2], features.shape[-2])
						loss = self.loss(
								scores,
								batch_dict,
								self.model.get_metric_scale(),
								self.config
						)
						total_loss += loss.item()
						# AMP update step
						scaler.scale(loss).backward()
						scaler.step(self.model.optimizer)
						scaler.update()
				# print to output
				if (iteration > 0 and iteration % 100 == 0):
					now = time.time()
					print('EP.', epoch, end='\t')
					print('{}%'.format(np.round(i/len(train_loader)*100,1)), end='\t')
					print('Loss:', np.round(total_loss/(((iteration-1)%self.iters_per_val)+1),3), end='\t')
					print('i/s:', np.round((iteration+1)/(now-training_start_time),1), end='\t')
					print('t/e:', int((now-training_start_time)/((iteration+1)/len(train_loader))), 's', end='\t')
					print('etv:', int((self.iters_per_val-(iteration%self.iters_per_val))/((iteration+1)/(now-training_start_time))), 's',end='\t')
					print('t:', int(now-training_start_time), 's', end='\r')
				## validate
				if (iteration>0 and iteration%self.iters_per_val==0):
					print('')
					print('VAL {}'.format(checkpoint), end='\t')
					
					print('Iter {}\tEpoch {}-{}'.format(
						iteration,
						epoch,
						i
					))
					
					train_loss = total_loss / self.iters_per_val
					
					print('Validating...')
					val_dict = test(self.model, val_loader)
					
					# check if model should continue training
					self.model, loop = self.model.scheduler.check(self.model, val_dict['all']['AUC'], checkpoint = checkpoint, iter = iteration)
					
					# update best scores if new best score
					if self.model.best:
						print('new best!')
						self.best_val_AUC = val_dict['all']['AUC']
					print('')
					
					
					if not loop:
						# stop training
						print('Training Complete')
						print('Best val mean AUC: {}'.format(self.best_val_AUC,3))
						print('')
						loop = False
						break
					
					# reset trackers
					total_loss = 0
					checkpoint += 1
					
					# set models back to training mode
					self.model.to(self.device)
					self.model.train()
				## end validation
				
				# lr warmup
				# iteration goes up to (and including) self.warmup_iters-2
				# last multiplication cancels out original (/= self.warmup_iters)
				if iteration <= (self.warmup_iters-2):
					factor = (iteration+2)/(iteration+1)
					for group in self.model.optimizer.param_groups:
						group['lr'] *= factor
			
			##### end for loop
			epoch += 1
			print('')
		
		##### end while loop
		
		# load best model
		self.model.cpu()
		checkpoint = torch.load('{}/best.pt'.format(self.config['_savedir']), map_location='cpu')
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.to(self.device)
		# rerun val set
		print('Revalidating...')
		val_dict = test(self.model, val_loader)
		
		
		# run test set
		print('Testing...')
		test_dict = test(self.model, test_loader)
		
		
		with open('{}/test_dict.pkl'.format(self.config['_savedir']),'wb') as fp:
			pickle.dump(test_dict,fp)
		
		return self.model


def test(model, loader):
	device = next(model.parameters()).device
	
	# inference
	all_features = []
	all_id2patient = []
	all_id2part = []
	
	model.eval()
	print('\tbuilding features...')
	with torch.no_grad():
		for batch_dict in loader:
			features = model(batch_dict['images'].to(device))
			if len(features.shape) == 3:
				# head per part
				# H x A x D
				all_features.append(features.cpu().detach())
			else:
				# A x D
				all_features.append(features.cpu().detach())
			all_id2patient.append(batch_dict['id2patient'])
			all_id2part.append(batch_dict['id2part'])
		model.cpu()
		all_features = torch.cat(all_features, dim=-2).to(device, non_blocking=True)
		
		# collect mappings
		# mappings are 0-indexed
		# [0,1,2] [0,1,2,3] etc.
		# target:
		# [0,1,2] [3,4,5,6]
		current_offset = 0
		for i, thing in enumerate(all_id2patient):
			temp = max(thing)+1
			all_id2patient[i] += current_offset
			current_offset += temp
		all_id2patient = torch.cat(all_id2patient).to(device, non_blocking=True)
		all_id2part = torch.cat(all_id2part).to(device, non_blocking=True)
		
		assert len(all_features.shape) == 3
		
		
		score_matrix = model.calc_similarity_scores(all_features, all_id2part)
		
		metrics = head_per_part_evaluation(
			score_matrix,	# A x A
			all_id2patient,	# A
			all_id2part,	# A
		)
	
	model.to(device, non_blocking=True)
	print('')
	
	return metrics

# calculate ROC
# CMC requires patient-wise pos/negatives, so it is not calculated here
def _calc_ROC(labels, scores):
	outputs = {}
	
	# ROC
	fpr, tpr, _ = roc_curve(labels, scores)
	for i in range(10):
		temp = tpr[fpr < 10**(-i)]
		outputs['sen_fpr_1e-{}'.format(i)] = temp[-1] if len(temp) > 0 else 0
	fpr, tpr = dedupe(fpr, tpr)
	tpr, fpr = dedupe(tpr, fpr)
	outputs['ROC_fpr'], outputs['ROC_tpr'] = fpr, tpr
	outputs['AUC'] = auc(fpr, tpr)
	
	return outputs




'''
CMC (in this study) is calculated by aggregating P x P-1 step functions
---
for each of the P x P-1 positive pairs
	rank the score of the positive pair among the scores in the negative gallery
	(so sort N+1 scores, find where the positive sample lies)
	this positive pair's step function is offset by the rank of the positive pair
average all of the step functions for CMC
---

but sorting G for P x P-1 positive pairs is expensive and most of the work is duplicated
O(P*P*(N+1)log(N+1))

instead, for each P, sort G
then iterate smartly

---
for each P
	sort P+N-1 (or G)
	use argwhere to get indices of positives in G
	for each P
		add to CMC sum in correct index
---
O(P*(P+N)log(P+N))
in the end this should be approximately P times faster
'''
# INPUT SHAPE: P x P+N(-1) or P x G
def _calc_CMC(labels, scores):
	# sort P x G scores
	idx = scores.argsort(dim=-1, descending=True).cpu()
	# sort labels according to scores
	labels = labels[torch.arange(len(idx)).reshape(-1,1),idx]
	# G
	CMC_sum = torch.zeros(labels.shape[1]+1, device = scores.device)
	for _labels in labels:
		# _labels (G)
		# argwhere returns indices of nonzero entries (1's, positives)
		# each positive is offset by +1 for each preceding positive
		# enumerating the argwhere output gives the offset to reverse
		for offset, i in enumerate(torch.argwhere(_labels).flatten()):
			CMC_sum[i-offset] += 1
	
	# cumsum on indices is used instead of addition to range sum[:i-offset+1] for /maybe/ less memory access
	CMC = CMC_sum.cumsum(0) / labels.sum().item()
	return CMC.cpu().numpy()

# P x P-1+N or P x total_gallery_images
def _calc_CMC_yeardeltas(labels, scores, yeardeltas):
	idx = scores.argsort(dim=-1, descending=True).cpu()
	_arange = torch.arange(len(idx)).reshape(-1,1)
	sorted_labels = labels[_arange,idx]
	sorted_yeardeltas = yeardeltas[_arange,idx]
	K = len(labels[0])+1
	CMC_sums = defaultdict(lambda : torch.zeros(K))
	counts = defaultdict(int)
	for _labels, _yeardeltas in zip(sorted_labels, sorted_yeardeltas):
		for offset, i in enumerate(torch.argwhere(_labels).flatten()):
			CMC_sums[_yeardeltas[i].item()][i-offset] += 1
			counts[_yeardeltas[i].item()] += 1
	
	sums = {}
	for k,v in CMC_sums.items():
		sums[k] = (v.cumsum(0) / counts[k]).numpy()
	return sums

# remove intermediate points along straight line of ROC, DET, etc.
def dedupe(x, y):
	x = list(x)
	y = list(y)
	assert len(x) == len(y)
	for i in range(len(x)-3, -1, -1):
		if x[i] == x[i+2]:
			del x[i+1]
			del y[i+1]
	return x, y

def remove_diagonal(M):
	size = len(M)
	return M.flatten()[1:].view(size-1,size+1)[:,:-1].reshape(size,size-1)

# evaluate scores of all pairs and calculate metrics
def head_per_part_evaluation(
		score_matrix,	# A x A
		id2patient,		# A
		id2part		# A
	):
	A = len(id2patient)
	outputs = {}
	
	# A x A
	label_matrix = (id2patient.reshape(1,-1) == id2patient.reshape(-1,1)).int()
	del id2patient
	
	## calculate pairwise part metrics
	triu_mask = torch.triu(torch.ones((A,A), device=score_matrix.device),1).bool()
	unidentity_mask = ~torch.eye(A, device=score_matrix.device).bool()
	parts = torch.unique(id2part, sorted=True)
	id2part = id2part.unsqueeze(1) # A x 1
	xpart_mask = (id2part != id2part.T)
	
	for i, part_a in enumerate(parts):
		for part_b in parts:
			if part_a > part_b:
				continue
			# matching r-hand with bi-hand is redundant
			elif part_a == 2 and part_b == 3:
				continue
			print('\tevaluating [{}-{}]...'.format(part_a, part_b), end='\t')
			mask_a = id2part==part_a
			mask_b = id2part==part_b
			part_mask = mask_a * mask_b.T
			
			score_mask = (triu_mask*part_mask)
			scores_ab = score_matrix[score_mask].cpu().numpy()
			labels_ab = label_matrix[score_mask].cpu().numpy()
			
			outputs['{}-{}'.format(part_a, part_b)] = _calc_ROC(labels_ab, scores_ab)
			print(np.round(outputs['{}-{}'.format(part_a, part_b)]['AUC'],4),'AUC')
			
			score_mask = (unidentity_mask*part_mask)
			scores_ab = score_matrix[score_mask]
			labels_ab = label_matrix[score_mask]
			a_count = mask_a.sum()
			b_count = mask_b.sum() - int(part_a == part_b)
			outputs['{}-{}'.format(part_a, part_b)]['CMC'] = _calc_CMC(	
									labels_ab.reshape(a_count, b_count),
									scores_ab.reshape(a_count, b_count))
	
	# aggressively clear gpu memory because calculating CMC takes up a lot of memory
	del mask_a, mask_b, part_mask, scores_ab, labels_ab, score_mask, a_count, b_count, part_a, part_b, unidentity_mask, id2part, parts 
	
	
	## calculate overall and xpart metrics
	# get upper triangle (excluding diagonal)
	all_scores = score_matrix[triu_mask].cpu().numpy()
	all_labels = label_matrix[triu_mask].cpu().numpy()
	
	print('\tevaluating [all]...', end='\t')
	outputs['all'] = _calc_ROC(all_labels, all_scores)
	print(np.round(outputs['all']['AUC'], 4),'AUC')
	
	print('\tevaluating xpart...', end='\t')
	xpart_scores = score_matrix[xpart_mask*triu_mask].cpu().numpy()
	xpart_labels = label_matrix[xpart_mask*triu_mask].cpu().numpy()
	outputs['xpart'] = _calc_ROC(xpart_labels, xpart_scores)
	print(np.round(outputs['xpart']['AUC'],4),'AUC')
	
	del triu_mask
	
	# CMC
	# matrix without diagonal
	# A x A-1
	score_matrix = remove_diagonal(score_matrix)
	label_matrix = remove_diagonal(label_matrix)
	
	outputs['all']['CMC'] = _calc_CMC(label_matrix, score_matrix)
	
	xpart_mask = remove_diagonal(xpart_mask)
	# unlike above, each row will now have a variable number of items
	# because eval_CMC requires fixed size input, we have to do things differently and mask instead of filter+reshape
	score_matrix.masked_fill_(~xpart_mask, -1_000_000_000)
	label_matrix.masked_fill_(~xpart_mask, 0)
	outputs['xpart']['CMC'] = _calc_CMC(label_matrix, score_matrix)
	
	return outputs

# these are defined in losses.py
loss_mapping = {
	'bce':		all_pairs_BCE_loss,
	'ce':		all_pairs_CE_loss,
	'margin':	all_pairs_margin_loss,
	'triplet':	all_triplet_loss
}