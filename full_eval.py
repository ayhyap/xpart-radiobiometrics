print('Importing libraries...')

import os
import json
import argparse

import torch
import numpy as np
import pandas as pd
import pickle

from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression

from dataset import *
from models import ModelConstructor

# python full_eval.py runs/OAI-4part-public.json 0
# python -i full_eval.py runs/debug_1val.json 3
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
					help='path to a hyperparameter configs json')
parser.add_argument('device', type=str,
					help='cuda device to use')

args = parser.parse_args()
CONFIG_DIR = '/'.join(args.config.split('/')[:-1])
CONFIG_FILE = args.config.split('/')[-1]

def dedupe(x, y):
	x = list(x)
	y = list(y)
	assert len(x) == len(y)
	for i in range(len(x)-3, -1, -1):
		if x[i] == x[i+2]:
			del x[i+1]
			del y[i+1]
	return x, y

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

def remove_diagonal(M):
	size = len(M)
	return M.flatten()[1:].view(size-1,size+1)[:,:-1].reshape(size,size-1)


if __name__ == '__main__':
	# load globals
	with open('globals.json','r') as fp:
		globals = json.load(fp)
	
	# load hyperparams
	# open baseline
	with open(globals['baseline_json'],'r') as fp:
		config = json.load(fp)
	
	# update baseline with new hyperparams
	with open('{}/{}'.format(CONFIG_DIR, CONFIG_FILE),'r') as fp:
		config.update(json.load(fp))
	
	if args.device == 'cpu':
		config['cuda_device'] = torch.device('cpu')
	else:
		config['cuda_device'] = torch.device('cuda:{}'.format(args.device))
	
	# run save dir
	config['_savedir'] = SAVEDIR = '.'.join(args.config.split('.')[:-1])
	
	# IMAGEFILES
	IMAGE_DIR = globals['image_dir']
	print('using images in {}'.format(IMAGE_DIR))
	
	print(CONFIG_FILE)
	print('Initializing Trainer...')
	
	###################
	### Dataframe stuff
	# metadata_csv defines all images to be used; excluded images are removed during preprocessing
	df = pd.read_csv(globals['metadata_csv'])
	df['exam_idx'] = -1
	parts_todo = ['AP Pelvis','Bilateral PA Fixed Flexion Knee','PA Bilateral Hand','PA Right Hand']
	for i, part in enumerate(parts_todo):
		df.loc[df.exam == part, 'exam_idx'] = i
	
	config['num_parts'] = len(parts_todo)
	df = df[df.exam.isin(parts_todo)]
	
	# replace M/F with 0/1
	try:
		df.loc[df.sex=='M','sex'] = 0
		df.loc[df.sex=='F','sex'] = 1
		df['sex'] = df.sex.astype(int)
	except AttributeError:
		print('sex column not found in metadata csv and will not be included in analyses')
	
	
	# group into age groups
	try:
		df['age_years'] = df['interview_age']/12
		df['age_group'] = ''
		for i in range(45,90,10):
			df.loc[(df.age_years >= i)&(df.age_years < i+10),'age_group'] = '{}-{}'.format(i,i+9)

		df.loc[df.age_group == '75-84', 'age_group'] = '75-'
		df.loc[df.age_group == '85-94', 'age_group'] = '75-'
	except AttributeError:
		print('interview_age column not found in metadata csv and will not be included in analyses')

	# race
	try:
		df['race_white'] = np.nan
		df.loc[df.race == 0, 'race_white'] = 0
		df.loc[df.race == 1, 'race_white'] = 1
		df.loc[df.race == 2, 'race_white'] = 0
		df.loc[df.race == 3, 'race_white'] = 0
	except AttributeError:
		print('race column not found in metadata csv and will not be included in analyses')
	
	# KL grade
	try:
		df['KL_max_binary'] = np.nan
		df.loc[df.KL_max == 0, 'KL_max_binary'] = 0
		df.loc[df.KL_max == 1, 'KL_max_binary'] = 0
		df.loc[df.KL_max == 2, 'KL_max_binary'] = 1
		df.loc[df.KL_max == 3, 'KL_max_binary'] = 1
		df.loc[df.KL_max == 4, 'KL_max_binary'] = 1
	except AttributeError:
		print('KL_max column not found in metadata csv and will not be included in analyses')
	
	
	###################
	
	# dicts for collecting metrics
	# main task metrics
	aucdict = defaultdict(list)
	rocdict = defaultdict(list)
	cmcdict = defaultdict(list)
	# probe task metrics
	ydict = defaultdict(list)
	xdict = defaultdict(list)
	probedict = defaultdict(list)
	for XVAL_SPLIT in range(4):
		print('SPLIT', XVAL_SPLIT)
		# dataset init
		xval_set = ContrastivePatientDS(
						df,
						IMAGE_DIR, 
						img_size=config['img_size'], 
						augment=False,
						img_limit=9999)
		collate_fn = patient_collate_flatten
		
		xval_loader = DataLoader(xval_set, 
							batch_size = 16, 
							shuffle = False, 
							num_workers = globals['dataloader_workers'], 
							drop_last = False, 
							pin_memory = True, 
							collate_fn = collate_fn)
		
		model = ModelConstructor(config)
		model = model.to(config['cuda_device'])
		
		model.name = CONFIG_FILE.replace('.json','')
		
		# load best model
		config['_savedir'] = '{}/fold{}'.format(SAVEDIR,XVAL_SPLIT)
		try:
			checkpoint = torch.load('{}/best.pt'.format(config['_savedir']), map_location=config['cuda_device'])
		except:
			print('error loading model file, skipping to next fold...')
			continue
		model.load_state_dict(checkpoint['model_state_dict'])
		
		device = config['cuda_device']
		
		prehead_features = []	# for linear probing
		features = []			# after splitting into heads
		files = []
		
		model.eval()
		print('\tbuilding features...')
		with torch.no_grad():
			for i, batch_dict in enumerate(xval_loader):
				print('{}/{}'.format(i, len(xval_loader)), end='\r')
				# pre-head
				# A x D
				x = model.cnn.forward_features(batch_dict['images'].to(device))
				x = model.cnn.global_pool(x)
				x = model.cnn.head_drop(x)
				prehead_features.append(x.cpu().detach())
				# head per part
				# H x A x D
				x = model.cnn.classifier(x).reshape(len(x),model.num_heads,-1).transpose(0,1)
				features.append(x.cpu().detach())
				files.append(batch_dict['files'])
			calc_similarity_scores = model.calc_similarity_scores
			del x, model
			
			# A x D
			prehead_features = torch.cat(prehead_features)
			# H x A x D
			features = torch.cat(features, dim=-2)
			# A
			files = np.concatenate(files)
			
			## roc + cmc
			print('evaluating')
			_df = df.set_index('file').loc[files]
			fold_df = _df[_df.xval_split == XVAL_SPLIT]
			fold_mask = _df.xval_split.values == XVAL_SPLIT
			
			# filter for test set features only
			features = features[:,fold_mask].to(device, non_blocking=True)
			id2part = torch.from_numpy(fold_df.exam_idx.values)
			# A x A
			score_matrix = calc_similarity_scores(features, id2part)
			
			A = len(score_matrix)
			triu_mask = torch.triu(torch.ones((A,A)),1).bool()
			
			id2patient = torch.from_numpy(fold_df.src_subject_id.values)
			label_matrix = (id2patient.unsqueeze(0) == id2patient.unsqueeze(1)).int()
			
			print('overall', end='\t')
			scores = score_matrix[triu_mask].cpu().numpy()
			labels = label_matrix[triu_mask].cpu().numpy()
			fpr,tpr,_ = roc_curve(labels, scores)
			aucdict['overall'].append(auc(fpr,tpr))
			fpr,tpr = np.array(dedupe(fpr,tpr))
			tpr,fpr = np.array(dedupe(tpr,fpr))
			rocdict['overall'].append({'fpr':fpr,'tpr':tpr})
			cmc = _calc_CMC(remove_diagonal(label_matrix), remove_diagonal(score_matrix))
			cmcdict['overall'].append(cmc)
			print(aucdict['overall'][-1])
			pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-overall.csv'.format(config['_savedir']), index=False)
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-overall.csv'.format(config['_savedir']), index=False)
			
			## filter by race
			try:
				race = torch.from_numpy(fold_df.race.values)
				# R=0
				print('race:other', end='\t')
				subgroup_mask = (race == 0).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['race|other'].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['race|other'].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['race|other'].append(cmc)
				print(aucdict['race|other'][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-race_other.csv'.format(config['_savedir']), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-race_other.csv'.format(config['_savedir']), index=False)
				
				# R=1
				print('race:white', end='\t')
				subgroup_mask = (race == 1).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['race|white'].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['race|white'].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['race|white'].append(cmc)
				print(aucdict['race|white'][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-race_white.csv'.format(config['_savedir']), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-race_white.csv'.format(config['_savedir']), index=False)
				
				# R=2
				print('race:black', end='\t')
				subgroup_mask = (race == 2).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['race|black'].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['race|black'].append({'fpr':fpr,'tpr':tpr})
				
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['race|black'].append(cmc)
				print(aucdict['race|black'][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-race_black.csv'.format(config['_savedir']), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-race_black.csv'.format(config['_savedir']), index=False)
				
				# R=3
				print('race:asian', end='\t')
				subgroup_mask = (race == 3).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['race|asian'].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['race|asian'].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['race|asian'].append(cmc)
				print(aucdict['race|asian'][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-race_asian.csv'.format(config['_savedir']), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-race_asian.csv'.format(config['_savedir']), index=False)
			except AttributeError:
				print('race column not found in metadata csv, skipped')
			
			## filter by sex
			try:
				sex = torch.from_numpy(fold_df.sex.values)
				print('M', end='\t')
				# M=0
				subgroup_mask = (sex == 0).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['sex|male'].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['sex|male'].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['sex|male'].append(cmc)
				print(aucdict['sex|male'][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-sex_male.csv'.format(config['_savedir']), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-sex_male.csv'.format(config['_savedir']), index=False)
				
				print('F', end='\t')
				# F=1
				subgroup_mask = (sex == 1).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['sex|female'].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['sex|female'].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['sex|female'].append(cmc)
				print(aucdict['sex|female'][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-sex_female.csv'.format(config['_savedir']), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-sex_female.csv'.format(config['_savedir']), index=False)
			except AttributeError:
				print('race column not found in metadata csv, skipped')
			
			## filter by age group
			age_group = fold_df.age_group.values
			for target_group in ['55-64', '65-74', '75-', '45-54']:
				print(target_group, end='\t')
				subgroup_mask = torch.from_numpy(age_group == target_group).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['age_group|{}'.format(target_group)].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['age_group|{}'.format(target_group)].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['age_group|{}'.format(target_group)].append(cmc)
				print(aucdict['age_group|{}'.format(target_group)][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-age_group_{}.csv'.format(config['_savedir'],target_group), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-age_group_{}.csv'.format(config['_savedir'],target_group), index=False)
			
			## filter by part
			exam = fold_df.exam.values
			for part in parts_todo:
				print(part, end='\t')
				subgroup_mask = torch.from_numpy(exam == part).unsqueeze(0)
				size = subgroup_mask[0].sum()
				subgroup_mask = subgroup_mask * subgroup_mask.T
				mask = triu_mask * subgroup_mask
				
				scores = score_matrix[mask].cpu().numpy()
				labels = label_matrix[mask].cpu().numpy()
				fpr,tpr,_ = roc_curve(labels, scores)
				aucdict['part|{}'.format(part)].append(auc(fpr,tpr))
				fpr,tpr = np.array(dedupe(fpr,tpr))
				tpr,fpr = np.array(dedupe(tpr,fpr))
				rocdict['part|{}'.format(part)].append({'fpr':fpr,'tpr':tpr})
				
				scores = remove_diagonal(score_matrix[subgroup_mask].reshape(size,size))
				labels = remove_diagonal(label_matrix[subgroup_mask].reshape(size,size))
				cmc = _calc_CMC(labels, scores)
				cmcdict['part|{}'.format(part)].append(cmc)
				print(aucdict['part|{}'.format(part)][-1])
				pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-part_{}.csv'.format(config['_savedir'],part), index=False)
				pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-part_{}.csv'.format(config['_savedir'],part), index=False)
			
			# cross-part
			print('xpart', end='\t')
			mask = torch.from_numpy(exam.reshape(1,-1) != exam.reshape(-1,1))
			scores = score_matrix[mask].cpu().numpy()
			labels = label_matrix[mask].cpu().numpy()
			fpr,tpr,_ = roc_curve(labels, scores)
			aucdict['xpart'].append(auc(fpr,tpr))
			fpr,tpr = np.array(dedupe(fpr,tpr))
			tpr,fpr = np.array(dedupe(tpr,fpr))
			rocdict['xpart'].append({'fpr':fpr,'tpr':tpr})
			
			# unlike above, each row will now have a variable number of items
			# because eval_CMC requires fixed size input, we have to do things differently and mask instead of filter+reshape
			A = len(mask)
			score_matrix.masked_fill_(~mask.to(device), -1_000_000_000)
			label_matrix.masked_fill_(~mask, 0)
			score_matrix = score_matrix.flatten()[1:].view(A-1,A+1)[:,:-1].reshape(A,A-1)
			label_matrix = label_matrix.flatten()[1:].view(A-1,A+1)[:,:-1].reshape(A,A-1)
			cmc = _calc_CMC(label_matrix, score_matrix)
			cmcdict['xpart'].append(cmc)
			print(aucdict['xpart'][-1])
			pd.DataFrame({'rank':np.arange(len(cmc))+1, 'tpr':cmc}).to_csv('{}/cmc-xpart.csv'.format(config['_savedir']), index=False)
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/roc-xpart.csv'.format(config['_savedir']), index=False)
			
			
			
			
			
			
			## linear probing			
			print('linear probing')
			test_mask = (_df.xval_split.values == XVAL_SPLIT)
			train_mask = ~test_mask
			print('age', end='\t')
			# age (regression; r-squared)
			# 0.6129619342418585 0.009172378702358246
			notnan_mask = ~np.isnan(_df.age_years.values)
			train_features = prehead_features[train_mask & notnan_mask]
			test_features = prehead_features[test_mask & notnan_mask]
			train_labels = _df.age_years.values[train_mask & notnan_mask]
			test_labels = _df.age_years.values[test_mask & notnan_mask]
			
			# ridge model only does 0.0003 better, which is within 0.3 stdev
			model = LinearRegression(n_jobs = 4)
			model.fit(train_features, train_labels)
			preds = model.predict(test_features)
			ydict['age'].append(test_labels)
			xdict['age'].append(preds)
			probedict['age'].append(r2_score(test_labels, preds))
			print(probedict['age'][-1])
			pd.DataFrame({'predicted_age':preds, 'actual_age':test_labels}).to_csv('{}/probe-age.csv'.format(config['_savedir']), index=False)
			
			## sex
			print('sex', end='\t')
			# sex (logreg; AUC)
			# 0.9919182050190996 0.002791617326711773
			train_features = prehead_features[train_mask]
			test_features = prehead_features[test_mask]
			groupdict = {}
			train_labels = _df.sex.values[train_mask]
			test_labels = _df.sex.values[test_mask]
			
			model = LogisticRegression(penalty=None, max_iter=10_000, n_jobs = 4)
			model.fit(train_features, train_labels)
			preds = model.predict_proba(test_features)[:,1]
			fpr,tpr,_ = roc_curve(test_labels,preds)
			ydict['sex'].append(tpr)
			xdict['sex'].append(fpr)
			probedict['sex'].append(auc(fpr,tpr))
			print(probedict['sex'][-1])
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/probe-sex.csv'.format(config['_savedir']), index=False)
			
			## KL grade
			print('KL', end='\t')
			# KL_min_binary
			# 0.8444282344856796 0.01277066985909927
			groupdict = {}
			knee_mask = (_df.exam.values == 'Bilateral PA Fixed Flexion Knee')
			notnan_mask = ~np.isnan(_df.KL_max_binary.values)
			train_features = prehead_features[train_mask & notnan_mask & knee_mask]
			test_features = prehead_features[test_mask & notnan_mask & knee_mask]
			train_labels = _df.KL_max_binary.values[train_mask & notnan_mask & knee_mask]
			test_labels = _df.KL_max_binary.values[test_mask & notnan_mask & knee_mask]
			
			model = LogisticRegression(penalty=None, max_iter=10_000, n_jobs = 4)
			model.fit(train_features, train_labels)
			preds = model.predict_proba(test_features)[:,1]
			fpr,tpr,_ = roc_curve(test_labels,preds)
			ydict['KL_max_binary'].append(tpr)
			xdict['KL_max_binary'].append(fpr)
			probedict['KL_max_binary'].append(auc(fpr,tpr))
			print(probedict['KL_max_binary'][-1])
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/probe-KL_max.csv'.format(config['_savedir']), index=False)
			
			# race
			'''
			0   other
			1   white/caucasian
			2   black/african american
			3   asian
			'''
			print('race', end='\t')
			
			# race_white
			# 0.9495079505458935 0.007955175705465794
			notnan_mask = ~np.isnan(_df.race_white.values)
			train_features = prehead_features[train_mask & notnan_mask]
			test_features = prehead_features[test_mask & notnan_mask]
			groupdict = {}
			train_labels = _df.race_white.values[train_mask & notnan_mask]
			test_labels = _df.race_white.values[test_mask & notnan_mask]
			
			model = LogisticRegression(penalty=None, max_iter=10_000, n_jobs = 4)
			model.fit(train_features, train_labels)
			preds = model.predict_proba(test_features)[:,1]
			fpr,tpr,_ = roc_curve(test_labels,preds)
			ydict['race_white'].append(tpr)
			xdict['race_white'].append(fpr)
			probedict['race_white'].append(auc(fpr,tpr))
			print(probedict['race_white'][-1])
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/probe-race_white.csv'.format(config['_savedir']), index=False)
			
			## the labels for prostheses detection were derived from manual inspection and not OAI data
			# knee_pros
			# 0.9524113710580884 0.03199176342740359
			print('knee prosthesis', end='\t')
			knee_mask = (_df.exam.values == 'Bilateral PA Fixed Flexion Knee')
			train_features = prehead_features[train_mask & knee_mask]
			test_features = prehead_features[test_mask & knee_mask]
			train_labels = (_df.knee_prosthesis.values[train_mask & knee_mask] == 1).astype(int)
			test_labels = (_df.knee_prosthesis.values[test_mask & knee_mask] == 1).astype(int)
			
			model = LogisticRegression(penalty=None, max_iter=10_000, n_jobs = 4)
			model.fit(train_features, train_labels)
			preds = model.predict_proba(test_features)[:,1]
			fpr,tpr,_ = roc_curve(test_labels,preds)
			ydict['knee_pros'].append(tpr)
			xdict['knee_pros'].append(fpr)
			probedict['knee_pros'].append(auc(fpr,tpr))
			print(probedict['knee_pros'][-1])
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/probe-knee_pros.csv'.format(config['_savedir']), index=False)

			# pelvis_pros
			# 0.9777188883303648 0.010412920148202293
			print('pelvis prosthesis', end='\t')
			knee_mask = (_df.exam.values == 'AP Pelvis')
			train_features = prehead_features[train_mask & knee_mask]
			test_features = prehead_features[test_mask & knee_mask]
			train_labels = (_df.pelvis_prosthesis_nospine.values[train_mask & knee_mask] == 1).astype(int)
			test_labels = (_df.pelvis_prosthesis_nospine.values[test_mask & knee_mask] == 1).astype(int)
			
			model = LogisticRegression(penalty=None, max_iter=10_000, n_jobs = 4)
			model.fit(train_features, train_labels)
			preds = model.predict_proba(test_features)[:,1]
			fpr,tpr,_ = roc_curve(test_labels,preds)
			ydict['pelv_pros'].append(tpr)
			xdict['pelv_pros'].append(fpr)
			probedict['pelv_pros'].append(auc(fpr,tpr))
			print(probedict['pelv_pros'][-1])
			pd.DataFrame({'fpr':fpr, 'tpr':tpr}).to_csv('{}/probe-pelv_pros.csv'.format(config['_savedir']), index=False)
	
	if len(aucdict['overall']) == 4:
		print('')
		print('AVERAGE RESULTS (AUROC)    STD')
		for k,v in aucdict.items():
			print(k, np.mean(v), np.std(v, ddof=1))

		TPRs_at_fpr_1e_3 = defaultdict(list)
		TPRs_at_fpr_1e_6 = defaultdict(list)
		TPRs_at_rank_1 = defaultdict(list)
		TPRs_at_rank_10 = defaultdict(list)
		
		for k,v in rocdict.items():
			for folddict in v:
				temp = folddict['tpr'][folddict['fpr'] < 1e-3]
				TPRs_at_fpr_1e_3[k].append(temp[-1] if len(temp) > 0 else 0)
				temp = folddict['tpr'][folddict['fpr'] < 1e-6]
				TPRs_at_fpr_1e_6[k].append(temp[-1] if len(temp) > 0 else 0)
		
		for k,v in cmcdict.items():
			for cmc in v:
				TPRs_at_rank_1[k].append(cmc[0])
				TPRs_at_rank_10[k].append(cmc[9])
		
		print('AVERAGE RESULTS (TPR@FPR1e-3)    STD')
		for k,v in TPRs_at_fpr_1e_3.items():
			print(k, np.mean(v), np.std(v, ddof=1))
		
		print('AVERAGE RESULTS (TPR@FPR1e-6)    STD')
		for k,v in TPRs_at_fpr_1e_6.items():
			print(k, np.mean(v), np.std(v, ddof=1))
		
		print('AVERAGE RESULTS (TPR@Rank1)    STD')
		for k,v in TPRs_at_rank_1.items():
			print(k, np.mean(v), np.std(v, ddof=1))
		
		print('AVERAGE RESULTS (TPR@Rank10)    STD')
		for k,v in TPRs_at_rank_10.items():
			print(k, np.mean(v), np.std(v, ddof=1))
			

		print('AVERAGE PROBE RESULTS (R2 or AUROC)    STD')
		for k,v in probedict.items():
			print(k, np.mean(v), np.std(v, ddof=1))
		
		summarydict = {'auc':aucdict, 'tpr_at_fpr_1e-3':TPRs_at_fpr_1e_3, 'tpr_at_fpr_1e-6':TPRs_at_fpr_1e_6, 'tpr_at_rank_1':TPRs_at_rank_1,'tpr_at_rank_10':TPRs_at_rank_10}
		# numpy floats aren't json serializable so convert all lists of numpy floats to python floats
		for metric,subdict in summarydict.items():
			for k,v in subdict.items():
				summarydict[metric][k] = np.array(v).tolist()
		
		with open('{}/results_summary.json'.format(SAVEDIR),'w') as fp:
			json.dump(summarydict,fp)
		
		with open('{}/probe_results_summary.json'.format(SAVEDIR),'w') as fp:
			json.dump(probedict,fp)