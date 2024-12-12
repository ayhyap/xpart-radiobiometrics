print('Importing libraries...')

import os
import json
import argparse

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from dataset import *
from trainer import Trainer
from models import ModelConstructor
from schedulers import StepLR as SCHEDULER

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
					help='path to a hyperparameter configs json')

parser.add_argument('device', type=str, 
					help='cuda device to use')
parser.add_argument('xval', type=str,
					help='xval split todo (0-3, 0123 means do all)')

'''
> python main.py runs/OAI-4part-public.json 3 0123
'''

args = parser.parse_args()
CONFIG_DIR = '/'.join(args.config.split('/')[:-1])
CONFIG_FILE = args.config.split('/')[-1]

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
	os.makedirs(config['_savedir'], exist_ok=True)
	
	# IMAGEFILES
	IMAGE_DIR = globals['image_dir']
	print('using images in {}'.format(IMAGE_DIR))
	
	def set_random_seed(seed):
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	print(CONFIG_FILE)
	# set seed for reproducibility
	set_random_seed(config['seed'])
	print('Initializing Trainer...')
	
	# dataframe splits
	
	# metadata_csv defines all images to be used; excluded images are removed during preprocessing
	train_df = pd.read_csv(globals['metadata_csv'])
	train_df['exam_idx'] = -1
	parts_todo = ['AP Pelvis','Bilateral PA Fixed Flexion Knee','PA Bilateral Hand','PA Right Hand']
	for i, part in enumerate(parts_todo):
		train_df.loc[train_df.exam == part, 'exam_idx'] = i
	config['num_parts'] = len(parts_todo)
	
	if type(args.xval) == str:
		SPLITS = [int(split) for split in args.xval]
	else:
		SPLITS = [args.xval]
	
	for XVAL_SPLIT in SPLITS:
		print('================================')
		meta_train =	train_df[(train_df.xval_split != XVAL_SPLIT) & (train_df.exam.isin(parts_todo))]
		meta_val =		train_df[(train_df.xval_split == XVAL_SPLIT) & (train_df.exam.isin(parts_todo))]
		meta_test =		train_df[(train_df.xval_split == XVAL_SPLIT) & (train_df.exam.isin(parts_todo))]
		assert len(meta_test) > 0
		print('----------------------------')
		print('Split {}'.format(XVAL_SPLIT))
		# if validation set too big, remove patients until below limit
		patients = meta_val.src_subject_id.unique().tolist()
		while len(meta_val) > 2100:
			victim = patients.pop() # gets and removes last value in list
			meta_val = meta_val[meta_val.src_subject_id != victim]
		
		# dataset init
		train_set = ContrastivePatientDS(
						meta_train,
						IMAGE_DIR, 
						img_size=config['img_size'], 
						augment=True,
						img_limit=config['patient_image_limit'])
		val_set =	ContrastivePatientDS(
						meta_val,
						IMAGE_DIR, 
						img_size=config['img_size'], 
						augment=False,
						img_limit=9999)
		test_set =	ContrastivePatientDS(
						meta_test,
						IMAGE_DIR, 
						img_size=config['img_size'], 
						augment=False,
						img_limit=9999)
		
		train_loader = 	DataLoader(
							train_set,
							batch_size = config['batch_size'],
							shuffle = True,
							num_workers = globals['dataloader_workers'], 
							drop_last = True, 
							pin_memory = True, 
							collate_fn = patient_collate_flatten)
		val_loader = 	DataLoader(val_set,
							batch_size = 16,
							shuffle = False,
							num_workers = globals['dataloader_workers'],
							drop_last = False,
							pin_memory = True,
							collate_fn = patient_collate_flatten)
		test_loader = 	DataLoader(test_set,
							batch_size = 16,
							shuffle = False,
							num_workers = globals['dataloader_workers'],
							drop_last = False,
							pin_memory = True,
							collate_fn = patient_collate_flatten)
		config['validations_per_epoch'] = int(len(train_loader) / globals['iterations_per_validation']) + 1
		
		model = ModelConstructor(config)
		
		model.name = CONFIG_FILE.replace('.json','__XVAL{}'.format(XVAL_SPLIT))
		config['_savedir'] = '{}/fold{}'.format(SAVEDIR,XVAL_SPLIT)
		os.makedirs(config['_savedir'], exist_ok=True)
		
		model.config = config
		model.scheduler = SCHEDULER(config, globals)
		
		trainer = Trainer(model, config, globals)
		
		## whole training process
		model = trainer.train(train_loader, val_loader, test_loader)
		
		# clean up, just in case some stuff is left in gpu memory
		del trainer
		del model
		torch.cuda.empty_cache()