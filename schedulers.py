import torch

class StepLR():
	def __init__(self, config, globals):
		self.decay_factor = config['decay_factor']
		self.current_factor = 1
		
		self.step_lengths = config['step_lengths']
		self.decays = len(self.step_lengths)-1
		
		self.current_decay_checkpoints = 0
		
		self.best_score = 0.0
		self.config = config
		
	def get_patience(self):
		return self.decays
	
	def decay(self, model):
		checkpoint = torch.load('{}/best.pt'.format(self.config['_savedir']))
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.current_factor /= self.decay_factor
		for group in model.optimizer.param_groups:
			group['lr'] /= self.decay_factor
		
		self.current_decay_checkpoints = 0
		return model
	
	# returns True if training should continue, else False
	def check(self, model, score, **kwargs):
		self.current_decay_checkpoints += 1
		
		if score > self.best_score:
			self.best_score = score
			# save best
			savedict = {
				'name' : model.name,
				'checkpoint': kwargs['checkpoint'],
				'iteration': kwargs['iter'],
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict' : model.optimizer.state_dict(),
				'scheduler': 'StepLR',
				'decay_factor': self.decay_factor,
				'current_factor': self.current_factor,
				'score': score,
			}
			torch.save(savedict, '{}/best.pt'.format(self.config['_savedir']))
			model.best = True
		else:
			model.best = False
		
		if self.current_decay_checkpoints >= self.step_lengths[len(self.step_lengths)-self.decays-1]:
			self.decays -= 1
			if self.decays >= 0:
				model = self.decay(model)
			
		return model, (self.decays >= 0)

