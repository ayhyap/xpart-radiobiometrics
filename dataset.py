import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# creates batches of PATIENTS of images
# all patients' images are unrolled to yield a sequence of images
def patient_collate_flatten(inputs):
	outputs = {}
	
	# this list comprehension flattens the variable-length multidimensional lists
	# total(images) x c x h x w
	outputs['images'] = torch.stack([img for patient in inputs for img in patient['images']])
	
	# make mappings
	patient_image_counts = np.array([len(patient['images']) for patient in inputs])
	id2patient = np.zeros(len(outputs['images']), dtype=int)
	'''
	counts
	1 3 2
	cumsum [:-1]
	1 4
	indexed on zeros
	0 1 0 0 1 0
	cumsum
	0 1 1 1 2 2
	'''
	_cumsum = patient_image_counts.cumsum()
	id2patient[_cumsum[:-1]] = 1
	id2patient = id2patient.cumsum()
	
	# 0 to N-1
	outputs['id2patient'] = torch.from_numpy(id2patient)
	
	# files
	outputs['files'] = np.concatenate([patient['files'] for patient in inputs])
	outputs['id2part'] = torch.tensor([part for patient in inputs for part in patient['body_part']])
	outputs['dataset_idx'] = torch.tensor([idx for patient in inputs for idx in patient['dataset_idx']])
	return outputs




# helper class for padding to square using torchvision transforms
class SquarePad():
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return transforms.functional.pad(image, padding, 0, 'constant')

'''
PARAMETERS
meta_df		metadata dataframe (has 1 entry per image)
			NOTE: Preprocess the df (filter out exclusions)
				-filter out exclusions, only include images to be used

img_dir		directory string that contains ALL images (images must not be in subdirectories)

img_size	2-tuple of (frontal_size, lateral_size) to resize images to

augment		boolean of whether to augment images

img_limit	integer number of images to limit per patient to avoid overloading GPU memory
'''
class ContrastivePatientDS(Dataset):
	def __init__(	self, 
					meta_df,
					img_dir, 
					img_size, 
					augment,
					img_limit=10
				):
		self.meta_df = meta_df
		self.img_dir = img_dir
		self.image_limit = img_limit
		
		self.patients = self.meta_df['src_subject_id'].unique()
		
		preprocess = []
		preprocess.append(transforms.RandomEqualize(p=1.0))
		preprocess.append(SquarePad())
		preprocess.append(transforms.Resize((img_size,img_size)))
		if augment == 1:
			preprocess.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
			preprocess.append(transforms.RandomAffine(	10, 
														translate=(0.2,0.2), 
														scale=(0.9,1.1), 
														shear=5, 
														interpolation=transforms.InterpolationMode.BILINEAR, 
														fill=0))
		elif augment == 0:
			pass
		else:
			raise ValueError(augment)
	
		preprocess += [transforms.ToTensor(),
					   transforms.Normalize( (0.502,) , (0.289,) )]
		
		self.preprocess = transforms.Compose(preprocess)
	
	def __len__(self):
		return len(self.patients)
	
	def __getitem__(self,i):
		'''
		EXPECTED OUTPUTS
		images			var(images) x c x h x w
		files			total_images
		augment_pairs	1
		'''
		patient = self.patients[i]
		rows = self.meta_df[self.meta_df.src_subject_id == patient]
		
		if len(rows) > self.image_limit:
			rows = rows.sample(self.image_limit)
		
		images = []
		
		for _, row in rows.iterrows():
			file = '{}/{}'.format(self.img_dir,row.file)
			img = Image.open(file).convert('RGB')
			img = img.crop(img.getbbox())
			images.append(self.preprocess(img))
		
		outputs = {}
		
		outputs['images'] = images
		outputs['files'] = rows.file.values
		outputs['dataset_idx'] = [i for _ in images]
		outputs['body_part'] = rows.exam_idx.values
		return outputs

