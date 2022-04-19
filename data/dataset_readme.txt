##########################################
########	VIRDO DATASET	   ########
########   CREATER : YOUNGSUN WI  ########
########  CONTACT: yswi@umich.edu ########
###########################################

1. DESCRIPTION: This dataset is written in 'dtype=torch.float64'. This dataset consists of total 144 deformation scenes from 6 different objects generated through MATLAB. It is divided into 'train' and 'test' dataset, where data['train'][OBJECT IDX = i][DEFORM IDX = j ] and data['test'][OBJECT IDX = i][DEFORM IDX = j ] indicates the same scene, but they are two different subsets of query points.


2. STRUCTURE: The dataset structure is as follows:
VIRDO_simul_dataset = {
'train':{
	<OBJECT IDX>: {
		'nominal': {
			'coords': tensor([1, M, 3]),
			'normals': tensor([1, M, 3]),
			'gt': tensor([1, M, 3]),
			'scale': float
			},
		<DEFORM IDX>: {
			'coords': tensor([1, M, 3]),
			'contact': tensor([1, M_c, 3]),
			'normals': tensor([1, M, 3]),
			'gt': tensor([1, M, 3]),
			'scale': float,
			'reaction': tensor([1,3]
			},
		},
		
	},
'test':{
	<OBJECT IDX>: {
		'nominal': {
			'coords': tensor([1, M, 3]),
			'normals': tensor([1, M, 3]),
			'gt': tensor([1, M, 3]),
			'scale': float
			},
		<DEFORM IDX>: {
			'coords': tensor([1, M, 3]),
			'contact': tensor([1, M_c, 3]),
			'normals': tensor([1, M, 3]),
			'gt': tensor([1, M, 3]),
			'scale': float,
			'reaction': tensor([1,3]
			},
		},
		
	},
}

	* <OBJECT IDX> = Interger from 0 ~ 5. Each number indicates different object.
	* <DEFORM IDX> = Unique integer for each deformation.
	* M = total points (on-surface + off-surface)
	* M_c = a subset of on-surface points that are in contact
	* [:,i,:] elements of 'coords', 'normals', and 'gt' refers ith query point of a scene. To get on-surface points of data_def = data['train'][<OBJECT IDX>][<DEFORM IDX>], you should do data_def['coords'][:,torch.where(data_def['gt'] == 0)[1],:]. 

