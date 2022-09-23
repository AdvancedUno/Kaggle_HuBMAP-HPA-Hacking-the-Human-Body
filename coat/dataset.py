from kaggle_hubmap_kv3 import *
from common import *
from sklearn.model_selection import KFold
from augmentation import *

image_size = 768 

TRAIN = '/kaggle/input/hubmapdatasom/hubmap-organ-segmentation/train_images/'
MASKS = '/kaggle/input/hubmap-768-mask-01/masks/'
#------------------------------
def make_fold(fold=0):
	df = pd.read_csv('/kaggle/input/hubmapdatasom/hubmap-organ-segmentation/train.csv')
	
	num_fold = 5
	skf = KFold(n_splits=num_fold, shuffle=True,random_state=42)
	
	df.loc[:,'fold']=-1
	for f,(t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
		df.iloc[v_idx,-1]=f
	
	#check
	if 0:
		for f in range(num_fold):
			train_df=df[df.fold!=f].reset_index(drop=True)
			valid_df=df[df.fold==f].reset_index(drop=True)
			
			print('fold %d'%f)
			t = train_df.organ.value_counts().to_dict()
			v = valid_df.organ.value_counts().to_dict()
			for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
				print('%32s %3d (%0.3f)  %3d (%0.3f)'%(k,t.get(k,0),t.get(k,0)/len(train_df),v.get(k,0),v.get(k,0)/len(valid_df)))
			
			print('')
			zz=0
	
	train_df=df[df.fold!=fold].reset_index(drop=True)
	valid_df=df[df.fold==fold].reset_index(drop=True)
	return train_df,valid_df

class CustomDataset(Dataset):
    def __init__(self, real_df, stage, augment):
        self.df = real_df
        self.length = len(self.df)
        self.fnames = [str(fname)+'.tiff' for fname in self.df['id']]
        self.augment = augment
        self.organ_to_label = {
                       'kidney' : 1,
                       'prostate' : 2,
                       'largeintestine' : 3,
                       'spleen' : 4,
                       'lung' : 5}
            
    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        
        fname = self.fnames[index]
        d = self.df.iloc[index]
        organ = self.organ_to_label[d.organ]
        img = fname

        img_number = int(fname.split(".")[0])

        
        image = cv2.imread(os.path.join(TRAIN,fname))
        mask = cv2.imread(os.path.join(MASKS,fname[:-5]+'.png'),cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)/255
        mask  = mask.astype(np.float32)
        
        
        if self.augment is not None:
            image, mask = self.augment(image, mask, organ)
            
#         mask = mask*255
#         mask = np.where(mask>0.5, 1, 0)

        r ={}
        r['index']= index
        r['id'] = img_number
        r['organ'] = torch.tensor([organ], dtype=torch.long)
        r['image'] = image_to_tensor(image)
        r['mask' ] = mask_to_tensor(mask>0.5)
        return r

def image_to_tensor(image, mode='bgr'): #image mode
    if mode=='bgr':
        image = image[:,:,::-1]
    x = image
    x = x.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


def mask_to_tensor(mask):
    x = mask
    x = torch.tensor(x, dtype=torch.float)
    return x

def tensor_to_image(x, mode='bgr'):
	image = x.data.cpu().numpy()
	image = image.transpose(1,2,0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	image = image.astype(np.float32)
	return image

def tensor_to_mask(x):
	mask = x.data.cpu().numpy()
	mask = mask.astype(np.float32)
	return mask

def valid_augment5(image, mask, organ):
    return image, mask

def train_augment5b(image, mask, organ):
    image, mask = do_random_flip(image, mask)
    image, mask = do_random_rot90(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_noise(image, mask, mag=0.1),
        # lambda image, mask: do_random_contast(image, mask, mag=0.40),
        # lambda image, mask: do_random_hsv(image, mask, mag=[0.40, 0.40, 0])
    ], 2): image, mask = fn(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_rotate_scale(image, mask, angle=45, scale=[1, 1]),
    ], 1): image, mask = fn(image, mask)

    return image, mask

