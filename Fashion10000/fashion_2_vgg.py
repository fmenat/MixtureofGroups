import numpy as np
import pandas as pd
from PIL import Image
from optparse import OptionParser
import repackage, gc, math
repackage.up()
from code.learning_models import through_VGG

op = OptionParser()
op.add_option("-p", "--path", type="string", default='./', help="path for data of Label me")
op.add_option("-s", "--set", type="string", default='train', help="set used to pass VGG (train/test)")
op.add_option("-m", "--poolm", type="string", default='', help="pooling mode used on VGG (None or empty/avg/max)")

(opts, args) = op.parse_args()
folder = opts.path
set_name = opts.set 
pool_mo = opts.poolm


images_names = pd.read_csv("./imgpath_"+set_name+".txt", header=None).values[:,0]

X_images = []
for i,values in enumerate(images_names):
	if i%500 == 0:
		print("Va en el ",i)
		gc.collect()

	if values != "error":
	    I = Image.open(folder+"/"+values).convert('RGB') #rodrigues resize images to 150x150 (but VGG was trained with 224x224)
	    I = I.resize((224,224), Image.ANTIALIAS) #reshape it
	    I = np.asarray(I).astype('uint8')
	else:
		I = np.zeros((224,224,3), dtype='uint8')
	X_images.append(I)
X_images = np.asarray(X_images)
print("Images shapes: ",X_images.shape)
print("Images dtype: ",X_images.dtype)

if pool_mo == "":
    pool_mo = None

save_result = []
L_s = 3000
for split in range(int(math.ceil(X_images.shape[0]/L_s))):
	new_X = X_images[split*L_s : (1+split)*L_s]

	#now pass through VGG
	new_X = through_VGG(new_X.astype('float32'),pooling_mode=pool_mo)
	print("New shape through VGG: ",new_X.shape)
	save_result.append(new_X)

save_result = np.concatenate(save_result, axis=0)
print("Final shape through VGG: ",save_result.shape)

if pool_mo == None:
    np.save('Fashion10000_VGG_'+set_name+".npy",save_result) #none pooling
else:
    np.save('Fashion10000_VGG_'+pool_mo+'_'+set_name+".npy",save_result) #avg/max pooling
