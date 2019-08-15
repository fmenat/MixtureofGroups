import numpy as np
import pandas as pd
from PIL import Image
from optparse import OptionParser
import repackage
repackage.up()
from code.learning_models import through_VGG

op = OptionParser()
op.add_option("-p", "--path", type="string", default='data/', help="path for data of Label me")
op.add_option("-s", "--set", type="string", default='train', help="set used to pass VGG (train/test)")
op.add_option("-m", "--poolm", type="string", default='', help="pooling mode used on VGG (None or empty/avg/max)")

(opts, args) = op.parse_args()
folder = opts.path #"../Birds/"
set_name = opts.set 
pool_mo = opts.poolm

#images_names = pd.read_csv(folder+"/lists/files.txt", header=None).values[:,0] ##test and train..
images_names = pd.read_csv(folder+"/lists/"+set_name+".txt", header=None).values[:,0]

X_images = []
for values in images_names:
    I = Image.open(folder+"/images/"+values) #rodrigues resize images to 150x150 (but VGG was trained with 224x224)
    I = I.resize((224,224), Image.ANTIALIAS) #reshape it
    X_images.append(np.asarray(I))
X_images = np.asarray(X_images)
print("Images shapes: ",X_images.shape)

if pool_mo == "":
    pool_mo = None
#now pass through VGG
new_X = through_VGG(X_images.astype('float32'),pooling_mode=pool_mo)
print("New shape through VGG: ",new_X.shape)
if pool_mo == None:
    np.save('Birds_VGG_'+set_name+".npy",new_X) #none pooling
else:
    np.save('Birds_VGG_'+pool_mo+'_'+set_name+".npy",new_X) #avg/max pooling