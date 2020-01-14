from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/rotten.....)")
op.add_option("-m", "--poolmode", type="string", default='', help="path to Glove embeddings")
op.add_option("-e", "--executed", type="string", default='', help="executed models separated by /.. ex (hardmv/ds/raykar)")

(opts, args) = op.parse_args()
folder = opts.path
M_seted = opts.Ngroups 
pool_mo = opts.poolmode
executed_models = opts.executed  #arg
if len(executed_models) == 0: #put alls
    executed_models = ["mv","oursglobal"]
else:
    executed_models = executed_models.split("/") 

#GLOBAL Variables
BATCH_SIZE = 128
EPOCHS_BASE = 50
OPT = 'adam' #optimizer for neural network 
TOL = 3e-2 #tolerance for relative variation of parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras, time, sys, os, gc
from sklearn.metrics import confusion_matrix

DTYPE_OP = 'float32'
keras.backend.set_floatx(DTYPE_OP)

if DTYPE_OP == 'float64':
    keras.backend.set_epsilon(np.finfo(np.float64).eps)
elif DTYPE_OP == 'float32':
    keras.backend.set_epsilon(np.finfo(np.float32).eps)

def continous_2_cat(data):
    if data == -1:
        return -1
    return np.where(limits <= data )[0][-1]

### LOAD DATA ######3
columns = ["GT","file name","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10"]
df_ann = pd.read_csv(folder+"fgnet_age_estimations.csv",header=None,names=columns)

gt_aux = df_ann["GT"].values
limits = [0, 2.5, 7.5, 12.5, 19.5, 36.5, 50.5, np.inf ] 
Z_data = np.zeros((df_ann.shape[0]), dtype='int')
for i in range(Z_data.shape[0]):
    Z_data[i] = continous_2_cat(gt_aux[i]) #int(gt_aux[i]/10)
K = Z_data.max()+1

file_names = []
for value in df_ann["file name"].values:
    aux = value[7:-4].replace("_", '')
    if "M" in aux:
        split_v = "M"
    elif "F" in aux:
        split_v = "F"
    subject, age = aux.split(split_v)
    file_names.append( subject.zfill(3)+ "A"+ age.zfill(2)+".JPG" )
    
from PIL import Image
folder_img = folder+"FGNET/images/"
X_images = []
for i,values in enumerate(file_names):
    I = Image.open(folder_img+values).convert("RGB") 
    I = I.resize((224,224), Image.ANTIALIAS) #reshape it
    #print(I.size)
    X_images.append(np.asarray(I).astype("uint8"))
    I.close()
gc.collect()
X_images = np.asarray(X_images)
print("Images shapes: ",X_images.shape)

from code.learning_models import through_CNNFace
if len(pool_mo) == 0:
    pool_mo = None
#"senet50" # 'vgg16', "resnet50", "senet50"
new_X = through_CNNFace(X_images.astype('float32'), weights_path="vgg16",pooling_mode=pool_mo)
print("New shape through VGG: ",new_X.shape)

np.random.seed(0)
mask_test = np.random.rand(Z_data.shape[0]) < 0.3
Z_test = Z_data[mask_test]
Z_train = Z_data[~mask_test]
X_train = new_X[~mask_test]
X_test = new_X[mask_test]

print("Input train shape:",X_train.shape)
print("Label train shape:",Z_train.shape)

print("Input test shape:",X_test.shape)
print("Label test shape:",Z_test.shape)

from code.learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from code.learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple, Clonable_Model, default_RNN_text #deep learning

from code.evaluation import Evaluation_metrics
from code.representation import *
from code.utils import *
from code.baseline import LabelInference, RaykarMC
from code.MixtureofGroups import GroupMixtureGlo, project_and_cluster,clusterize_annotators, GroupMixtureInd

### Delta convergence criteria
from code.utils import EarlyStopRelative
ourCallback = EarlyStopRelative(monitor='loss',patience=1,min_delta=TOL)

start_time_exec = time.time()

# data from Amazon Mechanical Turk
print("Loading AMT data...")
df_ann = pd.read_csv(folder+"fgnet_age_estimations.csv",header=None,names=columns)
r_obs = np.zeros((df_ann.shape[0],K))
for i,anns in enumerate(df_ann.iloc[:,2:].values):
    for value in anns:
        r_obs[i, continous_2_cat(value) ] +=1
if r_obs.shape[0] != Z_train.shape[0]:
    r_obs = r_obs[~mask_test]

N, K = r_obs.shape
print("repeats shape: ",r_obs.shape)
print("Classes: ",K)


model_UB = CNN_simple(X_train.shape[1:],r_obs.shape[1],128,1, BN=True,drop=0.25,double=False,
                      global_pool=True,dense_units=0)
model_UB.summary()
clone_UB = Clonable_Model(model_UB)

results_softmv_train = []
results_softmv_test = []
results_hardmv_train = []
results_hardmv_test = []
results_ours_global_train = []
results_ours_global_test = []
results_ours_global_testA = []

############### MV/DS and calculate representations##############################
#Deterministic
confe_matrix_G = get_Global_confusionM(Z_train,r_obs)
if "mv" in executed_models:
    start_time = time.time()
    
    mv_probas = majority_voting(r_obs,repeats=True,probas=True) 
    mv_conf_probas = generate_confusionM(mv_probas, r_obs) #confusion matrix of all annotators

    mv_onehot = keras.utils.to_categorical(mv_probas.argmax(axis=-1))
    mv_conf_onehot = generate_confusionM(mv_onehot, r_obs) #confusion matrix of all annotators

for _ in range(20): #repetitions
    ############# EXECUTE ALGORITHMS #############################
    if "mv" in executed_models:
        model_mvsoft = clone_UB.get_model() 
        model_mvsoft.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist = model_mvsoft.fit(X_train, mv_probas, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over soft-MV, Epochs to converge =",len(hist.epoch))
        Z_train_p_mvsoft = model_mvsoft.predict(X_train)
        Z_train_pred_mvsoft = Z_train_p_mvsoft.argmax(axis=-1) #model_mvsoft.predict_classes(X_train)
        Z_test_pred_mvsoft = model_mvsoft.predict_classes(X_test)
        keras.backend.clear_session()

        model_mvhard = clone_UB.get_model() 
        model_mvhard.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_mvhard.fit(X_train, mv_onehot, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over hard-MV, Epochs to converge =",len(hist.epoch))
        Z_train_p_mvhard = model_mvhard.predict(X_train)
        Z_train_pred_mvhard = Z_train_p_mvhard.argmax(axis=-1) # model_mvhard.predict_classes(X_train)
        Z_test_pred_mvhard = model_mvhard.predict_classes(X_test)
        keras.backend.clear_session()

    if "oursglobal" in executed_models:
        gMixture_Global = GroupMixtureGlo(X_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Global.define_model("simple cnn",128, 1, BatchN=True,drop=0.25,double=False,h_units=0,glo_p=True)
        logL_hists,i = gMixture_Global.multiples_run(20,X_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_OG = gMixture_Global.get_predictions(X_train)
        Z_test_p_OG = gMixture_Global.get_predictions(X_test)
        keras.backend.clear_session()

    ################## MEASURE PERFORMANCE ##################################
    if "mv" in executed_models:
        evaluate = Evaluation_metrics(model_mvsoft,'keras',X_train.shape[0],plot=False)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvsoft,
                                      conf_true_G =confe_matrix_G, conf_pred_G = mv_conf_probas)        
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvsoft)
        results_softmv_train += results1
        results_softmv_test += results2

        evaluate = Evaluation_metrics(model_mvhard,'keras',X_train.shape[0],plot=False)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvhard,
                                      conf_true_G =confe_matrix_G, conf_pred_G = mv_conf_onehot)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvhard)

        results_hardmv_train += results1
        results_hardmv_test += results2

    if "oursglobal" in executed_models:
        evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False) 
        prob_Yz = gMixture_Global.calculate_Yz()
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_p_OG.argmax(axis=-1), 
                                    conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        
        c_M = gMixture_Global.get_confusionM()
        y_o_groups = gMixture_Global.get_predictions_groups(X_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_global_train += results1
        results_ours_global_testA.append(results2[0])
        results_ours_global_test.append(results2[1])

    print("All Performance Measured")
    if "mv" in executed_models:
        del model_mvsoft,model_mvhard
    if "oursglobal" in executed_models:
        del gMixture_Global
    del evaluate
    gc.collect()
    
#plot measures 
if "mv" in executed_models:
    get_mean_dataframes(results_softmv_train).to_csv("AirlineS_softMV_train.csv",index=False)
    get_mean_dataframes(results_softmv_train, mean_std=False).to_csv("AirlineS_softMV_train_std.csv",index=False)
    get_mean_dataframes(results_softmv_test).to_csv("AirlineS_softMV_test.csv",index=False)
    get_mean_dataframes(results_softmv_test, mean_std=False).to_csv("AirlineS_softMV_test_std.csv",index=False)

    get_mean_dataframes(results_hardmv_train).to_csv("AirlineS_hardMV_train.csv",index=False)
    get_mean_dataframes(results_softmv_train, mean_std=False).to_csv("AirlineS_hardMV_train_std.csv",index=False)
    get_mean_dataframes(results_hardmv_test).to_csv("AirlineS_hardMV_test.csv",index=False)
    get_mean_dataframes(results_hardmv_test, mean_std=False).to_csv("AirlineS_hardMV_test_std.csv",index=False)

if "oursglobal" in executed_models:
    get_mean_dataframes(results_ours_global_train).to_csv("AirlineS_OursGlobal_train.csv",index=False)
    get_mean_dataframes(results_ours_global_train, mean_std=False).to_csv("AirlineS_OursGlobal_train_std.csv",index=False)
    get_mean_dataframes(results_ours_global_test).to_csv("AirlineS_OursGlobal_test.csv",index=False)
    get_mean_dataframes(results_ours_global_test, mean_std=False).to_csv("AirlineS_OursGlobal_test_std.csv",index=False)
    get_mean_dataframes(results_ours_global_testA).to_csv("AirlineS_OursGlobal_testAux.csv",index=False)
    get_mean_dataframes(results_ours_global_testA, mean_std=False).to_csv("AirlineS_OursGlobal_testAux_std.csv",index=False)

print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))
