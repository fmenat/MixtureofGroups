from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/LabelMe_Z_train.....)")
op.add_option("-v", "--version", type="int", default=1, help="version of annotations (1, 2 or 3)")

(opts, args) = op.parse_args()
folder = opts.path
M_seted = opts.Ngroups 
version = opts.version 

#GLOBAL Variables
BATCH_SIZE = 64 #128
EPOCHS_BASE = 50
OPT = 'adam' #optimizer for neural network 
TOL = 3e-2 #tolerance for relative variation of parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras, time, sys, os, gc
from sklearn.metrics import confusion_matrix
from keras.models import clone_model

DTYPE_OP = 'float32'
keras.backend.set_floatx(DTYPE_OP)

if DTYPE_OP == 'float64':
    keras.backend.set_epsilon(np.finfo(np.float64).eps)
elif DTYPE_OP == 'float32':
    keras.backend.set_epsilon(np.finfo(np.float32).eps)
    
### Load Data
Xstd_train = np.load(folder+"/LabelMe_VGG_train.npy")
Z_train = np.loadtxt(folder+"/LabelMe_Z_train.txt",dtype='int')

Xstd_val = np.load(folder+"/LabelMe_VGG_valid.npy")
Z_val = np.loadtxt(folder+"/LabelMe_Z_valid.txt",dtype='int')

Xstd_test = np.load(folder+"/LabelMe_VGG_test.npy")
Z_test = np.loadtxt(folder+"/LabelMe_Z_test.txt",dtype='int')

print("Input shape:",Xstd_train.shape)
print("Label shape:",Z_train.shape)


from code.learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from code.learning_models import default_CNN,default_RNN,default_RNNw_emb,CNN_simple, RNN_simple #deep learning

from code.evaluation import Evaluation_metrics
from code.representation import *
from code.utils import *
from code.baseline import LabelInference, RaykarMC
from code.MixtureofGroups import GroupMixtureOpt, project_and_cluster,clusterize_annotators

### Delta convergence criteria
from code.utils import EarlyStopRelative
ourCallback = EarlyStopRelative(monitor='loss',patience=1,min_delta=TOL)

start_time_exec = time.time()

#upper bound model
Z_train_onehot = keras.utils.to_categorical(Z_train)

model_UB = MLP_Keras(Xstd_train.shape[1:],Z_train_onehot.shape[1],128,1,BN=False,drop=0.5)
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
hist = model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
print("Trained Ideal Model , Epochs to converge =",len(hist.epoch))

evaluate = Evaluation_metrics(model_UB,'keras',Xstd_train.shape[0],plot=False)
Z_train_pred = model_UB.predict_classes(Xstd_train)
results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred)
Z_test_pred = model_UB.predict_classes(Xstd_test)
results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)

results1[0].to_csv("LabelMe_UpperBound_train.csv",index=False)
results2[0].to_csv("LabelMe_UpperBound_test.csv",index=False)
del evaluate,Z_train_pred,Z_test_pred,results1,results2
gc.collect()
keras.backend.clear_session()

def get_mean_dataframes(df_values):
    if df_values[0].iloc[:,0].dtype == object:
        RT = pd.DataFrame(data=None,columns = df_values[0].columns[1:], index= df_values[0].index)
    else:
        RT = pd.DataFrame(data=None,columns = df_values[0].columns, index= df_values[0].index)
        
    data = []
    for df_value in df_values:
        if df_value.iloc[:,0].dtype == object:
            data.append( df_value.iloc[:,1:].values )
        else:
            data.append(df_value.values)
    RT[:] = np.mean(data,axis=0)
    
    if df_values[0].iloc[:,0].dtype == object:
        RT.insert(0, "", df_values[0].iloc[:,0].values )
    return RT


# data from Amazon Mechanical Turk
print("Loading AMT data...")
if version == 1:
    file_annot = "/answers.txt"
else:
    file_annot = "/answers_v"+str(version)+".txt"
y_obs = np.loadtxt(folder+file_annot,dtype='int16') #not annotation symbol ==-1
T_weights = np.sum(y_obs != -1,axis=0) #distribucion de anotaciones

print("Remove %d annotators that do not annotate on this set "%(np.sum(T_weights==0)))
y_obs = y_obs[:,T_weights!=0]
T_weights = np.sum(y_obs != -1,axis=0) #distribucion de anotaciones

N,T = y_obs.shape
K = np.max(y_obs)+1 # asumiendo que estan ordenadas
print("Shape (data,annotators): ",(N,T))
print("Classes: ",K)

#generate conf matrix...
confe_matrix = np.zeros((T,K,K),dtype=DTYPE_OP)
for t in range(T):    
    for i in range(N):
        if y_obs[i,t] != -1:
            confe_matrix[t,Z_train[i],y_obs[i,t]] +=1
    mask_nan = confe_matrix[t,:,:].sum(axis=-1) == 0
    mean_replace = np.mean(confe_matrix[t,:,:][~mask_nan],axis=0)
    for value in np.arange(K)[mask_nan]:
        confe_matrix[t,value,:] =  1 #Rodrigues 1./K -- similar  to laplace smooth (prior 1)
    confe_matrix[t,:,:] = confe_matrix[t,:,:]/confe_matrix[t,:,:].sum(axis=-1,keepdims=True) #normalize


results_softmv_train = []
results_softmv_test = []
results_hardmv_train = []
results_hardmv_test = []
results_ds_train = []
results_ds_test = []
results_raykar_train = []
results_raykar_trainA = []
results_raykar_test = []
results_ours_global_train = []
results_ours_global_trainA = []
results_ours_global_test = []
results_ours_global_testA = []


############### MV/DS and calculate representations##############################
start_time = time.time()
label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels
print("Representation for Our/MV/D&S in %f mins"%((time.time()-start_time)/60.) )

if True: #version == 1 or version ==3:
    mv_onehot = label_I.mv_labels('onehot')
    mv_probas = label_I.mv_labels('probas')
    print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))

#Deterministic
ds_labels, ds_conf = label_I.DS_labels()
print("ACC D&S on train:",np.mean(ds_labels.argmax(axis=1)==Z_train))

#get representation needed for Raykar
start_time = time.time()
y_obs_categorical = set_representation(y_obs,'onehot') 
print("shape:",y_obs_categorical.shape)
print("Representation for Raykar in %f mins"%((time.time()-start_time)/60.) )

if version == 1 or version ==3: #normal
    #get our global representation 
    r_obs = label_I.y_obs_repeat.copy() #set_representation(y_obs_categorical,"repeat")
    print("vector of repeats:\n",r_obs)
    print("shape:",r_obs.shape)

    #analysis
    aux = [entropy(example)/np.log(r_obs.shape[1]) for example in mv_probas]
    print("Normalized entropy (0-1) of repeats annotations:",np.mean(aux))

for _ in range(30): #repetitions
    ############# EXECUTE ALGORITHMS #############################
    if version == 1 or version == 3:
        model_mvsoft = clone_model(model_UB) 
        model_mvsoft.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist = model_mvsoft.fit(Xstd_train, mv_probas, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over soft-MV, Epochs to converge =",len(hist.epoch))
        Z_train_pred_mvsoft = model_mvsoft.predict_classes(Xstd_train)
        Z_test_pred_mvsoft = model_mvsoft.predict_classes(Xstd_test)
        keras.backend.clear_session()
        
        model_mvhard = clone_model(model_UB) 
        model_mvhard.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_mvhard.fit(Xstd_train, mv_onehot, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over hard-MV, Epochs to converge =",len(hist.epoch))
        Z_train_pred_mvhard = model_mvhard.predict_classes(Xstd_train)
        Z_test_pred_mvhard = model_mvhard.predict_classes(Xstd_test)
        keras.backend.clear_session()

    model_ds = clone_model(model_UB) 
    model_ds.compile(loss='categorical_crossentropy',optimizer=OPT)
    hist=model_ds.fit(Xstd_train, ds_labels, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
    print("Trained model over D&S, Epochs to converge =",len(hist.epoch))
    Z_train_pred_ds = model_ds.predict_classes(Xstd_train)
    Z_test_pred_ds = model_ds.predict_classes(Xstd_test)
    keras.backend.clear_session()
    
    raykarMC = RaykarMC(Xstd_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
    raykarMC.define_model("mlp",128,1,BatchN=False,drop=0.5) 
    logL_hists,i_r = raykarMC.multiples_run(20,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
    Z_train_p_Ray = raykarMC.base_model.predict(Xstd_train)
    Z_test_pred_Ray = raykarMC.base_model.predict_classes(Xstd_test)
    keras.backend.clear_session()
    
    if version == 1 or version == 3:
        gMixture_Global = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=0,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Global.define_model("mlp",128,1,BatchN=False,drop=0.5)
        gMixture_Global.lambda_random = True #with lambda random --necessary
        logL_hists,i = gMixture_Global.multiples_run(20,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL
                                       ,cluster=True)
        Z_train_p_OG = gMixture_Global.base_model.predict(Xstd_train)
        Z_test_p_OG = gMixture_Global.base_model.predict(Xstd_test)
        keras.backend.clear_session()
   
    ################## MEASURE PERFORMANCE ##################################
    if version == 1 or version == 3:
        evaluate = Evaluation_metrics(model_mvsoft,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        prob_Yzt = np.tile(normalize(confusion_matrix(y_true=Z_train,y_pred=Z_train_pred_mvsoft),norm='l1'), (T,1,1) )
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvsoft,conf_pred=prob_Yzt,conf_true=confe_matrix)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvsoft)
        results_softmv_train += results1
        results_softmv_test += results2

        evaluate = Evaluation_metrics(model_mvhard,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        prob_Yzt = np.tile(normalize(confusion_matrix(y_true=Z_train,y_pred=Z_train_pred_mvhard),norm='l1'), (T,1,1) )
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvhard,conf_pred=prob_Yzt,conf_true=confe_matrix)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvhard)

        results_hardmv_train += results1
        results_hardmv_test += results2

    evaluate = Evaluation_metrics(model_ds,'keras',Xstd_train.shape[0],plot=False)
    evaluate.set_T_weights(T_weights)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_ds,conf_pred=ds_conf,conf_true=confe_matrix)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_ds)
    
    results_ds_train += results1
    results_ds_test += results2
    
    evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
    prob_Yzt = raykarMC.get_confusionM()
    prob_Yxt = raykarMC.get_predictions_annot(Xstd_train,data=Z_train_p_Ray)
    Z_train_pred_Ray = Z_train_p_Ray.argmax(axis=-1)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt)
    results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)
    
    results_raykar_train += results1
    results_raykar_trainA += results1_aux
    results_raykar_test += results2
    
    if version == 1 or version == 3:
        evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False) 
        aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True,p_z=Z_train_p_OG)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        Z_train_pred_OG = Z_train_p_OG.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Global.get_confusionM()
        y_o_groups = gMixture_Global.get_predictions_groups(Xstd_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_global_train +=  results1
        results_ours_global_trainA += results1_aux
        results_ours_global_testA.append(results2[0])
        results_ours_global_test.append(results2[1])
    
    print("All Performance Measured")
    if version == 1 or version ==3:
        del gMixture_Global,model_mvsoft,model_mvhard
    del model_ds,raykarMC,evaluate
    gc.collect()
    
#plot measures 
if version == 1 or version == 3:
    get_mean_dataframes(results_softmv_train).to_csv("LabelMe_softMV_train.csv",index=False)
    get_mean_dataframes(results_softmv_test).to_csv("LabelMe_softMV_test.csv",index=False)

    get_mean_dataframes(results_hardmv_train).to_csv("LabelMe_hardMV_train.csv",index=False)
    get_mean_dataframes(results_hardmv_test).to_csv("LabelMe_hardMV_test.csv",index=False)

get_mean_dataframes(results_ds_train).to_csv("LabelMe_DS_train.csv",index=False)
get_mean_dataframes(results_ds_test).to_csv("LabelMe_DS_test.csv",index=False)

get_mean_dataframes(results_raykar_train).to_csv("LabelMe_Raykar_train.csv",index=False)
get_mean_dataframes(results_raykar_trainA).to_csv("LabelMe_Raykar_trainAnn.csv",index=False)
get_mean_dataframes(results_raykar_test).to_csv("LabelMe_Raykar_test.csv",index=False)

if version == 1 or version ==3:
    get_mean_dataframes(results_ours_global_train).to_csv("LabelMe_OursGlobal_train.csv",index=False)
    get_mean_dataframes(results_ours_global_trainA).to_csv("LabelMe_OursGlobal_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_global_test).to_csv("LabelMe_OursGlobal_test.csv",index=False)
    get_mean_dataframes(results_ours_global_testA).to_csv("LabelMe_OursGlobal_testAux.csv",index=False)

print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))

