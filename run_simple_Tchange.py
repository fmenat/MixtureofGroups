from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/synthetic..)")

(opts, args) = op.parse_args()
path = opts.path
M_seted = opts.Ngroups 
state_sce = None

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
X_train = np.loadtxt(path+"/synthetic/simple/datasim_X_train.csv",delimiter=',')
Z_train = np.loadtxt(path+"/synthetic/simple/datasim_Z_train.csv",dtype='int') #groudn truth

X_test = np.loadtxt(path+"/synthetic/simple/datasim_X_test.csv",delimiter=',')
Z_test = np.loadtxt(path+"/synthetic/simple/datasim_Z_test.csv",dtype='int') #groudn truth

print("Input shape:",X_train.shape)

from sklearn.preprocessing import StandardScaler
std= StandardScaler(with_mean=True) #matrices sparse with_mean=False
std.fit(X_train)
Xstd_train = std.transform(X_train)
Xstd_test = std.transform(X_test)

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

model_UB = MLP_Keras(Xstd_train.shape[1:],Z_train_onehot.shape[1],16,1,BN=False,drop=0.2) #what about bn true?
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
hist = model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
print("Trained Ideal Model , Epochs to converge =",len(hist.epoch))

evaluate = Evaluation_metrics(model_UB,'keras',Xstd_train.shape[0],plot=False)
Z_train_pred = model_UB.predict_classes(Xstd_train)
results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred)
Z_test_pred = model_UB.predict_classes(Xstd_test)
results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)

results1[0].to_csv("synthetic_UpperBound_train.csv",index=False)
results2[0].to_csv("synthetic_UpperBound_test.csv",index=False)
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


from code.generate_data import SinteticData

#ANNOTATOR DENSITY CHOOSE
to_check = [100,500,1500,3500,6000,10000]
T_data = 5 


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
for Tmax in to_check:
    aux_softmv_train = []
    aux_softmv_test = []
    aux_hardmv_train = []
    aux_hardmv_test = []
    aux_ds_train = []
    aux_ds_test = []
    aux_raykar_train = []
    aux_raykar_trainA = []
    aux_raykar_test = []
    aux_ours_global_train = []
    aux_ours_global_trainA = []
    aux_ours_global_test = []
    aux_ours_global_testA = []
    
    GenerateData = SinteticData(state=state_sce) #por la semilla quedan similares..
    #CONFUSION MATRIX CHOOSE
    GenerateData.set_probas(asfile=True,file_matrix=path+'/synthetic/simple/matrix_datasim_normal.csv',file_groups =path+'/synthetic/simple/groups_datasim_normal.csv')
    real_conf_matrix = GenerateData.conf_matrix.copy()

    print("New Synthetic data is being generated...",flush=True,end='')
    y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False,hard=True)
    print("Done! ")
    if len(groups_annot.shape) ==1 or groups_annot.shape[1] ==  1: 
        groups_annot = keras.utils.to_categorical(groups_annot)  #only if it is hard clustering
    confe_matrix_R = np.tensordot(groups_annot,real_conf_matrix, axes=[[1],[0]])
    T_weights = np.sum(y_obs != -1,axis=0)
    print("Mean annotations by t= ",T_weights.mean())

    N,T = y_obs.shape
    K = np.max(y_obs)+1 # asumiendo que estan ordenadas
    print("Shape (data,annotators): ",(N,T))
    print("Classes: ",K)

    ############### MV/DS and calculate representations##############################
    label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels

    mv_probas, mv_conf_probas = label_I.mv_labels('probas')
    mv_onehot, mv_conf_onehot = label_I.mv_labels('onehot')
    confe_matrix_G = get_Global_confusionM(Z_train,label_I.y_obs_repeat)

    #Deterministic
    ds_labels, ds_conf = label_I.DS_labels()

    print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))
    print("ACC D&S on train:",np.mean(ds_labels.argmax(axis=1)==Z_train))

    #get representation needed for Raykar
    start_time = time.time()
    y_obs_categorical = set_representation(y_obs,'onehot') 
    print("shape:",y_obs_categorical.shape)
    print("Representation for Raykar in %f mins"%((time.time()-start_time)/60.) )

    #get our global representation 
    r_obs = label_I.y_obs_repeat.copy() #set_representation(y_obs_categorical,"repeat")
    print("vector of repeats:\n",r_obs)
    print("shape:",r_obs.shape)

    for _ in range(30): #repetitions
        ############# EXECUTE ALGORITHMS #############################
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
        raykarMC.define_model('mlp',16,1,BatchN=False,drop=0.2)
        logL_hists,i_r = raykarMC.multiples_run(20,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_Ray = raykarMC.base_model.predict(Xstd_train)
        Z_test_pred_Ray = raykarMC.base_model.predict_classes(Xstd_test)
        keras.backend.clear_session()

        gMixture_Global = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=0,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Global.define_model("mlp",16,1,BatchN=False,drop=0.2)
        gMixture_Global.lambda_random = False #with lambda random --necessary
        logL_hists,i = gMixture_Global.multiples_run(20,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL
                                       ,cluster=True)
        Z_train_p_OG = gMixture_Global.base_model.predict(Xstd_train)
        Z_test_p_OG = gMixture_Global.base_model.predict(Xstd_test)
        keras.backend.clear_session()

        ################## MEASURE PERFORMANCE ##################################
        evaluate = Evaluation_metrics(model_mvsoft,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        prob_Yzt = np.tile( mv_conf_probas, (T,1,1) )
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvsoft,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                     conf_true_G =confe_matrix_G, conf_pred_G = mv_conf_probas)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvsoft)

        aux_softmv_train += results1
        aux_softmv_test += results2

        evaluate = Evaluation_metrics(model_mvhard,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        prob_Yzt = np.tile( mv_conf_onehot, (T,1,1) )
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvhard,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                     conf_true_G =confe_matrix_G, conf_pred_G = mv_conf_onehot)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvhard)

        aux_hardmv_train += results1
        aux_hardmv_test += results2

        evaluate = Evaluation_metrics(model_ds,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_ds,conf_pred=ds_conf,conf_true=confe_matrix_R,
                                     conf_true_G =confe_matrix_G, conf_pred_G = ds_conf.mean(axis=0))
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_ds)

        aux_ds_train += results1
        aux_ds_test += results2

        evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
        prob_Yzt = raykarMC.get_confusionM()
        prob_Yxt = raykarMC.get_predictions_annot(Xstd_train,data=Z_train_p_Ray)
        Z_train_pred_Ray = Z_train_p_Ray.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                     y_o=y_obs,yo_pred=prob_Yxt,conf_true_G =confe_matrix_G, conf_pred_G = prob_Yzt.mean(axis=0))
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)

        aux_raykar_train += results1
        aux_raykar_trainA += results1_aux
        aux_raykar_test += results2

        evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False) 
        aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True,p_z=Z_train_p_OG)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        prob_Yz = gMixture_Global.calculate_Yz()
        Z_train_pred_OG = Z_train_p_OG.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                              y_o=y_obs,yo_pred=prob_Yxt,
                                             conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Global.get_confusionM()
        y_o_groups = gMixture_Global.get_predictions_groups(Xstd_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

        aux_ours_global_train +=  results1
        aux_ours_global_trainA += results1_aux
        aux_ours_global_testA.append(results2[0])
        aux_ours_global_test.append(results2[1])

        print("All Performance Measured")
        del model_mvsoft,model_mvhard,model_ds,raykarMC,gMixture_Global,evaluate
        gc.collect()

    #plot measures    
    results_softmv_train.append(get_mean_dataframes(aux_softmv_train))
    results_softmv_test.append(get_mean_dataframes(aux_softmv_test))
    results_hardmv_train.append(get_mean_dataframes(aux_hardmv_train))
    results_hardmv_test.append(get_mean_dataframes(aux_hardmv_test))
    results_ds_train.append(get_mean_dataframes(aux_ds_train))
    results_ds_test.append(get_mean_dataframes(aux_ds_test))
    results_raykar_train.append(get_mean_dataframes(aux_raykar_train))
    results_raykar_trainA.append(get_mean_dataframes(aux_raykar_trainA))
    results_raykar_test.append(get_mean_dataframes(aux_raykar_test))
    results_ours_global_train.append(get_mean_dataframes(aux_ours_global_train))
    results_ours_global_trainA.append(get_mean_dataframes(aux_ours_global_trainA))
    results_ours_global_test.append(get_mean_dataframes(aux_ours_global_test))
    results_ours_global_testA.append(get_mean_dataframes(aux_ours_global_testA))
    gc.collect()

import pickle
with open('synthetic_softMV_train.pickle', 'wb') as handle:
    pickle.dump(results_softmv_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_softMV_test.pickle', 'wb') as handle:
    pickle.dump(results_softmv_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_hardMV_train.pickle', 'wb') as handle:
    pickle.dump(results_hardmv_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_hardMV_test.pickle', 'wb') as handle:
    pickle.dump(results_hardmv_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_DS_train.pickle', 'wb') as handle:
    pickle.dump(results_ds_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_DS_test.pickle', 'wb') as handle:
    pickle.dump(results_ds_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_Raykar_train.pickle', 'wb') as handle:
    pickle.dump(results_raykar_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_Raykar_trainAnn.pickle', 'wb') as handle:
    pickle.dump(results_raykar_trainA, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_Raykar_test.pickle', 'wb') as handle:
    pickle.dump(results_raykar_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_OursGlobal_train.pickle', 'wb') as handle:
    pickle.dump(results_ours_global_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_OursGlobal_trainAnn.pickle', 'wb') as handle:
    pickle.dump(results_ours_global_trainA, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_OursGlobal_test.pickle', 'wb') as handle:
    pickle.dump(results_ours_global_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('synthetic_OursGlobal_testAux.pickle', 'wb') as handle:
    pickle.dump(results_ours_global_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))

