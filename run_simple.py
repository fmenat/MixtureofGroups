from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/synthetic..)")
op.add_option("-s", "--scenario", type="int", default='1', help="N scenario in where data is simmulated")

(opts, args) = op.parse_args()
path = opts.path
M_seted = opts.Ngroups 
scenario = opts.scenario  #arg

#GLOBAL Variables
BATCH_SIZE = 128
EPOCHS_BASE = 50
OPT = 'adam' #optimizer for neural network 
TOL = 1e-2 #tolerance for relative variation of parameters


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


#upper bound model
Z_train_onehot = keras.utils.to_categorical(Z_train)

model_UB = MLP_Keras(Xstd_train.shape[1:],Z_train_onehot.shape[1],8,1,BN=False,drop=0.2) #what about bn true?
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])

evaluate = Evaluation_metrics(model_UB,'keras',Xstd_train.shape[0],plot=False)
Z_train_pred = model_UB.predict_classes(Xstd_train)
results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred)
Z_test_pred = model_UB.predict_classes(Xstd_test)
results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)

results1[0].to_csv("synthetic_UpperBound_train.csv",index=False)
results2[0].to_csv("synthetic_UpperBound_test.csv",index=False)
del evaluate,Z_train_pred,Z_test_pred,results1,results2
gc.collect()

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

GenerateData = SinteticData()

#CONFUSION MATRIX CHOOSE
if scenario == 1 or scenario == 3 or scenario == 4 or scenario == 5 or scenario == 6:
    GenerateData.set_probas(asfile=True,file_matrix=path+'/synthetic/simple/matrix_datasim_normal.csv',file_groups =path+'/synthetic/simple/groups_datasim_normal.csv')

elif scenario == 2 or scenario == 7: #bad MV
    GenerateData.set_probas(asfile=True,file_matrix=path+'/synthetic/simple/matrix_datasim_badMV.csv',file_groups =path+'/synthetic/simple/groups_datasim_badMV.csv')

real_conf_matrix = GenerateData.conf_matrix.copy()

#ANNOTATOR DENSITY CHOOSE
if scenario == 1 or scenario ==2 or scenario == 3:
    Tmax = 100
    T_data = 10 
    
elif scenario == 4 or scenario == 7:
    Tmax = 2000
    T_data = 20 
    
elif scenario == 5:
    Tmax = 5000
    T_data = 25

elif scenario == 6:
    Tmax = 10000
    T_data = 40


results_softmv_train = []
results_softmv_train_A = [] #for Global KL
results_softmv_test = []
results_hardmv_train = []
results_hardmv_train_A = [] #for Global KL
results_hardmv_test = []
results_ds_train = []
results_ds_test = []
results_raykar_train = []
results_raykar_trainA = []
results_raykar_test = []
results_ours1_train = []
results_ours1_trainA = []
results_ours1_test = []
results_ours1_testA = []
results_ours2_train = []
results_ours2_trainA = []
results_ours2_test = []
results_ours2_testA = []
results_ours3_train = []
results_ours3_trainA = []
results_ours3_test = []
results_ours3_testA = []

for _ in range(20): #repetitions
    print("New Synthetic data is being generated...",flush=True,end='')
    if scenario == 3: #soft
        y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False,hard=False)
    else:
        y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False)
    print("Done! ")
    
    if len(groups_annot.shape) ==1 or groups_annot.shape[1] ==  1: 
        groups_annot = keras.utils.to_categorical(groups_annot)  #only if it is hard clustering
    confe_matrix = np.tensordot(groups_annot,real_conf_matrix, axes=[[1],[0]])

    N,T = y_obs.shape
    K = np.max(y_obs)+1 # asumiendo que estan ordenadas
    print("Shape (data,annotators): ",(N,T))
    print("Classes: ",K)
    
    ############# EXECUTE ALGORITHMS #############################

    label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels

    mv_onehot = label_I.mv_labels('onehot')
    mv_probas = label_I.mv_labels('probas')

    ds_labels, ds_conf = label_I.DS_labels()
    
    print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))
    print("ACC D&S on train:",np.mean(ds_labels.argmax(axis=1)==Z_train))
        
    model_mvsoft = clone_model(model_UB) 
    model_mvsoft.compile(loss='categorical_crossentropy',optimizer=OPT)
    model_mvsoft.fit(Xstd_train, mv_probas, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
    print("Trained model over soft-MV")

    model_mvhard = clone_model(model_UB) 
    model_mvhard.compile(loss='categorical_crossentropy',optimizer=OPT)
    model_mvhard.fit(Xstd_train, mv_onehot, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
    print("Trained model over hard-MV")

    model_ds = clone_model(model_UB) 
    model_ds.compile(loss='categorical_crossentropy',optimizer=OPT)
    model_ds.fit(Xstd_train, ds_labels, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
    print("Trained model over D&S")

    #get representation needed for Raykar
    y_obs_categorical = set_representation(y_obs,'onehot') 
    
    raykarMC = RaykarMC(Xstd_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
    raykarMC.define_model('mlp',8,1,BatchN=False,drop=0.2)
    logL_hist = raykarMC.stable_train(Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
    print("Trained model over Raykar")

    #get our representation 
    r_obs = set_representation(y_obs_categorical,"repeat")
    print("vector of repeats:\n",r_obs)
    print("shape:",r_obs.shape)
    
    #pre analysis
    annotators_pca = project_and_cluster(y_obs_categorical,DTYPE_OP=DTYPE_OP,printed=False)[0]
    print("Annotators PCA of annotations shape: ",annotators_pca.shape)

    aux = [entropy(example)/np.log(r_obs.shape[1]) for example in mv_probas]
    print("Normalized entropy (0-1) of repeats annotations:",np.mean(aux))
    
    gMixture1 = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=5,optimizer=OPT,dtype_op=DTYPE_OP) 
    gMixture1.define_model("mlp",8,1,BatchN=False,drop=0.2)
    gMixture1.lambda_random = False #lambda=1     
    logL_hists,i  = gMixture1.multiples_run(1,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL*2
                                       ,cluster=True,bulk_annotators=[y_obs_categorical,annotators_pca])
    print("Trained model over Ours (1)")
    
    gMixture2 = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=5,optimizer=OPT,dtype_op=DTYPE_OP) 
    gMixture2.define_model("mlp",8,1,BatchN=False,drop=0.2)
    gMixture2.lambda_random = True #lambda random
    logL_hists,i = gMixture2.multiples_run(1,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL*2
                                       ,cluster=True,bulk_annotators=[y_obs_categorical,annotators_pca])
    print("Trained model over Ours (2)")
    
    gMixture3 = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=5,optimizer=OPT,dtype_op=DTYPE_OP) 
    gMixture3.define_model("mlp",8,1,BatchN=False,drop=0.2)
    gMixture3.lambda_random = True #with lambda random --necessary
    logL_hists,i = gMixture3.multiples_run(1,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL*2
                                   ,cluster=True)
    print("Trained model over Ours (3)")

    
    ################## MEASURE PERFORMANCE ##################################
    
    evaluate = Evaluation_metrics(model_mvsoft,'keras',Xstd_train.shape[0],plot=False)
    Z_train_p = model_mvsoft.predict(Xstd_train)
    prob_Yzt = get_confusionM(Z_train_p,y_obs_categorical)
    Z_train_pred = Z_train_p.argmax(axis=1)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)
    prob_Yzt = np.tile(confusion_matrix(y_true=Z_train,y_pred=Z_train_pred), (T,1,1) )
    results1_A = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)
    Z_test_pred = model_mvsoft.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)
    
    results_softmv_train += results1
    results_softmv_train_A += results1_A
    results_softmv_test += results2

    evaluate = Evaluation_metrics(model_mvhard,'keras',Xstd_train.shape[0],plot=False)
    Z_train_p = model_mvhard.predict(Xstd_train)
    prob_Yzt = get_confusionM(Z_train_p,y_obs_categorical)
    Z_train_pred = Z_train_p.argmax(axis=1)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)
    prob_Yzt = np.tile(confusion_matrix(y_true=Z_train,y_pred=Z_train_pred), (T,1,1) )
    results1_A = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)
    Z_test_pred = model_mvhard.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)
    
    results_hardmv_train += results1
    results_hardmv_train_A += results1_A
    results_hardmv_test += results2

    evaluate = Evaluation_metrics(model_ds,'keras',Xstd_train.shape[0],plot=False)
    Z_train_pred = model_ds.predict_classes(Xstd_train)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=ds_conf,conf_true=confe_matrix)
    Z_test_pred = model_ds.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)
    
    results_ds_train += results1
    results_ds_test += results2
    
    evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
    Z_train_pred = raykarMC.base_model.predict_classes(Xstd_train)
    prob_Yzt = raykarMC.get_confusionM()
    prob_Yxt = raykarMC.get_predictions_annot(Xstd_train)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt)

    results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)

    Z_test_pred = raykarMC.base_model.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)
    
    results_raykar_train += results1
    results_raykar_trainA += results1_aux
    results_raykar_test += results2

    evaluate = Evaluation_metrics(gMixture1,'our1',plot=False) 
    aux = gMixture1.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True)
    predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
    Z_train_pred = gMixture1.base_model.predict_classes(Xstd_train)
    y_o_groups = predictions_m.argmax(axis=-1)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt, y_o_groups=y_o_groups)

    results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)

    c_M = gMixture1.get_confusionM()
    y_o_groups = gMixture1.get_predictions_groups(Xstd_test).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
    Z_test_pred = gMixture1.base_model.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)
    
    results_ours1_train  += results1
    results_ours1_trainA += results1_aux
    results_ours1_testA.append(results2[0])
    results_ours1_test.append(results2[1])

    evaluate = Evaluation_metrics(gMixture2,'our1',plot=False) 
    aux = gMixture2.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True)
    predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
    Z_train_pred = gMixture2.base_model.predict_classes(Xstd_train)
    y_o_groups = predictions_m.argmax(axis=-1)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt, y_o_groups=y_o_groups)

    results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)

    c_M = gMixture2.get_confusionM()
    y_o_groups = gMixture2.get_predictions_groups(Xstd_test).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
    Z_test_pred = gMixture2.base_model.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)
    
    results_ours2_train += results1
    results_ours2_trainA += results1_aux
    results_ours2_testA.append(results2[0])
    results_ours2_test.append(results2[1])

    evaluate = Evaluation_metrics(gMixture3,'our1',plot=False) 
    aux = gMixture3.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True)
    predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
    Z_train_pred = gMixture3.base_model.predict_classes(Xstd_train)
    y_o_groups = predictions_m.argmax(axis=-1)
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt, y_o_groups=y_o_groups)

    results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)

    c_M = gMixture3.get_confusionM()
    y_o_groups = gMixture3.get_predictions_groups(Xstd_test).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
    Z_test_pred = gMixture3.base_model.predict_classes(Xstd_test)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)

    results_ours3_train +=  results1
    results_ours3_trainA += results1_aux
    results_ours3_testA.append(results2[0])
    results_ours3_test.append(results2[1])
    
    print("All Performance Measured")
    del model_mvsoft,model_mvhard,model_ds
    gc.collect()

#plot measures    
get_mean_dataframes(results_softmv_train).to_csv("synthetic_softMV_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_softmv_train_A).to_csv("synthetic_softMV_trainG_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_softmv_test).to_csv("synthetic_softMV_test_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_hardmv_train).to_csv("synthetic_hardMV_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_hardmv_train_A).to_csv("synthetic_hardMV_trainG_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_hardmv_test).to_csv("synthetic_hardMV_test_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_ds_train).to_csv("synthetic_DS_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ds_test).to_csv("synthetic_DS_test_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_raykar_train).to_csv("synthetic_Raykar_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_raykar_trainA).to_csv("synthetic_Raykar_trainAnn_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_raykar_test).to_csv("synthetic_Raykar_test_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_ours1_train).to_csv("synthetic_Ours1_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours1_trainA).to_csv("synthetic_Ours1_trainAnn_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours1_test).to_csv("synthetic_Ours1_test_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours1_testA).to_csv("synthetic_Ours1_testAux_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_ours2_train).to_csv("synthetic_Ours2_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours2_trainA).to_csv("synthetic_Ours2_trainAnn_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours2_test).to_csv("synthetic_Ours2_test_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours2_testA).to_csv("synthetic_Ours2_testAux_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_ours3_train).to_csv("synthetic_Ours3_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours3_trainA).to_csv("synthetic_Ours3_trainAnn_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours3_test).to_csv("synthetic_Ours3_test_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours3_testA).to_csv("synthetic_Ours3_testAux_s"+str(scenario)+".csv",index=False)
