from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/synthetic..)")
op.add_option("-s", "--scenario", type="int", default='1', help="N scenario in where data is simmulated")

(opts, args) = op.parse_args()
path = opts.path
M_seted = opts.Ngroups 
scenario = opts.scenario  #arg
state_sce = path+"/synthetic/CIFAR/state_CIFAR_s"+str(scenario)+".pickle" #once this work
#state_sce = None

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
from keras.models import clone_model

DTYPE_OP = 'float32'
keras.backend.set_floatx(DTYPE_OP)

if DTYPE_OP == 'float64':
    keras.backend.set_epsilon(np.finfo(np.float64).eps)
elif DTYPE_OP == 'float32':
    keras.backend.set_epsilon(np.finfo(np.float32).eps)
    
### Load Data and preprocess
from keras.datasets import cifar10
(X_train, Z_train), (X_test, Z_test) = cifar10.load_data()
print("Input shape:",X_train.shape)

Xstd_train = X_train.astype(DTYPE_OP)/255
Xstd_test = X_test.astype(DTYPE_OP)/255
Z_train = Z_train[:,0]
Z_test = Z_test[:,0]

from code.learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from code.learning_models import default_CNN,default_RNN,default_RNNw_emb,CNN_simple, RNN_simple #deep learning

from code.evaluation import Evaluation_metrics
from code.representation import *
from code.utils import *
from code.baseline import LabelInference, RaykarMC
from code.MixtureofGroupsV2 import GroupMixtureOpt, project_and_cluster,clusterize_annotators

### Delta convergence criteria
from code.utils import EarlyStopRelative
ourCallback = EarlyStopRelative(monitor='loss',patience=1,min_delta=TOL)

start_time_exec = time.time()

#upper bound model
Z_train_onehot = keras.utils.to_categorical(Z_train)
model_UB = default_CNN(Xstd_train.shape[1:],Z_train_onehot.shape[1])
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
hist = model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
print("Trained Ideal Model, Epochs to converge =",len(hist.epoch))

evaluate = Evaluation_metrics(model_UB,'keras',Xstd_train.shape[0],plot=False)
Z_train_pred = model_UB.predict_classes(Xstd_train,verbose=0)
results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred)
Z_test_pred = model_UB.predict_classes(Xstd_test,verbose=0)
results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)

results1[0].to_csv("simCIFAR_UpperBound_train.csv",index=False)
results2[0].to_csv("simCIFAR_UpperBound_test.csv",index=False)
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

GenerateData = SinteticData(state=state_sce)

#CONFUSION MATRIX CHOOSE
if scenario == 1 or scenario == 3 or scenario == 4 or scenario ==5 or scenario==7:
    GenerateData.set_probas(asfile=True,file_matrix=path+'/synthetic/CIFAR/matrix_CIFAR_normal.csv',file_groups =path+'/synthetic/CIFAR/groups_CIFAR_normal.csv')

elif scenario == 2 or scenario == 6: #bad MV
    GenerateData.set_probas(asfile=True,file_matrix=path+'/synthetic/CIFAR/matrix_CIFAR_badMV.csv',file_groups =path+'/synthetic/CIFAR/groups_CIFAR_badMV.csv')

real_conf_matrix = GenerateData.conf_matrix.copy()

#ANNOTATOR DENSITY CHOOSE
if scenario == 1 or scenario ==2 or scenario == 3:
    Tmax = 100
    T_data = 10 
    
elif scenario == 4 or scenario == 6 or scenario == 7: 
    Tmax = 2000
    T_data = 10 
    
elif scenario == 5:
    Tmax = 10000
    T_data = 20 #revisar eso


results_softmv_train = []
results_softmv_test = []
results_hardmv_train = []
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
results_ours_global_train = []
results_ours_global_trainA = []
results_ours_global_test = []
results_ours_global_testA = []


print("New Synthetic data is being generated...",flush=True,end='')
if scenario == 3 or scenario==7: #soft
    y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False,hard=False)
else:
    y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False)
print("Done! ")

if len(groups_annot.shape) ==1 or groups_annot.shape[1] ==  1: 
    groups_annot = keras.utils.to_categorical(groups_annot)  #only if it is hard clustering
confe_matrix = np.tensordot(groups_annot,real_conf_matrix, axes=[[1],[0]])
T_weights = np.sum(y_obs != -1,axis=0) #weight of annotators (how much appear in data)

N,T = y_obs.shape
K = np.max(y_obs)+1 # asumiendo que estan ordenadas
print("Shape (data,annotators): ",(N,T))
print("Classes: ",K)

############### MV/DS and calculate representations##############################
start_time = time.time()
label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels
print("Representation for Our/MV/D&S in %f mins"%((time.time()-start_time)/60.) )

mv_onehot = label_I.mv_labels('onehot')
mv_probas = label_I.mv_labels('probas')

print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))

#get our global representation 
r_obs = label_I.y_obs_repeat.copy() #set_representation(y_obs_categorical,"repeat")
print("vector of repeats:\n",r_obs)
print("shape:",r_obs.shape)

aux = [entropy(example)/np.log(r_obs.shape[1]) for example in mv_probas]
print("Normalized entropy (0-1) of repeats annotations:",np.mean(aux))        


if Tmax <3000: #other wise cannot be done
    #Deterministic
    ds_labels, ds_conf = label_I.DS_labels()
    print("ACC D&S on train:",np.mean(ds_labels.argmax(axis=1)==Z_train))

    #get representation needed for Raykar
    start_time = time.time()
    y_obs_categorical = set_representation(y_obs,'onehot') 
    print("shape:",y_obs_categorical.shape)
    print("Representation for Raykar in %f mins"%((time.time()-start_time)/60.) )

"""
if Tmax >= 3000 and Tmax<8000: #error de memoria solucionado con truncated SVD
    #pre analysis
    start_time = time.time()
    annotators_pca = project_and_cluster(y_obs_categorical,DTYPE_OP=DTYPE_OP,printed=False,mode_project="tpca")[0]
    print("Projection of annotations in %f mins"%((time.time()-start_time)/60.) )
    print("Annotators PCA of annotations shape: ",annotators_pca.shape)
elif Tmax< 3000:
    #pre analysis
    start_time = time.time()
    annotators_pca = project_and_cluster(y_obs_categorical,DTYPE_OP=DTYPE_OP,printed=False,mode_project="pca")[0]
    print("Projection of annotations in %f mins"%((time.time()-start_time)/60.) )
    print("Annotators PCA of annotations shape: ",annotators_pca.shape)
"""

for _ in range(5): #repetitions --- si se demora mucho bajar a 5
    ############# EXECUTE ALGORITHMS #############################
    model_mvsoft = clone_model(model_UB) 
    model_mvsoft.compile(loss='categorical_crossentropy',optimizer=OPT)
    hist=model_mvsoft.fit(Xstd_train, mv_probas, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
    print("Trained model over soft-MV, Epochs to converge =",len(hist.epoch))
    Z_train_pred_mvsoft = model_mvsoft.predict_classes(Xstd_train)
    Z_test_pred_mvsoft = model_mvsoft.predict_classes(Xstd_test)

    model_mvhard = clone_model(model_UB) 
    model_mvhard.compile(loss='categorical_crossentropy',optimizer=OPT)
    hist=model_mvhard.fit(Xstd_train, mv_onehot, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
    print("Trained model over hard-MV, Epochs to converge =",len(hist.epoch))
    Z_train_pred_mvhard = model_mvhard.predict_classes(Xstd_train)
    Z_test_pred_mvhard = model_mvhard.predict_classes(Xstd_test)
    
    if Tmax <3000: #other wise cannot be done
        model_ds = clone_model(model_UB) 
        model_ds.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_ds.fit(Xstd_train, ds_labels, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over D&S, Epochs to converge =",len(hist.epoch))
        Z_train_pred_ds = model_ds.predict_classes(Xstd_train)
        Z_test_pred_ds = model_ds.predict_classes(Xstd_test)

        raykarMC = RaykarMC(Xstd_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
        raykarMC.define_model("default cnn")
        logL_hists,i_r = raykarMC.multiples_run(15,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_Ray = raykarMC.base_model.predict(Xstd_train)
        Z_test_pred_Ray = raykarMC.base_model.predict_classes(Xstd_test)

    """
    if scenario != 5:
        gMixture1 = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=0,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture1.define_model("default cnn")
        gMixture1.lambda_random = False #lambda=1     
        logL_hists,i  = gMixture1.multiples_run(1,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL
                                           ,cluster=True,bulk_annotators=[y_obs_categorical,annotators_pca])
        print("Trained model over Ours (1)")

        gMixture2 = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=0,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture2.define_model("default cnn")
        gMixture2.lambda_random = True #lambda random
        logL_hists,i = gMixture2.multiples_run(1,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL
                                           ,cluster=True,bulk_annotators=[y_obs_categorical,annotators_pca])
        print("Trained model over Ours (2)")
    """
    gMixture_Global = GroupMixtureOpt(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,pre_init=0,optimizer=OPT,dtype_op=DTYPE_OP) 
    gMixture_Global.define_model("default cnn")
    gMixture_Global.lambda_random = True #with lambda random --necessary
    logL_hists,i = gMixture_Global.multiples_run(15,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL
                                   ,cluster=True)
    Z_train_p_OG = gMixture_Global.base_model.predict(Xstd_train)
    Z_test_p_OG = gMixture_Global.base_model.predict(Xstd_test)

    
    ################## MEASURE PERFORMANCE ##################################
    evaluate = Evaluation_metrics(model_mvsoft,'keras',Xstd_train.shape[0],plot=False)
    evaluate.set_T_weights(T_weights)
    prob_Yzt = np.tile(confusion_matrix(y_true=Z_train,y_pred=Z_train_pred_mvsoft), (T,1,1) )
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvsoft,conf_pred=prob_Yzt,conf_true=confe_matrix)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvsoft)
    
    results_softmv_train += results1
    results_softmv_test += results2

    evaluate = Evaluation_metrics(model_mvhard,'keras',Xstd_train.shape[0],plot=False)
    evaluate.set_T_weights(T_weights)
    prob_Yzt = np.tile(confusion_matrix(y_true=Z_train,y_pred=Z_train_pred_mvhard), (T,1,1) )
    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvhard,conf_pred=prob_Yzt,conf_true=confe_matrix)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvhard)
    
    results_hardmv_train += results1
    results_hardmv_test += results2
    
    if Tmax <3000: #other wise cannot be done
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
        """
        evaluate = Evaluation_metrics(gMixture1,'our1',plot=False) 
        aux = gMixture1.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        Z_train_pred = gMixture1.base_model.predict_classes(Xstd_train,verbose=0)
        #y_o_groups = predictions_m.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        #else: #pred annotator memory error
        #    aux = gMixture1.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=False)
        #    predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
        #    Z_train_pred = gMixture1.base_model.predict_classes(Xstd_train,verbose=0)
        #    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)      
        #    results1_aux = [None]
        c_M = gMixture1.get_confusionM()
        y_o_groups = gMixture1.get_predictions_groups(Xstd_test).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred = gMixture1.base_model.predict_classes(Xstd_test,verbose=0)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours1_train  += results1
        results_ours1_trainA += results1_aux
        results_ours1_testA.append(results2[0])
        results_ours1_test.append(results2[1])

        evaluate = Evaluation_metrics(gMixture2,'our1',plot=False) 
        aux = gMixture2.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        Z_train_pred = gMixture2.base_model.predict_classes(Xstd_train,verbose=0)
        #y_o_groups = predictions_m.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        #else: #pred annotator memory error
        #    aux = gMixture2.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=False)
        #    predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
        #    Z_train_pred = gMixture2.base_model.predict_classes(Xstd_train,verbose=0)
        #    results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)      
        #    results1_aux = [None]
        c_M = gMixture2.get_confusionM()
        y_o_groups = gMixture2.get_predictions_groups(Xstd_test).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred = gMixture2.base_model.predict_classes(Xstd_test,verbose=0)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours2_train += results1
        results_ours2_trainA += results1_aux
        results_ours2_testA.append(results2[0])
        results_ours2_test.append(results2[1])
        """
        
    evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False)  #no explota
    Z_train_pred_OG = Z_train_p_OG.argmax(axis=-1)
    if Tmax < 3000:
        aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True,p_z=Z_train_p_OG)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
    else: #pred annotator memory error
        aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=False,p_z=Z_train_p_OG)
        predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix)      
        results1_aux = [None]    
    c_M = gMixture_Global.get_confusionM()
    y_o_groups = gMixture_Global.get_predictions_groups(Xstd_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
    Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
    results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)
    
    results_ours_global_train +=  results1
    results_ours_global_trainA += results1_aux
    results_ours_global_testA.append(results2[0])
    results_ours_global_test.append(results2[1])
    
    print("All Performance Measured")
    del model_mvsoft,model_mvhard,gMixture_Global,evaluate
    if Tmax <3000: #other wise cannot be done
        del model_ds,raykarMC
    gc.collect()
    keras.backend.clear_session()

#plot measures    
get_mean_dataframes(results_softmv_train).to_csv("simCIFAR_softMV_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_softmv_test).to_csv("simCIFAR_softMV_test_s"+str(scenario)+".csv",index=False)

get_mean_dataframes(results_hardmv_train).to_csv("simCIFAR_hardMV_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_hardmv_test).to_csv("simCIFAR_hardMV_test_s"+str(scenario)+".csv",index=False)

if scenario != 5: #tmax >50000
    get_mean_dataframes(results_ds_train).to_csv("simCIFAR_DS_train_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_ds_test).to_csv("simCIFAR_DS_test_s"+str(scenario)+".csv",index=False)

    get_mean_dataframes(results_raykar_train).to_csv("simCIFAR_Raykar_train_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_raykar_test).to_csv("simCIFAR_Raykar_test_s"+str(scenario)+".csv",index=False)
    """
    get_mean_dataframes(results_ours1_train).to_csv("simCIFAR_Ours1_train_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_ours1_test).to_csv("simCIFAR_Ours1_test_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_ours1_testA).to_csv("simCIFAR_Ours1_testAux_s"+str(scenario)+".csv",index=False)

    get_mean_dataframes(results_ours2_train).to_csv("simCIFAR_Ours2_train_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_ours2_test).to_csv("simCIFAR_Ours2_test_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_ours2_testA).to_csv("simCIFAR_Ours2_testAux_s"+str(scenario)+".csv",index=False)
    """
get_mean_dataframes(results_ours_global_train).to_csv("simCIFAR_OursGlobal_train_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours_global_test).to_csv("simCIFAR_OursGlobal_test_s"+str(scenario)+".csv",index=False)
get_mean_dataframes(results_ours_global_testA).to_csv("simCIFAR_OursGlobal_testAux_s"+str(scenario)+".csv",index=False)

if Tmax < 3000: #calcualte pred annotators
    get_mean_dataframes(results_raykar_trainA).to_csv("simCIFAR_Raykar_trainAnn_s"+str(scenario)+".csv",index=False)
    #get_mean_dataframes(results_ours1_trainA).to_csv("simCIFAR_Ours1_trainAnn_s"+str(scenario)+".csv",index=False)
    #get_mean_dataframes(results_ours2_trainA).to_csv("simCIFAR_Ours2_trainAnn_s"+str(scenario)+".csv",index=False)
    get_mean_dataframes(results_ours_global_trainA).to_csv("simCIFAR_OursGlobal_trainAnn_s"+str(scenario)+".csv",index=False)

print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))
