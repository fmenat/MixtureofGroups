from optparse import OptionParser
import sys, os
dirpath = os.getcwd().split("RodriguesImplementation")[0]#+"/code/"
sys.path.append(dirpath)

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/LabelMe_Z_train.....)")
op.add_option("-v", "--version", type="int", default=1, help="version of annotations (1, 2 or 3)")
op.add_option("-e", "--executed", type="string", default='', help="executed models separated by /.. ex (hardmv/ds/raykar)")

(opts, args) = op.parse_args()
folder = opts.path
M_seted = opts.Ngroups 
version = opts.version 
executed_models = opts.executed  #arg
if len(executed_models) == 0: #put alls
    executed_models = ["rodriguesR","cmmR","cmoaR"]
else:
    executed_models = executed_models.split("/") 

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

DTYPE_OP = 'float32'
keras.backend.set_floatx(DTYPE_OP)

if DTYPE_OP == 'float64':
    keras.backend.set_epsilon(np.finfo(np.float64).eps)
elif DTYPE_OP == 'float32':
    keras.backend.set_epsilon(np.finfo(np.float32).eps)
    
### Load Data
Xstd_train = np.load(folder+"/LabelMe_VGG_avg_train.npy")
Z_train = np.loadtxt(folder+"/LabelMe_Z_train.txt",dtype='int')

Xstd_val = np.load(folder+"/LabelMe_VGG_avg_valid.npy")
Z_val = np.loadtxt(folder+"/LabelMe_Z_valid.txt",dtype='int')

Xstd_test = np.load(folder+"/LabelMe_VGG_avg_test.npy")
Z_test = np.loadtxt(folder+"/LabelMe_Z_test.txt",dtype='int')

print("Input shape:",Xstd_train.shape)
print("Label shape:",Z_train.shape)


from codeE.learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from codeE.learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple, Clonable_Model #deep learning

from codeE.evaluation import Evaluation_metrics
from codeE.representation import *
from codeE.utils import *
from codeE.baseline import LabelInference, RaykarMC
from codeE.MixtureofGroups import GroupMixtureGlo, project_and_cluster,clusterize_annotators, GroupMixtureInd

### Delta convergence criteria
from codeE.utils import EarlyStopRelative
ourCallback = EarlyStopRelative(monitor='loss',patience=1,min_delta=TOL)

start_time_exec = time.time()

#upper bound model
Z_train_onehot = keras.utils.to_categorical(Z_train)

model_UB = MLP_Keras(Xstd_train.shape[1:],Z_train_onehot.shape[1],128,1,BN=False,drop=0.5)
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
hist = model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
print("Trained Ideal Model , Epochs to converge =",len(hist.epoch))
clone_UB = Clonable_Model(model_UB)

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
confe_matrix_R = np.zeros((T,K,K),dtype=DTYPE_OP)
for t in range(T):    
    for i in range(N):
        if y_obs[i,t] != -1:
            confe_matrix_R[t,Z_train[i],y_obs[i,t]] +=1
    mask_nan = confe_matrix_R[t,:,:].sum(axis=-1) == 0
    mean_replace = np.mean(confe_matrix_R[t,:,:][~mask_nan],axis=0)
    for value in np.arange(K)[mask_nan]:
        confe_matrix_R[t,value,:] =  1 #Rodrigues 1./K -- similar  to laplace smooth (prior 1)
    confe_matrix_R[t,:,:] = confe_matrix_R[t,:,:]/confe_matrix_R[t,:,:].sum(axis=-1,keepdims=True) #normalize


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
results_ours_indiv_T_train = []
results_ours_indiv_T_trainA = []
results_ours_indiv_T_test = []
results_ours_indiv_T_testA = []
results_ours_indiv_K_train = []
results_ours_indiv_K_trainA = []
results_ours_indiv_K_test = []
results_ours_indiv_K_testA = []

############### MV/DS and calculate representations##############################
#Deterministic
    mv_probas, mv_conf_probas = label_I.mv_labels('probas')
    mv_onehot, mv_conf_onehot = label_I.mv_labels('onehot')
    confe_matrix_G = get_Global_confusionM(Z_train,label_I.y_obs_repeat)


if "rodriguesR" in executed_models:
    #get representation needed for Raykar
    start_time = time.time()
    y_obs_categorical = set_representation(y_obs,'onehot') 
    print("shape:",y_obs_categorical.shape)
    print("Representation for Raykar in %f mins"%((time.time()-start_time)/60.) )

if "cmmR" in executed_models:
    #get our global representation 
    if "mv" in executed_models:
        r_obs = label_I.y_obs_repeat.copy() #
    elif "raykar" in executed_models:
        r_obs = set_representation(y_obs_categorical,"repeat")
    else:
        r_obs = set_representation(y_obs,"repeat")
    print("vector of repeats:\n",r_obs)
    print("shape:",r_obs.shape)

    #analysis
    if "mv" in executed_models:
        aux = [entropy(example)/np.log(r_obs.shape[1]) for example in mv_probas]
        print("Normalized entropy (0-1) of repeats annotations:",np.mean(aux))
    confe_matrix_G = get_Global_confusionM(Z_train,r_obs)

if "cmoaR" in executed_models:
    Y_ann_train, T_idx = set_representation(y_obs,"onehotvar")
    T_idx_unique = np.arange(T).reshape(-1,1)

    A = keras.utils.to_categorical(np.arange(T), num_classes=T) #fast way
    print("Annotator representation (T, R_t)=", A.shape)

    A_rep = np.zeros((T, K))
    for i in range(N):
        for l, t_idx in enumerate(T_idx[i]):
            obs_t = Y_ann_train[i][l].argmax(axis=-1)
            A_rep[t_idx, obs_t] += 1

for _ in range(30): #repetitions
    ############# EXECUTE ALGORITHMS #############################
    if "rodriguesR" in executed_models:
        CrowdL = RodriguesCrowdLayer(Xstd_train.shape[1:],y_obs_catmasked.shape[1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)

        raykarMC = RaykarMC(Xstd_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
        raykarMC.define_model("mlp",128,1,BatchN=False,drop=0.5) 
        logL_hists,i_r = raykarMC.multiples_run(20,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_Ray = raykarMC.get_predictions(Xstd_train)
        Z_test_pred_Ray = raykarMC.get_predictions(Xstd_test).argmax(axis=-1)
        keras.backend.clear_session()
    
    if "oursglobal" in executed_models:
        gMixture_Global = GroupMixtureGlo(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Global.define_model("mlp",128,1,BatchN=False,drop=0.5)
        logL_hists,i = gMixture_Global.multiples_run(20,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_OG = gMixture_Global.get_predictions(Xstd_train)
        Z_test_p_OG = gMixture_Global.get_predictions(Xstd_test)
        keras.backend.clear_session()

    if "oursindividual" in executed_models:
        gMixture_Ind_T = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Ind_T.define_model("mlp",128,1,BatchN=False,drop=0.5)
        gMixture_Ind_T.define_model_group("perceptron",T, M_seted, embed=True, embed_M=A, BatchN=True,bias=False)
        logL_hists,i_r = gMixture_Ind_T.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=[], batch_size=BATCH_SIZE,
                                             pre_init_g=0, pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_OI_T = gMixture_Ind_T.get_predictions_z(Xstd_train)
        Z_test_p_OI_T = gMixture_Ind_T.get_predictions_z(Xstd_test)
        prob_Gt_OI_T = gMixture_Ind_T.get_predictions_g(T_idx_unique) 
        keras.backend.clear_session()

        gMixture_Ind_K = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Ind_K.define_model("mlp",128,1,BatchN=False,drop=0.5)
        gMixture_Ind_K.define_model_group("mlp", A_rep.shape[1], K*M_seted, 1, embed=True, embed_M=A_rep)
        logL_hists,i_r = gMixture_Ind_K.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=A_rep, batch_size=BATCH_SIZE,
                                              pre_init_g=0,pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_OI_K = gMixture_Ind_K.get_predictions_z(Xstd_train)
        Z_test_p_OI_K  = gMixture_Ind_K.get_predictions_z(Xstd_test)
        prob_Gt_OI_K   = gMixture_Ind_K.get_predictions_g(T_idx_unique) 
        keras.backend.clear_session()


    ################## MEASURE PERFORMANCE ##################################
    if "rodriguesR" in executed_models:
        evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
        prob_Yzt = raykarMC.get_confusionM()
        prob_Yxt = raykarMC.get_predictions_annot(Xstd_train,data=Z_train_p_Ray)
        Z_train_pred_Ray = Z_train_p_Ray.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                         y_o=y_obs,yo_pred=prob_Yxt,conf_true_G =confe_matrix_G, conf_pred_G = prob_Yzt.mean(axis=0))
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)
        
        results_raykar_train += results1
        results_raykar_trainA += results1_aux
        results_raykar_test += results2

    if "oursglobal" in executed_models:
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

        results_ours_global_train +=  results1
        results_ours_global_trainA += results1_aux
        results_ours_global_testA.append(results2[0])
        results_ours_global_test.append(results2[1])

    if "oursindividual" in executed_models:
        evaluate = Evaluation_metrics(gMixture_Ind_T,'our1',plot=False) 
        evaluate.set_Gt(prob_Gt_OI_T)
        aux = gMixture_Ind_T.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=True,p_z=Z_train_p_OI_T,p_g=prob_Gt_OI_T)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        prob_Yz = gMixture_Ind_T.calculate_Yz(prob_Gt)
        Z_train_pred_OI = Z_train_p_OI_T.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                             y_o=y_obs,yo_pred=prob_Yxt,
                                            conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Ind_T.get_confusionM()
        y_o_groups = gMixture_Ind_T.get_predictions_groups(Xstd_test,data=Z_test_p_OI_T).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OI = Z_test_p_OI_T.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_indiv_T_train +=  results1
        results_ours_indiv_T_trainA += results1_aux
        results_ours_indiv_T_testA.append(results2[0])
        results_ours_indiv_T_test.append(results2[1])

        evaluate = Evaluation_metrics(gMixture_Ind_K,'our1',plot=False) 
        evaluate.set_Gt(prob_Gt_OI_K)
        aux = gMixture_Ind_K.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=True,p_z=Z_train_p_OI_K,p_g=prob_Gt_OI_K)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        prob_Yz = gMixture_Ind_K.calculate_Yz(prob_Gt)
        Z_train_pred_OI = Z_train_p_OI_K.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                             y_o=y_obs,yo_pred=prob_Yxt,
                                            conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Ind_K.get_confusionM()
        y_o_groups = gMixture_Ind_K.get_predictions_groups(Xstd_test,data=Z_test_p_OI_K).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OI = Z_test_p_OI_K.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_indiv_K_train +=  results1
        results_ours_indiv_K_trainA += results1_aux
        results_ours_indiv_K_testA.append(results2[0])
        results_ours_indiv_K_test.append(results2[1])

    print("All Performance Measured")
    if "rodriguesR" in executed_models:
        del raykarMC
    if "oursglobal" in executed_models:
        del gMixture_Global
    if "oursindividual" in executed_models:
        del gMixture_Ind_T, gMixture_Ind_K
    del evaluate
    gc.collect()
    
#plot measures 
if "rodriguesR" in executed_models:
    get_mean_dataframes(results_raykar_train).to_csv("LabelMe_Rodrigues_train.csv",index=False)
    get_mean_dataframes(results_raykar_train,  mean_std=False).to_csv("LabelMe_Rodrigues_train_std.csv",index=False)
    get_mean_dataframes(results_raykar_trainA).to_csv("LabelMe_Rodrigues_trainAnn.csv",index=False)
    get_mean_dataframes(results_raykar_trainA,  mean_std=False).to_csv("LabelMe_Rodrigues_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_raykar_test).to_csv("LabelMe_Rodrigues_test.csv",index=False)
    get_mean_dataframes(results_raykar_test, mean_std=False).to_csv("LabelMe_Rodrigues_test_std.csv",index=False)

if "oursglobal" in executed_models:
    get_mean_dataframes(results_ours_global_train).to_csv("LabelMe_OursGlobal_train.csv",index=False)
    get_mean_dataframes(results_ours_global_train, mean_std=False).to_csv("LabelMe_OursGlobal_train_std.csv",index=False)
    get_mean_dataframes(results_ours_global_trainA).to_csv("LabelMe_OursGlobal_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_global_trainA, mean_std=False).to_csv("LabelMe_OursGlobal_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ours_global_test).to_csv("LabelMe_OursGlobal_test.csv",index=False)
    get_mean_dataframes(results_ours_global_test, mean_std=False).to_csv("LabelMe_OursGlobal_test_std.csv",index=False)
    get_mean_dataframes(results_ours_global_testA).to_csv("LabelMe_OursGlobal_testAux.csv",index=False)
    get_mean_dataframes(results_ours_global_testA, mean_std=False).to_csv("LabelMe_OursGlobal_testAux_std.csv",index=False)

if "oursindividual" in executed_models:
    get_mean_dataframes(results_ours_indiv_T_train).to_csv("LabelMe_OursIndividualT_train.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_train, mean_std=False).to_csv("LabelMe_OursIndividualT_train_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_trainA).to_csv("LabelMe_OursIndividualT_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_trainA, mean_std=False).to_csv("LabelMe_OursIndividualT_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_test).to_csv("LabelMe_OursIndividualT_test.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_test, mean_std=False).to_csv("LabelMe_OursIndividualT_test_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_testA).to_csv("LabelMe_OursIndividualT_testAux.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_testA, mean_std=False).to_csv("LabelMe_OursIndividualT_testAux_std.csv",index=False)

    get_mean_dataframes(results_ours_indiv_K_train).to_csv("LabelMe_OursIndividualK_train.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_train, mean_std=False).to_csv("LabelMe_OursIndividualK_train_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_trainA).to_csv("LabelMe_OursIndividualK_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_trainA, mean_std=False).to_csv("LabelMe_OursIndividualK_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_test).to_csv("LabelMe_OursIndividualK_test.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_test, mean_std=False).to_csv("LabelMe_OursIndividualK_test_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_testA).to_csv("LabelMe_OursIndividualK_testAux.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_testA, mean_std=False).to_csv("LabelMe_OursIndividualK_testAux_std.csv",index=False)

print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))

