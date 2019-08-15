from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/LabelMe_Z_train.....)")
op.add_option("-v", "--version", type="int", default=1, help="version of annotations (1, 2 or 3)")
op.add_option("-e", "--executed", type="string", default='', help="executed models separated by /.. ex (hardmv/ds/raykar)")
op.add_option("-b", "--perror", type=float, default=0.5, help="probability of error")
op.add_option("-q", "--question", type="string", default='Q4', help="Question to execute Fashion. ex (Q1,Q3,Q4,Q6)")

(opts, args) = op.parse_args()
folder = opts.path
M_seted = opts.Ngroups 
version = opts.version 
executed_models = opts.executed  #arg
if len(executed_models) == 0: #put alls
    executed_models = ["mv","ds","raykar","oursglobal","oursindividual"]
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
    
Q_s = opts.question#"Q4" #question selected
p_error = str(opts.perror) #"0.5"

### Load Data
X_vgg16 = np.load(folder+"Fashion10000_VGG_avg_all.npy")
Z_data = np.loadtxt(folder+"Z_data_"+Q_s+".txt",dtype='int')

sets = np.loadtxt(folder+"sets_"+p_error+"_"+Q_s+".txt", dtype='U')
mask_train = sets == "train"
mask_test = sets == "test"

Z_train = Z_data[mask_train]
Z_test = Z_data[mask_test]

X_train_vgg16 = X_vgg16[mask_train]
X_test_vgg16 = X_vgg16[mask_test]


#created before..
mask_val = np.random.rand(Z_train.shape[0]) < 0.33 ##tenerlo guardado antes..

#re build test set..
Xstd_test = X_train_vgg16[mask_val]
Xstd_train = X_train_vgg16[~mask_val]
Z_test = Z_train[mask_val]
Z_train = Z_train[~mask_val]

print("Input train shape:",X_train_vgg16.shape)
print("Label train shape:",Z_train.shape)

print("Input test shape:",X_test_vgg16.shape)
print("Label test shape:",Z_test.shape)


from code.learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from code.learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple, Clonable_Model #deep learning

from code.evaluation import Evaluation_metrics
from code.representation import *
from code.utils import *
from code.baseline import LabelInference, RaykarMC
from code.MixtureofGroups import GroupMixtureGlo, project_and_cluster,clusterize_annotators, GroupMixtureInd

### Delta convergence criteria
from code.utils import EarlyStopRelative
ourCallback = EarlyStopRelative(monitor='loss',patience=1,min_delta=TOL)

start_time_exec = time.time()

#upper bound model
Z_train_onehot = keras.utils.to_categorical(Z_train)

model_UB = MLP_Keras(Xstd_train.shape[1:],Z_train_onehot.shape[1],128,1,BN=True,drop=0.5) #en Q3 false va mejor
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
hist = model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
print("Trained Ideal Model , Epochs to converge =",len(hist.epoch))
clone_UB = Clonable_Model(model_UB)

evaluate = Evaluation_metrics(model_UB,'keras',Xstd_train.shape[0],plot=False)
Z_train_pred = model_UB.predict_classes(Xstd_train)
results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred)
Z_test_pred = model_UB.predict_classes(Xstd_test)
results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred)

results1[0].to_csv("Fashion_UpperBound_train.csv",index=False)
results2[0].to_csv("Fashion_UpperBound_test.csv",index=False) #varias veces...
del evaluate,Z_train_pred,Z_test_pred,results1,results2
gc.collect()
keras.backend.clear_session()

def get_mean_dataframes(df_values, mean_std = True):
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
    if mean_std:
        RT[:] = np.mean(data,axis=0)
    else:
        RT[:] = np.std(data,axis=0)
    
    if df_values[0].iloc[:,0].dtype == object:
        RT.insert(0, "", df_values[0].iloc[:,0].values )
    return RT


# data from Amazon Mechanical Turk
print("Loading AMT data...")
if version == 1:
    file_annot = "/answers_"+p_error+"_"+Q_s+".txt"
else:
    file_annot = "/answers_v"+str(version)+"_"+p_error+"_"+Q_s+".txt"
y_obs = np.loadtxt(folder+file_annot,dtype='int16') #not annotation symbol ==-1
y_obs = y_obs[~mask_val] 

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
results_raykar_test = []
results_ours_global_train = []
results_ours_global_test = []
results_ours_global_testA = []
results_ours_indiv_T_train = []
results_ours_indiv_T_test = []
results_ours_indiv_T_testA = []
results_ours_indiv_K_train = []
results_ours_indiv_K_test = []
results_ours_indiv_K_testA = []

############### MV/DS and calculate representations##############################
if "mv" in executed_models or "ds" in executed_models:
    start_time = time.time()
    label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels
    print("Representation for Our/MV/D&S in %f mins"%((time.time()-start_time)/60.) )

#Deterministic
if "mv" in executed_models:
    mv_probas, mv_conf_probas = label_I.mv_labels('probas')
    mv_onehot, mv_conf_onehot = label_I.mv_labels('onehot')
    confe_matrix_G = get_Global_confusionM(Z_train,label_I.y_obs_repeat)
    print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))

if "ds" in executed_models:
    ds_labels, ds_conf = label_I.DS_labels()
    print("ACC D&S on train:",np.mean(ds_labels.argmax(axis=1)==Z_train))

if "raykar" in executed_models:
    #get representation needed for Raykar
    start_time = time.time()
    y_obs_categorical = set_representation(y_obs,'onehot') 
    print("shape:",y_obs_categorical.shape)
    print("Representation for Raykar in %f mins"%((time.time()-start_time)/60.) )

if "oursglobal" in executed_models:
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

if "oursindividual" in executed_models:
    Y_ann_train, T_idx = set_representation(y_obs,"onehotvar")
    T_idx_unique = np.arange(T).reshape(-1,1)

    A = keras.utils.to_categorical(np.arange(T), num_classes=T) #fast way
    print("Annotator representation (T, R_t)=", A.shape)

    A_rep = np.zeros((T, K))
    for i in range(N):
        for l, t_idx in enumerate(T_idx[i]):
            obs_t = Y_ann_train[i][l].argmax(axis=-1)
            A_rep[t_idx, obs_t] += 1

for _ in range(20): #repetitions
    ############# EXECUTE ALGORITHMS #############################
    if "mv" in executed_models:
        model_mvsoft = clone_UB.get_model() 
        model_mvsoft.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist = model_mvsoft.fit(Xstd_train, mv_probas, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over soft-MV, Epochs to converge =",len(hist.epoch))
        Z_train_pred_mvsoft = model_mvsoft.predict_classes(Xstd_train)
        Z_test_pred_mvsoft = model_mvsoft.predict_classes(Xstd_test)
        keras.backend.clear_session()

        model_mvhard = clone_UB.get_model() 
        model_mvhard.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_mvhard.fit(Xstd_train, mv_onehot, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over hard-MV, Epochs to converge =",len(hist.epoch))
        Z_train_pred_mvhard = model_mvhard.predict_classes(Xstd_train)
        Z_test_pred_mvhard = model_mvhard.predict_classes(Xstd_test)
        keras.backend.clear_session()

    if "ds" in executed_models:
        model_ds = clone_UB.get_model() 
        model_ds.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_ds.fit(Xstd_train, ds_labels, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over D&S, Epochs to converge =",len(hist.epoch))
        Z_train_pred_ds = model_ds.predict_classes(Xstd_train)
        Z_test_pred_ds = model_ds.predict_classes(Xstd_test)
        keras.backend.clear_session()

    if "raykar" in executed_models:
        raykarMC = RaykarMC(Xstd_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
        raykarMC.define_model("mlp",128,1,BatchN=True,drop=0.5) 
        logL_hists,i_r = raykarMC.multiples_run(20,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        Z_train_p_Ray = raykarMC.get_predictions(Xstd_train)
        Z_test_pred_Ray = raykarMC.get_predictions(Xstd_test).argmax(axis=-1)
        keras.backend.clear_session()
    
    if "oursglobal" in executed_models:
        gMixture_Global = GroupMixtureGlo(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Global.define_model("mlp",128,1,BatchN=True,drop=0.5)
        logL_hists,i = gMixture_Global.multiples_run(20,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL*5/3)
        Z_train_p_OG = gMixture_Global.get_predictions(Xstd_train)
        Z_test_p_OG = gMixture_Global.get_predictions(Xstd_test)
        keras.backend.clear_session()

    if "oursindividual" in executed_models:
        gMixture_Ind_T = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Ind_T.define_model("mlp",128,1,BatchN=True,drop=0.5)
        gMixture_Ind_T.define_model_group("perceptron",T, M_seted, embed=True, embed_M=A, BatchN=True,bias=False)
        logL_hists,i_r = gMixture_Ind_T.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=[], batch_size=BATCH_SIZE,
                                             pre_init_g=0, pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL*5/3)
        Z_train_p_OI_T = gMixture_Ind_T.get_predictions_z(Xstd_train)
        Z_test_p_OI_T = gMixture_Ind_T.get_predictions_z(Xstd_test)
        prob_Gt_OI_T = gMixture_Ind_T.get_predictions_g(T_idx_unique) 
        keras.backend.clear_session()

        gMixture_Ind_K = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Ind_K.define_model("mlp",128,1,BatchN=True,drop=0.5)
        gMixture_Ind_K.define_model_group("mlp", A_rep.shape[1], K*M_seted, 1, BatchN=False, embed=False)
        logL_hists,i_r = gMixture_Ind_K.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=A_rep, batch_size=BATCH_SIZE,
                                              pre_init_g=0,pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL*5/3)
        Z_train_p_OI_K = gMixture_Ind_K.get_predictions_z(Xstd_train)
        Z_test_p_OI_K  = gMixture_Ind_K.get_predictions_z(Xstd_test)
        prob_Gt_OI_K   = gMixture_Ind_K.get_predictions_g(A_rep) 
        keras.backend.clear_session()


    ################## MEASURE PERFORMANCE ##################################
    if "mv" in executed_models:
        evaluate = Evaluation_metrics(model_mvsoft,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        prob_Yzt = np.tile( mv_conf_probas, (T,1,1) )
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvsoft,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                         conf_true_G =confe_matrix_G, conf_pred_G = mv_conf_probas)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvsoft)
        results_softmv_train += results1
        results_softmv_test += results2

        evaluate = Evaluation_metrics(model_mvhard,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        prob_Yzt = np.tile( mv_conf_onehot, (T,1,1) )
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_mvhard,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                         conf_true_G =confe_matrix_G, conf_pred_G = mv_conf_onehot)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvhard)

        results_hardmv_train += results1
        results_hardmv_test += results2

    if "ds" in executed_models:
        evaluate = Evaluation_metrics(model_ds,'keras',Xstd_train.shape[0],plot=False)
        evaluate.set_T_weights(T_weights)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_ds,conf_pred=ds_conf,conf_true=confe_matrix_R,
                                         conf_true_G =confe_matrix_G, conf_pred_G = ds_conf.mean(axis=0))
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_ds)
        
        results_ds_train += results1
        results_ds_test += results2
    
    if "raykar" in executed_models:
        evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
        prob_Yzt = raykarMC.get_confusionM()
        Z_train_pred_Ray = Z_train_p_Ray.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                        conf_true_G =confe_matrix_G, conf_pred_G = prob_Yzt.mean(axis=0))
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)
        
        results_raykar_train += results1
        results_raykar_test += results2

    if "oursglobal" in executed_models:
        evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False) 
        aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=False,p_z=Z_train_p_OG)
        predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
        prob_Yz = gMixture_Global.calculate_Yz()
        Z_train_pred_OG = Z_train_p_OG.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                             conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        c_M = gMixture_Global.get_confusionM()
        y_o_groups = gMixture_Global.get_predictions_groups(Xstd_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_global_train +=  results1
        results_ours_global_testA.append(results2[0])
        results_ours_global_test.append(results2[1])

    if "oursindividual" in executed_models:
        evaluate = Evaluation_metrics(gMixture_Ind_T,'our1',plot=False) 
        evaluate.set_Gt(prob_Gt_OI_T)
        aux = gMixture_Ind_T.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=False,p_z=Z_train_p_OI_T,p_g=prob_Gt_OI_T)
        predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
        prob_Yz = gMixture_Ind_T.calculate_Yz(prob_Gt)
        Z_train_pred_OI = Z_train_p_OI_T.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                            conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        c_M = gMixture_Ind_T.get_confusionM()
        y_o_groups = gMixture_Ind_T.get_predictions_groups(Xstd_test,data=Z_test_p_OI_T).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OI = Z_test_p_OI_T.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_indiv_T_train +=  results1
        results_ours_indiv_T_testA.append(results2[0])
        results_ours_indiv_T_test.append(results2[1])

        evaluate = Evaluation_metrics(gMixture_Ind_K,'our1',plot=False) 
        evaluate.set_Gt(prob_Gt_OI_K)
        aux = gMixture_Ind_K.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=False,p_z=Z_train_p_OI_K,p_g=prob_Gt_OI_K)
        predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
        prob_Yz = gMixture_Ind_K.calculate_Yz(prob_Gt)
        Z_train_pred_OI = Z_train_p_OI_K.argmax(axis=-1)
        results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                            conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
        c_M = gMixture_Ind_K.get_confusionM()
        y_o_groups = gMixture_Ind_K.get_predictions_groups(Xstd_test,data=Z_test_p_OI_K).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OI = Z_test_p_OI_K.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_indiv_K_train +=  results1
        results_ours_indiv_K_testA.append(results2[0])
        results_ours_indiv_K_test.append(results2[1])

    print("All Performance Measured")
    if "mv" in executed_models:
        del model_mvsoft,model_mvhard
    if "ds" in executed_models:
        del model_ds
    if "raykar" in executed_models:
        del raykarMC
    if "oursglobal" in executed_models:
        del gMixture_Global
    if "oursindividual" in executed_models:
        del gMixture_Ind_T, gMixture_Ind_K
    del evaluate
    gc.collect()
    
#plot measures 
if "mv" in executed_models:
    get_mean_dataframes(results_softmv_train).to_csv("Fashion_softMV_train.csv",index=False)
    get_mean_dataframes(results_softmv_train, mean_std=False).to_csv("Fashion_softMV_train_std.csv",index=False)
    get_mean_dataframes(results_softmv_test).to_csv("Fashion_softMV_test.csv",index=False)
    get_mean_dataframes(results_softmv_test, mean_std=False).to_csv("Fashion_softMV_test_std.csv",index=False)

    get_mean_dataframes(results_hardmv_train).to_csv("Fashion_hardMV_train.csv",index=False)
    get_mean_dataframes(results_softmv_train, mean_std=False).to_csv("Fashion_hardMV_train_std.csv",index=False)
    get_mean_dataframes(results_hardmv_test).to_csv("Fashion_hardMV_test.csv",index=False)
    get_mean_dataframes(results_hardmv_test, mean_std=False).to_csv("Fashion_hardMV_test_std.csv",index=False)

if "ds" in executed_models:
    get_mean_dataframes(results_ds_train).to_csv("Fashion_DS_train.csv",index=False)
    get_mean_dataframes(results_ds_train, mean_std=False).to_csv("Fashion_DS_train_std.csv",index=False)
    get_mean_dataframes(results_ds_test).to_csv("Fashion_DS_test.csv",index=False)
    get_mean_dataframes(results_ds_test, mean_std=False).to_csv("Fashion_DS_test_std.csv",index=False)

if "raykar" in executed_models:
    get_mean_dataframes(results_raykar_train).to_csv("Fashion_Raykar_train.csv",index=False)
    get_mean_dataframes(results_raykar_train,  mean_std=False).to_csv("Fashion_Raykar_train_std.csv",index=False)
    get_mean_dataframes(results_raykar_test).to_csv("Fashion_Raykar_test.csv",index=False)
    get_mean_dataframes(results_raykar_test, mean_std=False).to_csv("Fashion_Raykar_test_std.csv",index=False)

if "oursglobal" in executed_models:
    get_mean_dataframes(results_ours_global_train).to_csv("Fashion_OursGlobal_train.csv",index=False)
    get_mean_dataframes(results_ours_global_train, mean_std=False).to_csv("Fashion_OursGlobal_train_std.csv",index=False)
    get_mean_dataframes(results_ours_global_test).to_csv("Fashion_OursGlobal_test.csv",index=False)
    get_mean_dataframes(results_ours_global_test, mean_std=False).to_csv("Fashion_OursGlobal_test_std.csv",index=False)
    get_mean_dataframes(results_ours_global_testA).to_csv("Fashion_OursGlobal_testAux.csv",index=False)
    get_mean_dataframes(results_ours_global_testA, mean_std=False).to_csv("Fashion_OursGlobal_testAux_std.csv",index=False)

if "oursindividual" in executed_models:
    get_mean_dataframes(results_ours_indiv_T_train).to_csv("Fashion_OursIndividualT_train.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_train, mean_std=False).to_csv("Fashion_OursIndividualT_train_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_test).to_csv("Fashion_OursIndividualT_test.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_test, mean_std=False).to_csv("Fashion_OursIndividualT_test_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_testA).to_csv("Fashion_OursIndividualT_testAux.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_testA, mean_std=False).to_csv("Fashion_OursIndividualT_testAux_std.csv",index=False)

    get_mean_dataframes(results_ours_indiv_K_train).to_csv("Fashion_OursIndividualK_train.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_train, mean_std=False).to_csv("Fashion_OursIndividualK_train_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_test).to_csv("Fashion_OursIndividualK_test.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_test, mean_std=False).to_csv("Fashion_OursIndividualK_test_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_testA).to_csv("Fashion_OursIndividualK_testAux.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_testA, mean_std=False).to_csv("Fashion_OursIndividualK_testAux_std.csv",index=False)

print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))

