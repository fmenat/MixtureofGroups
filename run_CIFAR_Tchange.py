from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/synthetic..)")
op.add_option("-e", "--executed", type="string", default='', help="executed models separated by /.. ex (hardmv/ds/raykar)")

(opts, args) = op.parse_args()
path = opts.path
M_seted = opts.Ngroups 
state_sce = None
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
from keras.models import clone_model

DTYPE_OP = 'float32'
keras.backend.set_floatx(DTYPE_OP)

if DTYPE_OP == 'float64':
    keras.backend.set_epsilon(np.finfo(np.float64).eps)
elif DTYPE_OP == 'float32':
    keras.backend.set_epsilon(np.finfo(np.float32).eps)
    
### Load Data
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
from code.MixtureofGroups import GroupMixtureGlo, project_and_cluster,clusterize_annotators, GroupMixtureInd

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

#ANNOTATOR DENSITY CHOOSE
to_check = [100,500,1500,3500,6000,10000]
T_data = 3 #this could change


results_softmv_train = []
results_softmv_test = []
results_hardmv_train = []
results_hardmv_test = []
results_ds_train = []
results_ds_test = []
results_raykar_train = []
#results_raykar_trainA = []
results_raykar_test = []
results_ours_global_train = []
#results_ours_global_trainA = []
results_ours_global_test = []
results_ours_global_testA = []
results_ours_indiv_train = []
#results_ours_indiv_trainA = []
results_ours_indiv_test = []
results_ours_indiv_testA = []
results_ours_indiv2_train = []
results_ours_indiv2_test = []
results_ours_indiv2_testA = []
results_ours_indiv3_train = []
results_ours_indiv3_test = []
results_ours_indiv3_testA = []

for Tmax in to_check:
    aux_softmv_train = []
    aux_softmv_test = []
    aux_hardmv_train = []
    aux_hardmv_test = []
    aux_ds_train = []
    aux_ds_test = []
    aux_raykar_train = []
    #aux_raykar_trainA = []
    aux_raykar_test = []
    aux_ours_global_train = []
    #aux_ours_global_trainA = []
    aux_ours_global_test = []
    aux_ours_global_testA = []
    aux_ours_indiv_train = []
    #aux_ours_indiv_trainA = []
    aux_ours_indiv_test = []
    aux_ours_indiv_testA = []
    aux_ours_indiv2_train = []
    aux_ours_indiv2_test = []
    aux_ours_indiv2_testA = []
    aux_ours_indiv3_train = []
    aux_ours_indiv3_test = []
    aux_ours_indiv3_testA = []

    GenerateData = SinteticData(state=state_sce) #por la semilla quedan similares..
    #CONFUSION MATRIX CHOOSE
    GenerateData.set_probas(asfile=True,file_matrix=path+'/synthetic/CIFAR/matrix_CIFAR_Td3_v2.csv',file_groups =path+'/synthetic/CIFAR/groups_CIFAR_Td3_v2.csv') #mejor acc mv
    real_conf_matrix = GenerateData.conf_matrix.copy()

    print("New Synthetic data is being generated...",flush=True,end='')
    y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False)
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
    start_time = time.time()
    if Tmax <4000 and ("mv" in executed_models or "ds" in executed_models): #other wise cannot be done
        label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels
    if Tmax >= 4000 and "mv" in executed_models:
        label_I = LabelInference(y_obs,TOL,type_inf = 'mv')  #Infer Labels
    print("Representation for Our/MV/D&S in %f mins"%((time.time()-start_time)/60.) )

    if "mv" in executed_models:
        mv_probas, mv_conf_probas = label_I.mv_labels('probas')
        mv_onehot, mv_conf_onehot = label_I.mv_labels('onehot')
        print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))
        confe_matrix_G = get_Global_confusionM(Z_train,label_I.y_obs_repeat)
    
    #Deterministic
    if Tmax <4000 and "ds" in executed_models: #other wise cannot be done
        ds_labels, ds_conf = label_I.DS_labels()
        print("ACC D&S on train:",np.mean(ds_labels.argmax(axis=1)==Z_train))

    if Tmax <4000 and "raykar" in executed_models:
        #get representation needed for Raykar
        start_time = time.time()
        y_obs_categorical = set_representation(y_obs,'onehot') 
        print("shape:",y_obs_categorical.shape)
        print("Representation for Raykar in %f mins"%((time.time()-start_time)/60.) )

    if "oursglobal" in executed_models
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

    for _ in range(5): #repetitions
        ############# EXECUTE ALGORITHMS #############################
        if "mv" in executed_models:
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
    
        if Tmax <4000 and "ds" in executed_models: #other wise cannot be done
            model_ds = clone_model(model_UB) 
            model_ds.compile(loss='categorical_crossentropy',optimizer=OPT)
            hist=model_ds.fit(Xstd_train, ds_labels, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
            print("Trained model over D&S, Epochs to converge =",len(hist.epoch))
            Z_train_pred_ds = model_ds.predict_classes(Xstd_train)
            Z_test_pred_ds = model_ds.predict_classes(Xstd_test)
            keras.backend.clear_session()

        if Tmax <4000 and "raykar" in executed_models: #other wise cannot be done
            raykarMC = RaykarMC(Xstd_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
            raykarMC.define_model('default cnn')
            logL_hists,i_r = raykarMC.multiples_run(15,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_Ray = raykarMC.get_predictions(Xstd_train)
            Z_test_pred_Ray = raykarMC.get_predictions(Xstd_test).argmax(axis=-1)
            keras.backend.clear_session()

        if "oursglobal" in executed_models:
            gMixture_Global = GroupMixtureGlo(Xstd_train.shape[1:],Kl=r_obs.shape[1],M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
            gMixture_Global.define_model("default cnn")
            logL_hists,i = gMixture_Global.multiples_run(15,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_OG = gMixture_Global.get_predictions(Xstd_train)
            Z_test_p_OG = gMixture_Global.get_predictions(Xstd_test)
            keras.backend.clear_session()

        if "oursindividual" in executed_models:
            gMixture_Ind2 = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
            gMixture_Ind2.define_model("default cnn")
            logL_hists,i_r = gMixture_Ind2.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=[], batch_size=BATCH_SIZE,
                                                  pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_OI2 = gMixture_Ind2.get_predictions_z(Xstd_train)
            Z_test_p_OI2 = gMixture_Ind2.get_predictions_z(Xstd_test)
            prob_Gt_OI2 = gMixture_Ind2.get_predictions_g(T_idx_unique) 
            keras.backend.clear_session()
            
            gMixture_Ind3 = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
            gMixture_Ind3.define_model("default cnn")
            gMixture_Ind3.define_model_group("mlp", A_rep.shape[1], K*M_seted, 1, BatchN=False, embed=False)
            logL_hists,i_r = gMixture_Ind3.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=A_rep, batch_size=BATCH_SIZE,
                                                  pre_init_g=15,pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_OI3 = gMixture_Ind3.get_predictions_z(Xstd_train)
            Z_test_p_OI3  = gMixture_Ind3.get_predictions_z(Xstd_test)
            prob_Gt_OI3   = gMixture_Ind3.get_predictions_g(A_rep) 
            keras.backend.clear_session()

        ################## MEASURE PERFORMANCE ##################################
        if "mv" in executed_models:
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

        if Tmax <4000 and "ds" in executed_models: #other wise cannot be done
            evaluate = Evaluation_metrics(model_ds,'keras',Xstd_train.shape[0],plot=False)
            evaluate.set_T_weights(T_weights)
            results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_ds,conf_pred=ds_conf,conf_true=confe_matrix_R,
                                     conf_true_G =confe_matrix_G, conf_pred_G = ds_conf.mean(axis=0))
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_ds)

            aux_ds_train += results1
            aux_ds_test += results2

        if Tmax <4000 and "raykar" in executed_models: #other wise cannot be done
            evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
            prob_Yzt = raykarMC.get_confusionM()
            Z_train_pred_Ray = Z_train_p_Ray.argmax(axis=-1)
            if Tmax < 3000:
                prob_Yxt = raykarMC.get_predictions_annot(Xstd_train,data=Z_train_p_Ray)
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                     y_o=y_obs,yo_pred=prob_Yxt,conf_true_G =confe_matrix_G, conf_pred_G = prob_Yzt.mean(axis=0))
            else:
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                     conf_true_G =confe_matrix_G, conf_pred_G = prob_Yzt.mean(axis=0))
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)

            aux_raykar_train += results1
            aux_raykar_test += results2
        
        if "oursglobal" in executed_models:
            evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False)  #no explota
            prob_Yz = gMixture_Global.calculate_Yz()
            Z_train_pred_OG = Z_train_p_OG.argmax(axis=-1)
            if Tmax < 3000:
                aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=True,p_z=Z_train_p_OG)
                predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                  y_o=y_obs,yo_pred=prob_Yxt,
                                                 conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)        
            else: #pred annotator memory error
                aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=False,p_z=Z_train_p_OG)
                predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                 conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)      
            c_M = gMixture_Global.get_confusionM()
            y_o_groups = gMixture_Global.get_predictions_groups(Xstd_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
            Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

            aux_ours_global_train +=  results1
            aux_ours_global_testA.append(results2[0])
            aux_ours_global_test.append(results2[1])

        if "oursindividual" in executed_models:
            evaluate = Evaluation_metrics(gMixture_Ind2,'our1',plot=False) 
            evaluate.set_Gt(prob_Gt_OI2)
            Z_train_pred_OI = Z_train_p_OI2.argmax(axis=-1)
            prob_Yz = gMixture_Ind2.calculate_Yz(prob_Gt_OI2)
            if Tmax < 3000:
                aux = gMixture_Ind2.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=True,p_z=Z_train_p_OI2,p_g=prob_Gt_OI2)
                predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                     y_o=y_obs,yo_pred=prob_Yxt,
                                                    conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            else:
                aux = gMixture_Ind2.calculate_extra_components(Xstd_train,A,calculate_pred_annotator=False,p_z=Z_train_p_OI2,p_g=prob_Gt_OI2)
                predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                 conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            c_M = gMixture_Ind2.get_confusionM()
            y_o_groups = gMixture_Ind2.get_predictions_groups(Xstd_test,data=Z_test_p_OI2).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
            Z_test_pred_OI = Z_test_p_OI2.argmax(axis=-1)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

            aux_ours_indiv2_train +=  results1
            aux_ours_indiv2_testA.append(results2[0])
            aux_ours_indiv2_test.append(results2[1])

            evaluate = Evaluation_metrics(gMixture_Ind3,'our1',plot=False) 
            evaluate.set_Gt(prob_Gt_OI3)
            Z_train_pred_OI = Z_train_p_OI3.argmax(axis=-1)
            prob_Yz = gMixture_Ind3.calculate_Yz(prob_Gt_OI3)
            if Tmax < 3000:
                aux = gMixture_Ind3.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=True,p_z=Z_train_p_OI3,p_g=prob_Gt_OI3)
                predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                     y_o=y_obs,yo_pred=prob_Yxt,
                                                    conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            else:
                aux = gMixture_Ind3.calculate_extra_components(Xstd_train,A,calculate_pred_annotator=False,p_z=Z_train_p_OI3,p_g=prob_Gt_OI3)
                predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
                results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                 conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            c_M = gMixture_Ind3.get_confusionM()
            y_o_groups = gMixture_Ind3.get_predictions_groups(Xstd_test,data=Z_test_p_OI3).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
            Z_test_pred_OI = Z_test_p_OI3.argmax(axis=-1)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

            aux_ours_indiv3_train +=  results1
            aux_ours_indiv3_testA.append(results2[0])
            aux_ours_indiv3_test.append(results2[1])


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
            del gMixture_Ind2, gMixture_Ind3
        del evaluate
        gc.collect()

    #plot measures    
    if "mv" in executed_models:
        results_softmv_train.append(get_mean_dataframes(aux_softmv_train))
        results_softmv_test.append(get_mean_dataframes(aux_softmv_test))
        results_hardmv_train.append(get_mean_dataframes(aux_hardmv_train))
        results_hardmv_test.append(get_mean_dataframes(aux_hardmv_test))
    if "ds" in executed_models:
        if Tmax <4000: #other wise cannot be done 
            results_ds_train.append(get_mean_dataframes(aux_ds_train))
            results_ds_test.append(get_mean_dataframes(aux_ds_test))
        else:
            results_ds_train.append(np.nan)
            results_ds_test.append(np.nan)
    if "raykar" in executed_models:
        if Tmax <4000: #other wise cannot be done 
            results_raykar_train.append(get_mean_dataframes(aux_raykar_train))
            results_raykar_test.append(get_mean_dataframes(aux_raykar_test))
        else:
            results_raykar_train.append(np.nan)
            results_raykar_test.append(np.nan)
    if "oursglobal" in executed_models:
        results_ours_global_train.append(get_mean_dataframes(aux_ours_global_train))
        results_ours_global_test.append(get_mean_dataframes(aux_ours_global_test))
        results_ours_global_testA.append(get_mean_dataframes(aux_ours_global_testA))
    if "oursindividual" in executed_models:
        #results_ours_indiv_train.append(get_mean_dataframes(aux_ours_indiv_train))
        #results_ours_indiv_test.append(get_mean_dataframes(aux_ours_indiv_test))
        #results_ours_indiv_testA.append(get_mean_dataframes(aux_ours_indiv_testA))
        results_ours_indiv2_train.append(get_mean_dataframes(aux_ours_indiv2_train))
        results_ours_indiv2_test.append(get_mean_dataframes(aux_ours_indiv2_test))
        results_ours_indiv2_testA.append(get_mean_dataframes(aux_ours_indiv2_testA))
        results_ours_indiv3_train.append(get_mean_dataframes(aux_ours_indiv3_train))
        results_ours_indiv3_test.append(get_mean_dataframes(aux_ours_indiv3_test))
        results_ours_indiv3_testA.append(get_mean_dataframes(aux_ours_indiv3_testA))
    gc.collect()

import pickle
if "mv" in executed_models:
    with open('synthetic_softMV_train.pickle', 'wb') as handle:
        pickle.dump(results_softmv_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_softMV_test.pickle', 'wb') as handle:
        pickle.dump(results_softmv_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_hardMV_train.pickle', 'wb') as handle:
        pickle.dump(results_hardmv_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_hardMV_test.pickle', 'wb') as handle:
        pickle.dump(results_hardmv_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

if "ds" in executed_models:
    with open('synthetic_DS_train.pickle', 'wb') as handle:
        pickle.dump(results_ds_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_DS_test.pickle', 'wb') as handle:
        pickle.dump(results_ds_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

if "raykar" in executed_models:
    with open('synthetic_Raykar_train.pickle', 'wb') as handle:
        pickle.dump(results_raykar_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_Raykar_test.pickle', 'wb') as handle:
        pickle.dump(results_raykar_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

if "oursglobal" in executed_models:
    with open('synthetic_OursGlobal_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_global_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursGlobal_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_global_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursGlobal_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_global_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)

if "oursindividual" in executed_models:
    """
    with open('synthetic_OursIndividual_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    with open('synthetic_OursIndividual2_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv2_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual2_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv2_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual2_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv2_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual3_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv3_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual3_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv3_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividual3_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv3_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))

