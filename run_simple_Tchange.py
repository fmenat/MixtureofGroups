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
X_train = np.loadtxt(path+"/datasim_X_train.csv",delimiter=',')
Z_train = np.loadtxt(path+"/datasim_Z_train.csv",dtype='int') #groudn truth

X_test = np.loadtxt(path+"/datasim_X_test.csv",delimiter=',')
Z_test = np.loadtxt(path+"/datasim_Z_test.csv",dtype='int') #groudn truth

print("Input shape:",X_train.shape)

from sklearn.preprocessing import StandardScaler
std= StandardScaler(with_mean=True) #matrices sparse with_mean=False
std.fit(X_train)
Xstd_train = std.transform(X_train)
Xstd_test = std.transform(X_test)

from codeE.learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras, Clonable_Model
from codeE.learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple #deep learning

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

model_UB = MLP_Keras(Xstd_train.shape[1:],Z_train_onehot.shape[1],16,1,BN=False,drop=0.2) #what about bn true?
model_UB.compile(loss='categorical_crossentropy',optimizer=OPT)
hist = model_UB.fit(Xstd_train,Z_train_onehot,epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
print("Trained Ideal Model , Epochs to converge =",len(hist.epoch))
clone_UB = Clonable_Model(model_UB)

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

from codeE.generate_data import SinteticData

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
results_ours_indiv_T_train = []
results_ours_indiv_T_trainA = []
results_ours_indiv_T_test = []
results_ours_indiv_T_testA = []
results_ours_indiv_K_train = []
results_ours_indiv_K_trainA = []
results_ours_indiv_K_test = []
results_ours_indiv_K_testA = []
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
    aux_ours_indiv_T_train = []
    #aux_ours_indiv_T_trainA = []
    aux_ours_indiv_T_test = []
    aux_ours_indiv_T_testA = []
    aux_ours_indiv_K_train = []
    #aux_ours_indiv_K_trainA = []
    aux_ours_indiv_K_test = []
    aux_ours_indiv_K_testA = []
    
    for _ in range(30):
        #aux_acc = 0
        #while aux_acc < 0.71 and aux_acc > 0.74:
        GenerateData = SinteticData(state=state_sce) #por la semilla quedan similares..
        #CONFUSION MATRIX CHOOSE
        GenerateData.set_probas(asfile=True,file_matrix=path+'/matrix_datasim_normal.csv',file_groups =path+'/groups_datasim_normal.csv')
        real_conf_matrix = GenerateData.conf_matrix.copy()

        print("New Synthetic data is being generated...",flush=True,end='')
        y_obs, groups_annot = GenerateData.sintetic_annotate_data(Z_train,Tmax,T_data,deterministic=False,hard=True)
        print("Done! ")
        if len(groups_annot.shape) ==1 or groups_annot.shape[1] ==  1: 
            groups_annot = keras.utils.to_categorical(groups_annot)  #only if it is hard clustering
        confe_matrix_R = np.tensordot(groups_annot,real_conf_matrix, axes=[[1],[0]])

            #aux_acc = np.mean(GenerateData.yo_label == Z_train)

        T_weights = np.sum(y_obs != -1,axis=0)
        print("Mean annotations by t= ",T_weights.mean())
        N,T = y_obs.shape
        K = np.max(y_obs)+1 # asumiendo que estan ordenadas
        print("Shape (data,annotators): ",(N,T))
        print("Classes: ",K)

        ############### MV/DS and calculate representations##############################
        if "mv" in executed_models or "ds" in executed_models:
            label_I = LabelInference(y_obs,TOL,type_inf = 'all')  #Infer Labels

        #Deterministic
        if "mv" in executed_models:
            mv_probas, mv_conf_probas = label_I.mv_labels('probas')
            mv_onehot, mv_conf_onehot = label_I.mv_labels('onehot')
            print("ACC MV on train:",np.mean(mv_onehot.argmax(axis=1)==Z_train))
            confe_matrix_G = get_Global_confusionM(Z_train,label_I.y_obs_repeat)

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
                aux = [entropy(example)/np.log(r_obs.shape[1]) for example in mv_probas]
                print("Normalized entropy (0-1) of repeats annotations:",np.mean(aux))
            elif "raykar" in executed_models:
                r_obs = set_representation(y_obs_categorical,"repeat")
                confe_matrix_G = get_Global_confusionM(Z_train,r_obs)
            else:
                r_obs = set_representation(y_obs,"repeat")
                confe_matrix_G = get_Global_confusionM(Z_train,r_obs)
            print("vector of repeats:\n",r_obs)
            print("shape:",r_obs.shape)

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

    #for _ in range(20):
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
            raykarMC.define_model('mlp',16,1,BatchN=False,drop=0.2)
            logL_hists,i_r = raykarMC.multiples_run(20,Xstd_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_Ray = raykarMC.get_predictions(Xstd_train)
            Z_test_pred_Ray = raykarMC.get_predictions(Xstd_test).argmax(axis=-1)
            keras.backend.clear_session()

        if "oursglobal" in executed_models:
            gMixture_Global = GroupMixtureGlo(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
            gMixture_Global.define_model("mlp",16,1,BatchN=False,drop=0.2)
            gMixture_Global.lambda_random = False #with lambda random --necessary
            logL_hists,i = gMixture_Global.multiples_run(20,Xstd_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_OG = gMixture_Global.get_predictions(Xstd_train)
            Z_test_p_OG = gMixture_Global.get_predictions(Xstd_test)
            keras.backend.clear_session()

        if "oursindividual" in executed_models:
            gMixture_Ind_T = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
            gMixture_Ind_T.define_model("mlp",16,1,BatchN=False,drop=0.2) 
            gMixture_Ind_T.define_model_group("perceptron",T, M_seted, embed=True, embed_M=A, BatchN=True,bias=False)
            logL_hists,i_r = gMixture_Ind_T.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=[], batch_size=BATCH_SIZE,
                                                  pre_init_g=0, pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_OI_T = gMixture_Ind_T.get_predictions_z(Xstd_train)
            Z_test_p_OI_T = gMixture_Ind_T.get_predictions_z(Xstd_test)
            prob_Gt_OI_T = gMixture_Ind_T.get_predictions_g(T_idx_unique) 
            keras.backend.clear_session()
            
            gMixture_Ind_K = GroupMixtureInd(Xstd_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
            gMixture_Ind_K.define_model("mlp",16,1,BatchN=False,drop=0.2) 
            gMixture_Ind_K.define_model_group("mlp", A_rep.shape[1], K*M_seted, 1, embed=True, embed_M=A_rep)
            logL_hists,i_r = gMixture_Ind_K.multiples_run(20,Xstd_train,Y_ann_train, T_idx, A=A_rep, batch_size=BATCH_SIZE,
                                                  pre_init_g=0,pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
            Z_train_p_OI_K = gMixture_Ind_K.get_predictions_z(Xstd_train)
            Z_test_p_OI_K  = gMixture_Ind_K.get_predictions_z(Xstd_test)
            prob_Gt_OI_K   = gMixture_Ind_K.get_predictions_g(T_idx_unique) 
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

        if "ds" in executed_models:
            evaluate = Evaluation_metrics(model_ds,'keras',Xstd_train.shape[0],plot=False)
            evaluate.set_T_weights(T_weights)
            results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_ds,conf_pred=ds_conf,conf_true=confe_matrix_R,
                                         conf_true_G =confe_matrix_G, conf_pred_G = ds_conf.mean(axis=0))
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_ds)

            aux_ds_train += results1
            aux_ds_test += results2

        if "raykar" in executed_models:
            evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
            prob_Yzt = raykarMC.get_confusionM()
            #prob_Yxt = raykarMC.get_predictions_annot(Xstd_train,data=Z_train_p_Ray)
            Z_train_pred_Ray = Z_train_p_Ray.argmax(axis=-1)
            results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_Ray,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                         y_o=y_obs,conf_true_G =confe_matrix_G, conf_pred_G = prob_Yzt.mean(axis=0))
            #results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)

            aux_raykar_train += results1
            #aux_raykar_trainA += results1_aux
            aux_raykar_test += results2

        if "oursglobal" in executed_models:
            evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False) 
            aux = gMixture_Global.calculate_extra_components(Xstd_train,y_obs,T=T,calculate_pred_annotator=False,p_z=Z_train_p_OG)
            predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
            prob_Yz = gMixture_Global.calculate_Yz()
            Z_train_pred_OG = Z_train_p_OG.argmax(axis=-1)
            results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OG,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                  y_o=y_obs,
                                                 conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            #results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
            c_M = gMixture_Global.get_confusionM()
            y_o_groups = gMixture_Global.get_predictions_groups(Xstd_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
            Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

            aux_ours_global_train +=  results1
            #aux_ours_global_trainA += results1_aux
            aux_ours_global_testA.append(results2[0])
            aux_ours_global_test.append(results2[1])

        if "oursindividual" in executed_models:
            evaluate = Evaluation_metrics(gMixture_Ind_T,'our1',plot=False) 
            evaluate.set_Gt(prob_Gt_OI_T)
            aux = gMixture_Ind_T.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=False,p_z=Z_train_p_OI_T,p_g=prob_Gt_OI_T)
            predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
            prob_Yz = gMixture_Ind_T.calculate_Yz(prob_Gt)
            Z_train_pred_OI = Z_train_p_OI_T.argmax(axis=-1)
            results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                 y_o=y_obs,
                                                conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            #results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=_)
            c_M = gMixture_Ind_T.get_confusionM()
            y_o_groups = gMixture_Ind_T.get_predictions_groups(Xstd_test,data=Z_test_p_OI_T).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
            Z_test_pred_OI = Z_test_p_OI_T.argmax(axis=-1)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

            aux_ours_indiv_T_train +=  results1
            #aux_ours_indiv_T_trainA += results1_aux
            aux_ours_indiv_T_testA.append(results2[0])
            aux_ours_indiv_T_test.append(results2[1])

            evaluate = Evaluation_metrics(gMixture_Ind_K,'our1',plot=False) 
            evaluate.set_Gt(prob_Gt_OI_K)
            aux = gMixture_Ind_K.calculate_extra_components(Xstd_train, A,calculate_pred_annotator=False,p_z=Z_train_p_OI_K,p_g=prob_Gt_OI_K)
            predictions_m,prob_Gt,prob_Yzt,_ =  aux #to evaluate...
            prob_Yz = gMixture_Ind_K.calculate_Yz(prob_Gt)
            Z_train_pred_OI = Z_train_p_OI_K.argmax(axis=-1)
            results1 = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred_OI,conf_pred=prob_Yzt,conf_true=confe_matrix_R,
                                                 y_o=y_obs,
                                                conf_true_G =confe_matrix_G, conf_pred_G = prob_Yz)
            #results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=_)
            c_M = gMixture_Ind_K.get_confusionM()
            y_o_groups = gMixture_Ind_K.get_predictions_groups(Xstd_test,data=Z_test_p_OI_K).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
            Z_test_pred_OI = Z_test_p_OI_K.argmax(axis=-1)
            results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

            aux_ours_indiv_K_train +=  results1
            #aux_ours_indiv_K_trainA += results1_aux
            aux_ours_indiv_K_testA.append(results2[0])
            aux_ours_indiv_K_test.append(results2[1])

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
        results_softmv_train.append([get_mean_dataframes(aux_softmv_train), get_mean_dataframes(aux_softmv_train,mean_std=False)])
        results_softmv_test.append([get_mean_dataframes(aux_softmv_test), get_mean_dataframes(aux_softmv_test,mean_std=False)])
        results_hardmv_train.append([get_mean_dataframes(aux_hardmv_train), get_mean_dataframes(aux_hardmv_train,mean_std=False)])
        results_hardmv_test.append([get_mean_dataframes(aux_hardmv_test), get_mean_dataframes(aux_hardmv_test,mean_std=False)])
    if "ds" in executed_models:
        results_ds_train.append([get_mean_dataframes(aux_ds_train), get_mean_dataframes(aux_ds_train,mean_std=False)])
        results_ds_test.append([get_mean_dataframes(aux_ds_test), get_mean_dataframes(aux_ds_test,mean_std=False)])
    if "raykar" in executed_models:
        results_raykar_train.append([get_mean_dataframes(aux_raykar_train), get_mean_dataframes(aux_raykar_train,mean_std=False)])
        #results_raykar_trainA.append([get_mean_dataframes(aux_raykar_trainA), get_mean_dataframes(aux_raykar_trainA,mean_std=False)])
        results_raykar_test.append([get_mean_dataframes(aux_raykar_test), get_mean_dataframes(aux_raykar_test,mean_std=False)])
    if "oursglobal" in executed_models:
        results_ours_global_train.append([get_mean_dataframes(aux_ours_global_train), get_mean_dataframes(aux_ours_global_train,mean_std=False)])
        #results_ours_global_trainA.append([get_mean_dataframes(aux_ours_global_trainA), get_mean_dataframes(aux_ours_global_trainA,mean_std=False)])
        results_ours_global_test.append([get_mean_dataframes(aux_ours_global_test), get_mean_dataframes(aux_ours_global_test,mean_std=False)])
        results_ours_global_testA.append([get_mean_dataframes(aux_ours_global_testA), get_mean_dataframes(aux_ours_global_testA,mean_std=False)])
    if "oursindividual" in executed_models:
        results_ours_indiv_T_train.append([get_mean_dataframes(aux_ours_indiv_T_train), get_mean_dataframes(aux_ours_indiv_T_train,mean_std=False)])
        #results_ours_indiv_T_trainA.append([get_mean_dataframes(aux_ours_indiv_T_trainA), get_mean_dataframes(aux_ours_indiv_T_trainA,mean_std=False)])
        results_ours_indiv_T_test.append([get_mean_dataframes(aux_ours_indiv_T_test), get_mean_dataframes(aux_ours_indiv_T_test,mean_std=False)])
        results_ours_indiv_T_testA.append([get_mean_dataframes(aux_ours_indiv_T_testA), get_mean_dataframes(aux_ours_indiv_T_testA,mean_std=False)])
        results_ours_indiv_K_train.append([get_mean_dataframes(aux_ours_indiv_K_train), get_mean_dataframes(aux_ours_indiv_K_train,mean_std=False)])
        #results_ours_indiv_K_trainA.append([get_mean_dataframes(aux_ours_indiv_K_trainA), get_mean_dataframes(aux_ours_indiv_K_trainA,mean_std=False)])
        results_ours_indiv_K_test.append([get_mean_dataframes(aux_ours_indiv_K_test), get_mean_dataframes(aux_ours_indiv_K_test,mean_std=False)])
        results_ours_indiv_K_testA.append([get_mean_dataframes(aux_ours_indiv_K_testA), get_mean_dataframes(aux_ours_indiv_K_testA,mean_std=False)])
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
    #with open('synthetic_Raykar_trainAnn.pickle', 'wb') as handle:
    #    pickle.dump(results_raykar_trainA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_Raykar_test.pickle', 'wb') as handle:
        pickle.dump(results_raykar_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

if "oursglobal" in executed_models: 
    with open('synthetic_OursGlobal_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_global_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('synthetic_OursGlobal_trainAnn.pickle', 'wb') as handle:
    #    pickle.dump(results_ours_global_trainA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursGlobal_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_global_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursGlobal_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_global_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)

if "oursindividual" in executed_models: 
    with open('synthetic_OursIndividualT_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_T_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('synthetic_OursIndividualT_trainAnn.pickle', 'wb') as handle:
    #    pickle.dump(results_ours_indiv_T_trainA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividualT_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_T_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividualT_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_T_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividualK_train.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_K_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('synthetic_OursIndividualK_trainAnn.pickle', 'wb') as handle:
    #    pickle.dump(results_ours_indiv_K_trainA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividualK_test.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_K_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('synthetic_OursIndividualK_testAux.pickle', 'wb') as handle:
        pickle.dump(results_ours_indiv_K_testA, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))

