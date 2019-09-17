from optparse import OptionParser

op = OptionParser()
op.add_option("-M", "--Ngroups", type=int, default=3, help="number of groups in propose formulation")
op.add_option("-p", "--path", type="string", default='data/', help="path for data (path/rotten.....)")
op.add_option("-g", "--pathGlove", type="string", default='data/', help="path to Glove embeddings")
op.add_option("-v", "--version", type="string", default='', help="version of annotations default: '' or 'detail'")
op.add_option("-e", "--executed", type="string", default='', help="executed models separated by /.. ex (hardmv/ds/raykar)")

(opts, args) = op.parse_args()
folder = opts.path
GLOVE_folder = opts.pathGlove
M_seted = opts.Ngroups 
version = opts.version 
if version== None or len(version) == 0:
    version = ''
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
    
import re
FLAGS = re.MULTILINE | re.DOTALL
def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"
def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", "<hashtag>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)
    return text.lower()

def read_texts(filename):
    f = open(filename)
    data = [line.strip() for line in f]
    f.close()
    return data

### Load Data
texts_train = [tokenize(text) for text in read_texts(folder+"texts_train_.txt")]
Z_train = np.loadtxt(folder+"sent_train_%s.txt"%version, dtype='int')

### ground truth over many annotation data
mask_gt = Z_train != -1

texts_test = list(np.asarray(texts_train)[mask_gt])
texts_train = list(np.asarray(texts_train)[~mask_gt])
Z_test = Z_train[mask_gt]
Z_train = Z_train[~mask_gt] #all -1

print("Num. train texts: %d" % len(texts_train))
print("Num. test texts:  %d" % len(texts_test))
print("Label shape:",Z_train.shape)

##process data
from keras import preprocessing
MAX_NB_WORDS = 15000
tokenizer = preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train+texts_test)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)
MAX_NB_WORDS = len(tokenizer.word_index)
print('Found %s unique tokens.' % len(tokenizer.word_index))

max_L = 50 
print("Used max: ",max_L)
X_train = preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_L,dtype='int32', value=0,padding='pre')
X_test = preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_L,dtype='int32', value=0,padding='pre')
print('Shape of train tensor:', X_train.shape)

EMBEDDING_DIM = 100
GLOVE_FILE = GLOVE_folder+"/glove.twitter.27B.%dd.txt"%(EMBEDDING_DIM)
embeddings_index = {}
with open(GLOVE_FILE) as file:
    for line in file:
        values = line.split()
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[values[0]] = coefs
print('Preparing embedding matrix.')
sorted_x = sorted(tokenizer.word_counts.items(), key=lambda kv: kv[1], reverse=True)
vocab = {value[0]:tokenizer.word_index[value[0]] for i, value in enumerate(sorted_x) if i < MAX_NB_WORDS}
embedding_matrix = np.zeros((len(vocab)+1, EMBEDDING_DIM))
v=0
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector # words not found in embedding index will be all-zeros.
        v+=1
del embeddings_index, sorted_x, tokenizer
gc.collect()
print("Words found on glove: ",v)


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
file_annot = "/answers_"+version+".txt"
y_obs = np.loadtxt(folder+file_annot,dtype='int16') #not annotation symbol ==-1
y_obs = y_obs[~mask_gt] #se pierdne hartas anotadoras :c
T_weights = np.sum(y_obs != -1,axis=0) #distribucion de anotaciones

print("Remove %d annotators that do not annotate on this set "%(np.sum(T_weights==0)))
y_obs = y_obs[:,T_weights!=0]

N,T = y_obs.shape
K = np.max(y_obs)+1 # asumiendo que estan ordenadas
print("Shape (data,annotators): ",(N,T))
print("Classes: ",K)

model_UB = default_RNN_text(max_L, Kl, embed_M=embedding_matrix)
clone_UB = Clonable_Model(model_UB)

results_softmv_train = []
results_softmv_test = []
results_hardmv_train = []
results_hardmv_test = []
results_ds_train = []
results_ds_test = []
#results_raykar_train = []
results_raykar_trainA = []
results_raykar_test = []
#results_ours_global_train = []
results_ours_global_trainA = []
results_ours_global_test = []
results_ours_global_testA = []
#results_ours_indiv_T_train = []
results_ours_indiv_T_trainA = []
results_ours_indiv_T_test = []
results_ours_indiv_T_testA = []
#results_ours_indiv_K_train = []
results_ours_indiv_K_trainA = []
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

if "ds" in executed_models:
    ds_labels, ds_conf = label_I.DS_labels()

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
    else:
        r_obs = set_representation(y_obs,"repeat")
    print("vector of repeats:\n",r_obs)
    print("shape:",r_obs.shape)

    #analysis
    if "mv" in executed_models:
        aux = [entropy(example)/np.log(r_obs.shape[1]) for example in mv_probas]
        print("Normalized entropy (0-1) of repeats annotations:",np.mean(aux))

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
        hist = model_mvsoft.fit(X_train, mv_probas, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over soft-MV, Epochs to converge =",len(hist.epoch))
        Z_train_pred_mvsoft = model_mvsoft.predict_classes(X_train)
        Z_test_pred_mvsoft = model_mvsoft.predict_classes(X_test)
        keras.backend.clear_session()

        model_mvhard = clone_UB.get_model() 
        model_mvhard.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_mvhard.fit(X_train, mv_onehot, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over hard-MV, Epochs to converge =",len(hist.epoch))
        Z_train_pred_mvhard = model_mvhard.predict_classes(X_train)
        Z_test_pred_mvhard = model_mvhard.predict_classes(X_test)
        keras.backend.clear_session()

    if "ds" in executed_models:
        model_ds = clone_UB.get_model() 
        model_ds.compile(loss='categorical_crossentropy',optimizer=OPT)
        hist=model_ds.fit(X_train, ds_labels, epochs=EPOCHS_BASE,batch_size=BATCH_SIZE,verbose=0,callbacks=[ourCallback])
        print("Trained model over D&S, Epochs to converge =",len(hist.epoch))
        Z_train_pred_ds = model_ds.predict_classes(X_train)
        Z_test_pred_ds = model_ds.predict_classes(X_test)
        keras.backend.clear_session()

    if "raykar" in executed_models:
        raykarMC = RaykarMC(X_train.shape[1:],y_obs_categorical.shape[-1],T,epochs=1,optimizer=OPT,DTYPE_OP=DTYPE_OP)
        raykarMC.define_model("default rnn text", embed=embedding_matrix)
        logL_hists,i_r = raykarMC.multiples_run(20,X_train,y_obs_categorical,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        #Z_train_p_Ray = raykarMC.get_predictions(X_train)
        Z_test_pred_Ray = raykarMC.get_predictions(X_test).argmax(axis=-1)
        keras.backend.clear_session()
    
    if "oursglobal" in executed_models:
        gMixture_Global = GroupMixtureGlo(X_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Global.define_model("default rnn text", embed=embedding_matrix)
        logL_hists,i = gMixture_Global.multiples_run(20,X_train,r_obs,batch_size=BATCH_SIZE,max_iter=EPOCHS_BASE,tolerance=TOL)
        #Z_train_p_OG = gMixture_Global.get_predictions(X_train)
        Z_test_p_OG = gMixture_Global.get_predictions(X_test)
        keras.backend.clear_session()

    if "oursindividual" in executed_models:
        gMixture_Ind_T = GroupMixtureInd(X_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Ind_T.define_model("default rnn text", embed=embedding_matrix)
        gMixture_Ind_T.define_model_group("perceptron",T, M_seted, embed=True, embed_M=A, BatchN=True,bias=False)
        logL_hists,i_r = gMixture_Ind_T.multiples_run(20,X_train,Y_ann_train, T_idx, A=[], batch_size=BATCH_SIZE,
                                             pre_init_g=0, pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
        #Z_train_p_OI_T = gMixture_Ind_T.get_predictions_z(X_train)
        Z_test_p_OI_T = gMixture_Ind_T.get_predictions_z(X_test)
        prob_Gt_OI_T = gMixture_Ind_T.get_predictions_g(T_idx_unique) 
        keras.backend.clear_session()

        gMixture_Ind_K = GroupMixtureInd(X_train.shape[1:],Kl=K,M=M_seted,epochs=1,optimizer=OPT,dtype_op=DTYPE_OP) 
        gMixture_Ind_K.define_model("default rnn text", embed=embedding_matrix)
        gMixture_Ind_K.define_model_group("mlp", A_rep.shape[1], K*M_seted, 1, BatchN=False, embed=False)
        logL_hists,i_r = gMixture_Ind_K.multiples_run(20,X_train,Y_ann_train, T_idx, A=A_rep, batch_size=BATCH_SIZE,
                                              pre_init_g=0,pre_init_z=3, max_iter=EPOCHS_BASE,tolerance=TOL)
        #Z_train_p_OI_K = gMixture_Ind_K.get_predictions_z(X_train)
        Z_test_p_OI_K  = gMixture_Ind_K.get_predictions_z(X_test)
        prob_Gt_OI_K   = gMixture_Ind_K.get_predictions_g(A_rep) 
        keras.backend.clear_session()


    ################## MEASURE PERFORMANCE ##################################
    if "mv" in executed_models:
        evaluate = Evaluation_metrics(model_mvsoft,'keras',X_train.shape[0],plot=False)
        prob_Yx = np.tensordot(Z_train_pred_mvsoft, mv_conf_probas,axes=[[1],[0]])
        prob_Yxt = np.tile(prob_Yx, (T,1,1)).transpose([1,0,2])
        results1 = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvsoft)
        results_softmv_train += results1
        results_softmv_test += results2

        evaluate = Evaluation_metrics(model_mvhard,'keras',X_train.shape[0],plot=False)
        prob_Yx = np.tensordot(Z_train_pred_mvhard, mv_conf_onehot,axes=[[1],[0]])
        prob_Yxt = np.tile(prob_Yx, (T,1,1)).transpose([1,0,2])
        results1 = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_mvhard)

        results_hardmv_train += results1
        results_hardmv_test += results2

    if "ds" in executed_models:
        evaluate = Evaluation_metrics(model_ds,'keras',X_train.shape[0],plot=False)
        #Z_train_pred_ds, ds_conf
        prob_Yxt = None
        results1 = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_ds)
        
        results_ds_train += results1
        results_ds_test += results2
    
    if "raykar" in executed_models:
        evaluate = Evaluation_metrics(raykarMC,'raykar',plot=False)
        prob_Yxt = raykarMC.get_predictions_annot(X_train,data=Z_train_p_Ray)
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_Ray)
        
        results_raykar_trainA += results1_aux
        results_raykar_test += results2

    if "oursglobal" in executed_models:
        evaluate = Evaluation_metrics(gMixture_Global,'our1',plot=False) 
        aux = gMixture_Global.calculate_extra_components(X_train,y_obs,T=T,calculate_pred_annotator=True,p_z=Z_train_p_OG)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Global.get_confusionM()
        y_o_groups = gMixture_Global.get_predictions_groups(X_test,data=Z_test_p_OG).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OG = Z_test_p_OG.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OG,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_global_trainA += results1_aux
        results_ours_global_testA.append(results2[0])
        results_ours_global_test.append(results2[1])

    if "oursindividual" in executed_models:
        evaluate = Evaluation_metrics(gMixture_Ind_T,'our1',plot=False) 
        aux = gMixture_Ind_T.calculate_extra_components(X_train, A,calculate_pred_annotator=True,p_z=Z_train_p_OI_T,p_g=prob_Gt_OI_T)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Ind_T.get_confusionM()
        y_o_groups = gMixture_Ind_T.get_predictions_groups(X_test,data=Z_test_p_OI_T).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OI = Z_test_p_OI_T.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_indiv_T_trainA += results1_aux
        results_ours_indiv_T_testA.append(results2[0])
        results_ours_indiv_T_test.append(results2[1])

        evaluate = Evaluation_metrics(gMixture_Ind_K,'our1',plot=False) 
        aux = gMixture_Ind_K.calculate_extra_components(X_train, A,calculate_pred_annotator=True,p_z=Z_train_p_OI_K,p_g=prob_Gt_OI_K)
        predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...
        results1_aux = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)
        c_M = gMixture_Ind_K.get_confusionM()
        y_o_groups = gMixture_Ind_K.get_predictions_groups(X_test,data=Z_test_p_OI_K).argmax(axis=-1) #obtain p(y^o|x,g=m) and then argmax
        Z_test_pred_OI = Z_test_p_OI_K.argmax(axis=-1)
        results2 = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred_OI,conf_pred=c_M, y_o_groups=y_o_groups)

        results_ours_indiv_K_trainA += results1_aux
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
    get_mean_dataframes(results_softmv_train).to_csv("AirlineS_softMV_trainAnn.csv",index=False)
    get_mean_dataframes(results_softmv_train, mean_std=False).to_csv("AirlineS_softMV_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_softmv_test).to_csv("AirlineS_softMV_test.csv",index=False)
    get_mean_dataframes(results_softmv_test, mean_std=False).to_csv("AirlineS_softMV_test_std.csv",index=False)

    get_mean_dataframes(results_hardmv_train).to_csv("AirlineS_hardMV_trainAnn.csv",index=False)
    get_mean_dataframes(results_softmv_train, mean_std=False).to_csv("AirlineS_hardMV_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_hardmv_test).to_csv("AirlineS_hardMV_test.csv",index=False)
    get_mean_dataframes(results_hardmv_test, mean_std=False).to_csv("AirlineS_hardMV_test_std.csv",index=False)

if "ds" in executed_models:
    get_mean_dataframes(results_ds_train).to_csv("AirlineS_DS_trainAnn.csv",index=False)
    get_mean_dataframes(results_ds_train, mean_std=False).to_csv("AirlineS_DS_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ds_test).to_csv("AirlineS_DS_test.csv",index=False)
    get_mean_dataframes(results_ds_test, mean_std=False).to_csv("AirlineS_DS_test_std.csv",index=False)

if "raykar" in executed_models:
    get_mean_dataframes(results_raykar_trainA).to_csv("AirlineS_Raykar_trainAnn.csv",index=False)
    get_mean_dataframes(results_raykar_trainA,  mean_std=False).to_csv("AirlineS_Raykar_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_raykar_test).to_csv("AirlineS_Raykar_test.csv",index=False)
    get_mean_dataframes(results_raykar_test, mean_std=False).to_csv("AirlineS_Raykar_test_std.csv",index=False)

if "oursglobal" in executed_models:
    get_mean_dataframes(results_ours_global_trainA).to_csv("AirlineS_OursGlobal_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_global_trainA, mean_std=False).to_csv("AirlineS_OursGlobal_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ours_global_test).to_csv("AirlineS_OursGlobal_test.csv",index=False)
    get_mean_dataframes(results_ours_global_test, mean_std=False).to_csv("AirlineS_OursGlobal_test_std.csv",index=False)
    get_mean_dataframes(results_ours_global_testA).to_csv("AirlineS_OursGlobal_testAux.csv",index=False)
    get_mean_dataframes(results_ours_global_testA, mean_std=False).to_csv("AirlineS_OursGlobal_testAux_std.csv",index=False)

if "oursindividual" in executed_models:
    get_mean_dataframes(results_ours_indiv_T_trainA).to_csv("AirlineS_OursIndividualT_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_trainA, mean_std=False).to_csv("AirlineS_OursIndividualT_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_test).to_csv("AirlineS_OursIndividualT_test.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_test, mean_std=False).to_csv("AirlineS_OursIndividualT_test_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_testA).to_csv("AirlineS_OursIndividualT_testAux.csv",index=False)
    get_mean_dataframes(results_ours_indiv_T_testA, mean_std=False).to_csv("AirlineS_OursIndividualT_testAux_std.csv",index=False)

    get_mean_dataframes(results_ours_indiv_K_trainA).to_csv("AirlineS_OursIndividualK_trainAnn.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_trainA, mean_std=False).to_csv("AirlineS_OursIndividualK_trainAnn_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_test).to_csv("AirlineS_OursIndividualK_test.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_test, mean_std=False).to_csv("AirlineS_OursIndividualK_test_std.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_testA).to_csv("AirlineS_OursIndividualK_testAux.csv",index=False)
    get_mean_dataframes(results_ours_indiv_K_testA, mean_std=False).to_csv("AirlineS_OursIndividualK_testAux_std.csv",index=False)

print("Execution done in %f mins"%((time.time()-start_time_exec)/60.))
