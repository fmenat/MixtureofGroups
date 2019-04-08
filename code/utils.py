from sklearn.metrics import confusion_matrix,f1_score
from sklearn.preprocessing import normalize
import itertools, keras, math,gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import entropy

def get_confusionM(pred,y_ann):
    """
        * pred is prediction probabilities or one hot, p(z=gamma|x)
        * y_ann is annotator probabilities shape is (N,T,K)
    """
    aux = np.tensordot(pred, y_ann, axes=[[0],[0]]).transpose(1,0,2)
    return aux/np.sum(aux, axis=-1)[:,:,None] #normalize

def plot_confusion_matrix(conf, classes,title="Estimated",text=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(conf, interpolation='nearest', cmap=cm.YlOrRd)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = conf.max() / 2.
    if text:
        for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
            plt.text(j, i, format(conf[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if conf[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_keras(model,x,y,classes):
    y_pred_ohe = model.predict_classes(x)
    conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred_ohe)
    conf_matrix = normalize(conf_matrix, axis=1, norm='l1')
    plot_confusion_matrix(conf_matrix,classes)

def calculate_f1_keras(model,x,y):
    return f1_score(y_true=y, y_pred=model.predict_classes(x),average='micro')

def softmax(Xs):
    """Compute softmax values for each sets of scores in x."""
    values =[]
    for x in Xs:
        e_x = np.exp(x - np.max(x))
        values.append(e_x / e_x.sum())
    return np.asarray(values)

def distance_2_centroid(matrixs):
    """ Calculate inertia of all the confusion matrixs, based on Jensen-Shannon Divergence"""
    value = []
    for m1 in range(matrixs.shape[0]):
        for m2 in range(m1+1,matrixs.shape[0]):
            value.append(JS_confmatrixs(matrixs[m1],matrixs[m2]))
    return np.mean(value)

def calculate_diagional_mean(conf_matrix): #weight?
    """Calculate the Mean of the diagional of the confusion matrixs"""
    return np.mean([conf_matrix[l,l] for l in range(len(conf_matrix)) ])

def calculate_spamm_score(conf_matrix):
    """Mean - off diagonal"""
    return np.mean([conf_matrix[l,l]- np.mean(np.delete(conf_matrix[:,l],l)) for l in range(len(conf_matrix))])

def calculateKL_matrixs(confs_pred,confs_true):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    Kls = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t): #true
            Kls[m1] = KL_confmatrixs(confs_pred[m1],confs_true[m1])
        return Kls
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def calculateJS_matrixs(confs_pred,confs_true):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    JSs = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t): #true
            JSs[m1] = JS_confmatrixs(confs_pred[m1],confs_true[m1])
        return JSs
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def calculateNormF_matrixs(confs_pred,confs_true):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    NormFs = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t):
            NormFs[m1] = np.sqrt(np.sum((confs_pred[m1]-confs_true[m1])**2))/confs_pred[m1].shape[0]
            #np.linalg.norm(confs_pred[m1]-confs_true[m1], ord='fro')/confs_pred[m1].shape[0]
        return NormFs
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def compare_conf_mats(pred_conf_mat,true_conf_mat=[]):
    classes = np.arange(pred_conf_mat[0].shape[0])
    sp = plt.subplot(1,2,2)
    plt.imshow(pred_conf_mat, interpolation='nearest', cmap=cm.YlOrRd)
    plt.title("Estimated")
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.tight_layout()

    if len(true_conf_mat) != 0:
	    sp1 = plt.subplot(1,2,1)
	    plt.imshow(true_conf_mat, interpolation='nearest', cmap=cm.YlOrRd)
	    plt.title("True")
	    plt.xticks(np.arange(len(classes)), classes, rotation=45)
	    plt.yticks(np.arange(len(classes)), classes)
	    #plt.ylabel('True label')
	    #plt.xlabel('Predicted label')
	    plt.tight_layout()
    plt.show()

def KL_confmatrixs(conf_pred,conf_true):
    """
        * mean of KL between rows of confusion matrix: 1/K sum_z KL_y(p(y|z)|q(y|z))
    """ #mmean or sum??
    conf_pred = np.clip(conf_pred,1e-7,1.)
    conf_true = np.clip(conf_true,1e-7,1.)
    return np.mean([entropy(conf_true[j_z,:], conf_pred[j_z,:]) for j_z in range(conf_pred.shape[0])])

def JS_confmatrixs(conf_pred,conf_true):
    """
        * Jensen-Shannon Divergence between rows of confusion matrix (arithmetic average)
    """
    conf_pred = np.clip(conf_pred,1e-7,1.)
    conf_true = np.clip(conf_true,1e-7,1.)
    aux = 0.5*conf_pred + 0.5*conf_true
    return (0.5*KL_confmatrixs(aux,conf_pred) + 0.5*KL_confmatrixs(aux,conf_true))/np.log(2) #value between 0 and 1
    
    
def Entropy_confmatrix(conf_ma):
    """
        * Mean of entropy on rows of confusion matrix: mean H(q(y|z))
    """
    return np.mean([entropy(conf_ma[j_z]) for j_z in range(conf_ma.shape[0])])

def findmatch_confindexs(confs_pred,confs_true):
    """
        * Find all match between all the confusion matrices, based on KL between confs
        * Work with same conf matrices and if predicted is less than true
    """
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    print("There are %d real and %d predicted Conf matrices"%(M_t,M_p))
    order_KLs = np.zeros((M_t,M_p))
    indexs_tuple = []
    for m1 in range(M_t): #true
        for m2 in range(M_p): #prediction
            order_KLs[m1,m2] = KL_confmatrixs(confs_pred[m2],confs_true[m1])
            indexs_tuple.append([m1,m2])
    match_founded = {} #find best match on real conf matrixs
    if M_p <= M_t:
        for valor in np.argsort(order_KLs.flatten()): #find the minimum divergence/distance
            if indexs_tuple[valor][0] not in  match_founded: #if real conf not match up yet
                match_founded[indexs_tuple[valor][0]] =  [indexs_tuple[valor][1]]
    elif M_p>M_t: #find similars--close to cluster
        for m2 in range(M_p):
            valor = np.argmin(order_KLs[:,m2]) #find centroid on true (minimum divergence/distance)
            if valor not in  match_founded: #if real conf not match up yet
                match_founded[valor] =  [m2]
            else:
                match_founded[valor].append(m2)
        #in this type of search a true matrix could not be found
        cannot_found = M_t - len(match_founded.keys())
        if cannot_found > 0:
            print("%d real conf matrices cannot be found"%(cannot_found) )
            for m1 in range(M_t):
                if m1 not in match_founded:
                    match_founded[m1] = []
    return match_founded

class EarlyStopRelative(keras.callbacks.Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                restore_best_weights=False):
        super(EarlyStopRelative,self).__init__()
        
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        
        
    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best2 = self.best
        self.best3 = self.best
        self.b_before = self.best

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)        
        if current is None:
            return

        if epoch==0:
            self.best = current
            return
        
        delta_conv = np.abs(self.best-current)/self.best #relative 
        if self.monitor_op(-self.min_delta, delta_conv):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

def plot_Mchange(logL_Mchange,
         accTR_Mchange,
         accTE_Mchange,
         best_group_acc_Mchange,
         probas_Mchange,
         divergence1_Mchange,
        divergence2_Mchange,
         probGt_Mchange,
         inertia_Mchange,
                best_group=True):
    def add_plot(aux):
        aux.xticks(M_values)
        aux.xlabel("M change")
        aux.legend()
    #first some plots
    M_values = range(1,1+len(logL_Mchange))
    entropy_values = [entropy(value)/np.log(len(value)) for value in probas_Mchange]
    
    aux = math.ceil(len(M_values)/3)
    f,axx = plt.subplots(3,aux,figsize=(15,7))
    for m in range(len(M_values)):
        axx[int(m/aux),m%aux].bar(range(len(probas_Mchange[m])),probas_Mchange[m])
        axx[int(m/aux),m%aux].set_title("#%d groups"%(m+1))
    f.tight_layout()
    plt.show()

    try:
        plt.figure(figsize=(15,5))
        for m in range(len(M_values)):
            plt.plot(range(len(logL_Mchange[m])),logL_Mchange[m],'o-',label="Log-like training #"+str(m+1))
        plt.legend()
        plt.show()
        plot_logL = [L[-1] for L in logL_Mchange]
        plt.figure(figsize=(15,5))
        plt.plot(M_values,plot_logL,label="Log-like final")
        add_plot(plt) #add ticks, x label and legend
        plt.show()
    except:
        plt.clf() #clf()
        #plot_logL = [L[-1] for L in logL_Mchange]
        plt.figure(figsize=(15,5))
        plt.plot(M_values,logL_Mchange,'o-',label="Log-like final")
        add_plot(plt) #add ticks, x label and legend
        plt.show()
    
    plt.figure(figsize=(15,5))
    plt.plot(M_values,divergence2_Mchange,'o-',label="Divergence JS to real T matrixs")
    plt.plot(M_values,divergence1_Mchange,'o-',label="Norm F to real T matrixs")
    plt.plot(M_values,inertia_Mchange,'o-',label="Inertia of M matrixs")
    add_plot(plt) #add ticks, x label and legend
    plt.ylim(0)
    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(M_values,accTR_Mchange,'o-',label="Acc training")
    plt.plot(M_values,accTE_Mchange,'o-',label="Acc val")
    if best_group:
        plt.plot(M_values,best_group_acc_Mchange,'o-',label="Acc val best group")
    plt.plot(M_values,entropy_values,'o-',label="Entropy of p(g)")
    add_plot(plt) #add ticks, x label and legend
    plt.ylim(0,1)
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.plot(M_values,entropy_values,'o-',label="Entropy of p(g)")
    add_plot(plt) #add ticks, x label and legend
    plt.ylim(0,1)
    plt.show()