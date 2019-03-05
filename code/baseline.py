import gc, keras, time, sys
import numpy as np
from keras.models import clone_model
from .learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from .learning_models import default_CNN,default_RNN,default_RNNw_emb,CNN_simple, RNN_simple #deep learning
from .representation import *
from . import dawid_skene 

class LabelInference(object): #no predictive model
    def __init__(self,labels,tolerance,type_inf): 
        """
            *labels is annotations : should be (N,T) with symbol of no annotations =-1
        """
        if 'all' == type_inf.lower():
            type_inf = "mv and d&s"

        if 'mv' in type_inf.lower() or 'majority voting' in type_inf.lower():
            if len(labels.shape) ==2:
                self.y_obs_categ = set_representation(labels,'onehot') #for MV
            else:
                self.y_obs_categ = labels.copy()
        if 'd&s' in type_inf.lower() or "dawid" in type_inf.lower():
            self.annotations = set_representation(labels,'dawid') #for D&S

        self.Tol = tolerance #tolerance of D&S
        self.type = type_inf
        self.T = labels.shape[1]
         
        
    def mv_labels(self, type_return):
        mv_probas = majority_voting(self.y_obs_categ,repeats=False,probas=True) #aka soft-MV

        if type_return.lower() == 'probas': 
            return mv_probas
        elif type_return.lower() == "classes":
            return np.argmax(mv_probas,axis=1)
        elif type_return.lower() == 'onehot' or type_return.lower() == 'one-hot': #also known as hard-MV
            return keras.utils.to_categorical(np.argmax(mv_probas,axis=1))        
        
    def DS_labels(self):
        # https://github.com/dallascard/dawid_skene
        aux = dawid_skene.run(self.annotations,tol=self.Tol, max_iter=100, init='average')
        (_, _, _, _, class_marginals, error_rates, groundtruth_estimate) = aux
        return groundtruth_estimate
    
    def train(self):
        if 'mv' in type_inf.lower() or 'majority voting' in type_inf.lower(): 
            self.mv_classes = self.mv_labels('classes')
            self.mv_onehot = self.mv_labels('onehot')
            self.mv_probas = self.mv_labels('probas')
        elif 'd&s' in type_inf.lower() or "dawid" in type_inf.lower(): 
            self.ds_labels = self.DS_labels()
        gc.collect()

class RaykarMC(object):
    def __init__(self,input_dim,Kl,T,epochs=1,optimizer='adam',DTYPE_OP='float32'): #default stable parameteres
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.input_dim = input_dim
        self.Kl = Kl #number of classes of the problem
        self.T = T #number of annotators
        #params:
        self.epochs = epochs
        self.optimizer = optimizer
        self.DTYPE_OP = DTYPE_OP

        self.compile=False
        self.Keps = keras.backend.epsilon()
        self.priors=False #boolean of priors
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        """Get confusion matrices of every annotator p(yo^t|,z)"""  
        return self.betas
    def get_qestimation(self):
        return self.Qi_gamma

    def define_model(self,tipo,start_units=1,deep=1,double=False,drop=0.0,embed=True,BatchN=True):
        """Define the network of the base model"""
        self.type = tipo.lower()     
        if self.type == "keras_shallow" or 'perceptron' in self.type: 
            self.base_model = LogisticRegression_Keras(self.input_dim,self.Kl)
            #It's not a priority, since HF has been shown to underperform RMSprop and Adagrad, while being more computationally intensive.
            #https://github.com/keras-team/keras/issues/460
        elif self.type =="sklearn_shallow" or self.type =="sklearn_logistic":
            self.base_model = LogisticRegression_Sklearn(self.epochs)
            self.compile = True
            return
        elif self.type=='defaultcnn' or self.type=='default cnn':
            self.base_model = default_CNN(self.input_dim,self.Kl)
        elif self.type=='defaultrnn' or self.type=='default rnn':
            self.base_model = default_RNN(self.input_dim,self.Kl)
        elif self.type=='defaultrnnE' or self.type=='default rnn E': #with embedding
            self.base_mode = default_RNNw_emb(self.input_dim,self.Kl,len) #len is the length of the vocabulary

        elif self.type == "ff" or self.type == "mlp" or self.type=='dense': #classic feed forward
            print("Needed params (units,deep,drop,BatchN?)") #default activation is relu
            self.base_model = MLP_Keras(self.input_dim,self.Kl,start_units,deep,BN=BatchN,drop=drop)

        elif self.type=='simplecnn' or self.type=='simple cnn' or 'cnn' in self.type:
            print("Needed params (units,deep,drop,double?,BatchN?)") #default activation is relu
            self.base_model = CNN_simple(self.input_dim,self.Kl,start_units,deep,double=double,BN=BatchN,drop=drop)
        
        elif self.type=='simplernn' or self.type=='simple rnn' or 'rnn' in self.type:
            print("Needed params (units,deep,drop,embed?)")
            self.base_model = RNN_simple(self.input_dim,self.Kl,start_units,deep,drop=drop,embed=embed,len=0,out=start_units*2)

        self.base_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 
        self.compile = True
        
    def define_priors(self,priors):
        """
            Ojo que en raykar los priors deben tener una cierta forma (T,K,K) o hacerlos globales (T,K)
            para cualquier variable obs
            El obs t, dado que la clase "k" que tan probable es que diga que es una clase..
            se recomienda que sea uno
        """
        if type(priors) == str:
            if priors == "laplace":
                priors = 1
            else:
                print("Prior string do not understand")
                return
        else:
            if len(priors.shape)==2:
                priors=np.expand_dims(priors,axis=2)
        self.Mpriors = priors
        self.priors = True
        
    def init_E(self,X,y_ann):
        self.N = X.shape[0]
        #init p(z|x)
        mv_probs = majority_voting(y_ann,repeats=False,probas=True) #Majority voting start
        #init betas
        self.betas = np.zeros((self.T,self.Kl,self.Kl),dtype=self.DTYPE_OP)
        #init qi
        self.Qi_gamma = mv_probs
        print("Betas shape: ",self.betas.shape)
        print("Q estimate shape: ",self.Qi_gamma.shape)
                
    def E_step(self,X,y_ann,predictions=[]):
        if len(predictions)==0:
            predictions = self.get_predictions(X)
        
        #calculate sensitivity-specificity 
        a_igamma = np.tensordot(y_ann, np.log(self.betas + self.Keps),axes=[[1,2],[0,2]])
        a_igamma = a_igamma.astype(self.DTYPE_OP)
        aux = np.log(predictions + self.Keps) + a_igamma
        
        self.sum_unnormalized_q = np.sum(np.exp(aux),axis=-1)# p(y1,..,yt|x) #all anotations probabilities

        self.Qi_gamma = np.exp(aux-aux.max(axis=-1,keepdims=True)) #return to actually values
        self.Qi_gamma = self.Qi_gamma/np.sum(self.Qi_gamma,axis=-1)[:,None] #normalize q

    def M_step(self,X,y_ann): 
        #-------> base model ---- train to learn p(z|x)
        if "sklearn" in self.type:
            self.base_model.fit(X,np.argmax(self.Qi_gamma,axis=-1)) #to one hot 
        else:
            #epochs=1 as Rodriges says. and batch size as default
            history = self.base_model.fit(X,self.Qi_gamma,batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
        
        #-------> beta
        self.betas = np.tensordot(self.Qi_gamma, y_ann, axes=[[0],[0]]).transpose(1,0,2)
        if self.priors: #as a annotator not label all data:
            self.betas += self.Mpriors
        self.betas = self.betas/ np.sum(self.betas,axis=-1)[:,:,None] #normalize
    
    def compute_logL(self):#,yo,predictions):
        return np.sum( np.log( self.sum_unnormalized_q +self.Keps))
        
    def train(self,X_train,yo_train,batch_size=32,max_iter=500,relative=True,val=False,tolerance=1e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        print("Initializing new EM...")
        self.init_E(X_train,yo_train)
        self.batch_size = batch_size

        logL = []
        stop_c = False
        old_betas,tol = np.inf, np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d \nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train,yo_train)
            print(" done,  E step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x)
            self.E_step(X_train,yo_train,predictions)
            print(" done //  (in %.2f sec)\t"%(time.time()-start_time),end='',flush=True)
            #compute lowerbound
            logL.append(self.compute_logL())
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2]) #absolute
                if relative: #relative
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps))
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten() 
            if val:
                print("F1: %.4f"%(f1_score(Z_train,predictions.argmax(axis=-1),average='micro')),end='',flush=True)
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance):
                stop_c = True
        print("Finished training!")
        gc.collect()
        return logL
            
    def get_predictions(self,X,batch_size=None):
        if "sklearn" in self.type:
            return self.base_model.predict_proba(X) #or predict
        else:
            return self.base_model.predict(X,batch_size=batch_size)
        
    def stable_train(self,X,y_ann,batch_size=64,max_iter=50,tolerance=1e-2):
        self.define_priors('laplace') #cada anotadora dijo al menos una clase
        logL_hist = self.train(X,y_ann,batch_size=batch_size,max_iter=max_iter,relative=True,val=False,tolerance=tolerance)
        return logL_hist
    
    def multiples_run(self,Runs,X,y_ann,batch_size=64,max_iter=50,tolerance=1e-2):  #tolerance can change
        self.define_priors('laplace') #cada anotadora dijo al menos una clase
     
        found_betas = []
        found_model = []
        found_logL = []
        for run in range(Runs):
            self.base_model = clone_model(self.base_model) #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,y_ann,batch_size=batch_size,max_iter=max_iter,relative=True,tolerance=tolerance) 
            
            found_betas.append(self.betas.copy())
            found_model.append(self.base_model) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        self.base_model = found_model[indexs_sort[0]]
        self.E_step(X,y_ann,predictions=self.get_predictions(X)) #to set up Q
        gc.collect()
        return found_logL,indexs_sort[0]

    def get_predictions_annot(self,X):
        """ Predictions of all annotators , p(y^o | xi, t) """
        p_z = self.get_predictions(X)
        predictions_a= np.tensordot(p_z ,self.betas,axes=[[1],[1]] ) # sum_z p(z|xi) * p(yo|z,t)
        return predictions_a.transpose(1,0,2)

    def get_annotator_reliability(self,t):
        """Get annotator reliability, based on his identifier: t"""
        conf_M = self.betas[t,:,:]
        return conf_M #do something with it