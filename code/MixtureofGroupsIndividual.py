import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import DBSCAN,AffinityPropagation, MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import gc, keras, time, sys

from .learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from .learning_models import default_CNN,default_RNN,default_RNNw_emb,CNN_simple, RNN_simple #deep learning
from .representation import *
from .utils import softmax,estimate_batch_size


def aux_clusterize(data_to_cluster,M,DTYPE_OP='float32',option="hard",l=0.005):
    #Clusterize data#
    std = StandardScaler()
    data_to_cluster = std.fit_transform(data_to_cluster) 
        
    kmeans = MiniBatchKMeans(n_clusters=M, random_state=0,init='k-means++',batch_size=128)
    #KMeans(M,init='k-means++', n_jobs=-1,random_state=0)
    kmeans.fit(data_to_cluster)
    distances = kmeans.transform(data_to_cluster)

    if option=="fuzzy":
        probas_t = np.zeros_like(distances,dtype=DTYPE_OP)
        for t in range(probas_t.shape[0]):
            for m in range(probas_t.shape[1]):
                m_fuzzy = 1.2
                probas_t[t,m] = 1/(np.sum( np.power((distances[t,m]/(distances[t,:]+keras.backend.epsilon())), 2/(m_fuzzy-1)) ) + keras.backend.epsilon())
    elif option == "softmax":
        probas_t = softmax(-(distances+keras.backend.epsilon())/l).astype(DTYPE_OP)
    elif option == "softmax inv":
        probas_t = softmax(1/(l*distances+keras.backend.epsilon())).astype(DTYPE_OP)
    elif option == 'hard':
        probas_t = keras.utils.to_categorical(kmeans.labels_,num_classes=M)
    return probas_t
            
def clusterize_annotators(y_o,M,no_label=-1,bulk=True,cluster_type='mv_close',data=[],model=None,DTYPE_OP='float32',BATCH_SIZE=64,option="hard",l=0.005):
    start_time = time.time()
    if bulk: #Repeat version 
        if len(y_o.shape) == 2:
            M_itj = categorical_representation(y_o,no_label =no_label)
        else:
            M_itj = y_o.copy()
        mask_nan = M_itj.sum(axis=1,keepdims=True) == 0
        mask_nan = np.tile(mask_nan,(1,M_itj.shape[1],1))
        M_itj[mask_nan] = 1
        #M_itj = M_itj.astype(DTYPE_OP)
        #M_itj_norm = M_itj/M_itj.sum(axis=1,keepdims=True)#[:,None,:]
                
        if len(data) != 0:
            data_to_cluster = data.copy() #annotators_pca
        else:
            data_to_cluster = M_itj.transpose(1,0,2).reshape(M_itj.shape[1],M_itj.shape[0]*M_itj.shape[2])
        print("Doing clustering...",end='',flush=True)  
        probas_t = aux_clusterize(data_to_cluster,M,DTYPE_OP,option,l) #0.05 is close to one-hot
        print("Done!")
        #alphas_init = np.tensordot(M_itj_norm,probas_t, axes=[[1],[0]]) 
        alphas_init = np.tensordot(M_itj, probas_t, axes=[[1],[0]]) 
        alphas_init = alphas_init/alphas_init.sum(axis=-1,keepdims=True) #normalize here for efficiency
    else: #Global Version: y_o: is repeats
        if len(y_o.shape) == 2: 
            mv_soft = majority_voting(y_o,repeats=True,probas=True) 
        else:
            mv_soft = majority_voting(y_o,repeats=False,probas=True)
        if cluster_type=='loss': #cluster respecto to loss function
            aux_model = keras.models.clone_model(model)
            aux_model.compile(loss='categorical_crossentropy',optimizer=model.optimizer)
            aux_model.fit(data, mv_soft, batch_size=BATCH_SIZE,epochs=30,verbose=0)
            predicted = aux_model.predict(data,verbose=0)
        elif cluster_type == 'mv_close':
            predicted = np.clip(mv_soft, keras.backend.epsilon(), 1.)
       
        data_to_cluster = []
        for i in range(mv_soft.shape[0]):
            for j in range(mv_soft.shape[1]):
                ob = np.tile(keras.backend.epsilon(), mv_soft.shape[1])
                ob[j] = 1
                true = np.clip(predicted[i],keras.backend.epsilon(),1.)      
                f_l = distance_function(true, ob)  #funcion de distancia o similaridad
                data_to_cluster.append(f_l)  
        data_to_cluster = np.asarray(data_to_cluster)
        #if manny classes or low entropy?
        model = PCA(n_components=min(3,mv_soft.shape[0]) ) # 2-3
        data_to_cluster = model.fit_transform(data_to_cluster) #re ejecutar todo con esto
        print("Doing clustering...",end='',flush=True)
        probas_t = aux_clusterize(data_to_cluster,M,DTYPE_OP,option,l)
        print("Done!")
        alphas_init = probas_t.reshape(mv_soft.shape[0],mv_soft.shape[1],M)
    print("Get init alphas in %f mins"%((time.time()-start_time)/60.) )
    return alphas_init

def distance_function(predicted,ob):
    return -predicted*np.log(ob) #CE raw

def project_and_cluster(y_o,M_to_try=20,anothers_visions=True,DTYPE_OP='float32',printed=True,mode_project="pca"):
    ###another way to cluster..
    if len(y_o.shape) == 2:
        M_itj = categorical_representation(y_o,no_label =-1)
    else:
        M_itj = y_o.copy()
    data_to_cluster = M_itj.transpose(1,0,2).reshape(M_itj.shape[1],M_itj.shape[0]*M_itj.shape[2])
    data_to_cluster = data_to_cluster.astype(DTYPE_OP)
    
    if mode_project.lower() == "pca":
        model = PCA(n_components=4)
    elif mode_project.lower() == "tpca":
        model = TruncatedSVD(n_components=4)
    elif mode_project.lower() == "kpca":
        model = KernelPCA(n_components=4, kernel='rbf', n_jobs=-1)

    plot_data = model.fit_transform(data_to_cluster)
    to_return = [plot_data]
    
    if printed:
        model = BayesianGaussianMixture(n_components=M_to_try)
        model.fit(plot_data)
        M_founded = len(set(np.argmax(model.predict_proba(plot_data),axis=1))) 
        print("Bayesian gaussian mixture say is %d clusters "%M_founded)

        if anothers_visions:
            X_sim = metrics.pairwise_distances(plot_data,metric='euclidean',n_jobs=-1)
            #dos indicadores de numero de cluster
            model = DBSCAN(eps=np.mean(X_sim), min_samples=5, metric='precomputed', n_jobs=-1)
            model.fit(X_sim)
            print("DBSCAN say is %d clusters"%len(set(model.labels_)))
            model = AffinityPropagation(affinity='precomputed')
            model.fit(X_sim)
            print("Affinity Propagation say is %d clusters"%len(set(model.labels_)))

        to_return.append( M_founded )
    return to_return



class GroupMixtureInd(object):
    def __init__(self,input_dim,Kl,M=2,epochs=1,optimizer='adam',pre_init=0,dtype_op='float32'): 
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.input_dim = input_dim
        self.Kl = Kl #number of classes of the problem
        #params
        self.M = M #groups of annotators
        self.epochs = epochs
        self.optimizer = optimizer
        self.pre_init = pre_init
        self.DTYPE_OP = dtype_op
        
        self.Keps = keras.backend.epsilon() 
        self.priors = False #boolean of priors
        self.compile_z = False
        self.compile_g = False
        self.lambda_random = False
        #deleteeee
        self.seted_alphainit = False
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        """Get confusion matrices of every group p(y|g,z)"""  
        return self.betas.copy()
    def get_qestimation(self):
        """Get Q estimation param, this is Q_il(g,z) = p(g,z|x_i,a_il, r_il)"""
        return self.Qil_mgamma.copy()
    #deleteeeeee
    def set_alpha(self,alpha_init):
        """set alpha param with a previosuly method"""
        self.alpha_init = alpha_init.copy()
        self.seted_alphainit = True
        
    def define_model(self,tipo,start_units=1,deep=1,double=False,drop=0.0,embed=True,BatchN=True,h_units=128):
        """Define the base model p(z|x) and other structures"""
        self.type = tipo.lower()     
        if self.type =="sklearn_shallow" or self.type =="sklearn_logistic":
            self.base_model = LogisticRegression_Sklearn(self.epochs)
            self.compile_z = True
            return

        if self.type == "keras_shallow" or 'perceptron' in self.type: 
            self.base_model = LogisticRegression_Keras(self.input_dim,self.Kl)
            #It's not a priority, since HF has been shown to underperform RMSprop and Adagrad, while being more computationally intensive.
            #https://github.com/keras-team/keras/issues/460
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
            self.base_model = CNN_simple(self.input_dim,self.Kl,start_units,deep,double=double,BN=BatchN,drop=drop,dense_units=h_units)
        elif self.type=='simplernn' or self.type=='simple rnn' or 'rnn' in self.type:
            print("Needed params (units,deep,drop,embed?)")
            self.base_model = RNN_simple(self.input_dim,self.Kl,start_units,deep,drop=drop,embed=embed,len=0,out=start_units*2)
            #and what is with embedd
        self.base_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 
        self.max_Bsize_base = estimate_batch_size(self.base_model)
        self.compile_z = True
        
    def define_model_group(self, input_dim, tipo, start_units=64, deep=1, drop=0.0, BatchN=True, bias=False):
        #define model over annotators -- > p(g|a)
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.type_g = tipo.lower()  

        if self.type_g =="sklearn_shallow" or self.type_g =="sklearn_logistic":
            self.group_model = LogisticRegression_Sklearn(self.epochs)
            self.compile_g = True
            return 

        if self.type_g == "keras_shallow" or 'perceptron' in self.type_g: 
            self.group_model = LogisticRegression_Keras(input_dim, self.M, bias)
        elif self.type_g == "ff" or self.type_g == "mlp" or self.type_g=='dense': #classic feed forward
            print("Needed params (units,deep,drop,BatchN?)") #default activation is relu
            self.group_model = MLP_Keras(input_dim, self.M, start_units, deep, BN=BatchN,drop=drop)
        self.group_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 
        self.max_Bsize_group = estimate_batch_size(self.group_model)
        self.compile_g = True
        
    def get_predictions_z(self,X):
        """Return the predictions of the model p(z|x) if is from sklearn or keras"""
        if "sklearn" in self.type:
            return self.base_model.predict_proba(X) 
        else:
            return self.base_model.predict(X,batch_size=self.max_Bsize_base) #self.batch_size

    def aux_pred_g(self,A):
        """Return the predictions of the model p(g|a) if is from sklearn or keras"""
        if "sklearn" in self.type_g:
            return self.group_model.predict_proba(A) 
        else:
            return self.group_model.predict(A, batch_size=self.max_Bsize_group)

    def get_predictions_g(self,A, same_shape=True):
        """Return the predictions of the model if is from sklearn or keras"""
        if len(A.shape) != 1:
            return self.aux_pred_g(A)
        else: #come from batch of examples..
            A = self.flatten_il(A)
            to_return = self.aux_pred_g(A)
            if same_shape:
                to_return = self.reshape_il(to_return) #add some execution time... carefully
            return to_return

    def flatten_il(self,array):
        return np.concatenate(array)

    def reshape_il(self,array):
        to_return = []
        sum_Ti_n1 = 0
        for i in range(self.N):
            sum_Ti_n   = sum_Ti_n1
            sum_Ti_n1  = sum_Ti_n1 + self.T_i_all[i]
            to_return.append( array[sum_Ti_n : sum_Ti_n1] )
        del array
        gc.collect()
        return np.asarray(to_return)

    def define_priors(self,priors):
        """
            Priors with shape: (M,K,K), need counts for every group and every pair (k,k) ir global (M,K)
            The group m, given a class "k" is probably that say some class
            it is recomended that has full of ones
        """
        if type(priors) == str:
            if priors == "laplace":
                self.Bpriors = 1
            else:
                print("Prior string do not understand")
                return
        else:
            if len(priors.shape)==2:
                self.Bpriors = np.expand_dims(priors,axis=2)
            else:
                print("Error on prior")
        self.priors = True

    def init_E(self,X,Y_ann, A):
        """Realize the initialziation of the E step on the EM algorithm"""
        start_time = time.time()
        self.T_i_all = [y_ann.shape[0] for y_ann in Y_ann] 

        #-------> init Majority voting --- in another way-- check how..
        mv_probs_j = majority_voting(Y_ann,repeats=False,probas=True) # soft -- p(y=j|xi)
        
        #-------> Initialize p(z=gamma|xi,y=j,g): Combination of mv and belive observable
        lambda_group = np.ones((self.M),dtype=self.DTYPE_OP)  #or zeros
        if self.lambda_random:
            for m in range(self.M):
                lambda_group[m] = np.random.beta(1,1)
        print("Lambda by group: ",lambda_group)
        Zilm =  [] #np.zeros((self.N,self.Kl,self.M,self.Kl),dtype=self.DTYPE_OP)
        for i in range(self.N):
            #T_i = Y_ann[i].shape[0]
            #Zlm = np.zeros((T_i,self.M,self.Kl),dtype=self.DTYPE_OP)
            #for l in range(T_i):
                #for m in range(self.M):                
                    #Zlm[l, m ,:] = lambda_group[m]*mv_probs_j[i] + (1-lambda_group[m])*Y_ann[i,l] 
                #Zlm[l] = (lambda_group*mv_probs_j[i][:,None] + (1-lambda_group)*Y_ann[i][l][:,None] ).T
            Zlm = (mv_probs_j[i][None,:,None]*lambda_group + Y_ann[i][:,:,None]*(1-lambda_group) ).transpose(0,2,1)
            
            Zilm.append(Zlm)
        Zilm = np.asarray(Zilm)

        #calculate p(g|a) 
        #or clustering??
        #if arreglos en A son ortonormales .. generar representacion con matriz de conf.. (sino se hizo afuera)
              
        #-------> init q_il
        self.Qil_mgamma = []
        for i in range(self.N):
            self.Qil_mgamma.append( self.alpha_init[i][:,:,None] * Zilm[i] ) #self.alpha_init[:,:,:,None]*Zijm
        self.Qil_mgamma = np.asarray(self.Qil_mgamma)

        #-------> init betas
        self.betas = np.zeros((self.M,self.Kl,self.Kl),dtype=self.DTYPE_OP) 

        if self.pre_init != 0:
            print("Pre-train networks over *z* and over *g* on %d epochs..."%(self.pre_init),end='',flush=True)
            self.base_model.fit(X,mv_probs_j,batch_size=self.batch_size,epochs=self.pre_init,verbose=0)
            self.group_model.fit(self.flatten_il(A), self.flatten_il(alpha_init), batch_size=self.batch_size,epochs=self.pre_init,verbose=0)         
            #reset optimizer but hold weights--necessary for stability 
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)
            self.group_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)            
            print(" Done!")
        print("MV init: ",mv_probs_j.shape)
        print("Betas: ",self.betas.shape)
        print("Q estimate: ",self.Qil_mgamma.shape)
        gc.collect()
        self.init_exectime = time.time()-start_time
            
    def E_step(self, Y_ann, z_pred, g_pred):
        """ Realize the E step in matrix version"""   
        self.aux_for_like = []

        p_new = np.log( np.clip(z_pred, self.Keps, 1.) )[:,None,None,:] #safe logarithmn
        for i in range(self.N):
            a_new = np.log( np.clip(g_pred[i], self.Keps, 1.) )[:,:,None]  #safe logarithmn
            b_new = np.tensordot(Y_ann[i], np.log(self.betas + self.Keps),axes=[[1],[2]]) #safe logarithmn

            #maybe to low values?? -- check raykar control of zero values..
            self.Qil_mgamma[i] = np.exp( p_new[i] + a_new + b_new)
    
            self.aux_for_like.append( (self.Qil_mgamma[i].sum(axis=-1)).sum(axis=-1)) #p(y|x,a) --marginalized
            self.Qil_mgamma[i] = self.Qil_mgamma[i]/self.aux_for_like[i][:,None,None] #normalize
        self.aux_for_like = self.flatten_il(self.aux_for_like) #to logL
        
    def M_step(self,X, Y_ann, A): 
        """ Realize the M step"""
        Qil_mgamma_flat = self.flatten_il(self.Qil_mgamma)
        A_flat = self.flatten_il(A)
        Y_ann_flat = self.flatten_il(Y_ann)

        #-------> base model     
        r_estimate = np.zeros((self.N,self.Kl),dtype=self.DTYPE_OP)
        for i in range(self.N):
            r_estimate[i] = self.Qil_mgamma[i].sum(axis=0).sum(axis=0) #create the "estimate"/"ground truth"

        if "sklearn" in self.type:#train to learn p(z|x)
            self.base_model.fit(X, np.argmax(z_estimate,axis=1) ) 
        else:
            self.base_model.fit(X, r_estimate, batch_size=self.batch_size,epochs=self.epochs,verbose=1) 
    
        #-------> alpha 
        Qil_m_flat = Qil_mgamma_flat.sum(axis=-1)  #qil(m)

        if "sklearn" in self.type_g:#train to learn p(g|a)
            self.group_model.fit(A_flat, np.argmax(Qil_m_flat,axis=1) ) 
        else:
            self.group_model.fit(A_flat, Qil_m_flat, batch_size=self.N/self.batch_size,epochs=self.epochs,verbose=1) #batch should be prop to N

        #-------> beta
        self.betas =  np.tensordot(Qil_mgamma_flat, Y_ann_flat , axes=[[0],[0]]) # ~p(yo=j|g,z) 
        if self.priors:
            self.betas += self.Bpriors #priors has to be shape: (M,Kl,Kl)--read define-prior functio
        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        """ Compute the log-likelihood of the optimization schedule"""
        return np.sum( np.log(self.aux_for_like+self.Keps) )  #safe logarithm
                                                  
    def train(self,X_train,Y_ann_train, A_train ,batch_size=64,max_iter=500,relative=True,tolerance=3e-2):
        if not self.compile_z:
            print("You need to create the model first, set .define_model")
            return
        if not self.compile_g:
            print("You need to create the model of groups first, set .define_model_group")
            return
        if len(Y_ann_train.shape) == 1 and len(A_train.shape)==0:
            print("ERROR! Needed Y and A in variable length array")
            return
        print("Initializing new EM...")
        self.batch_size = batch_size
        self.N = X_train.shape[0]
        #self.T = #calculate in some
        self.init_E(X_train,Y_ann_train,A_train)

        logL = []
        stop_c = False
        tol,old_betas = np.inf,np.inf
        self.current_iter = 1
        while(not stop_c):
            start_time = time.time()
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time2 = time.time()
            self.M_step(X_train, Y_ann_train, A_train)
            print("M step en %f sec"%(time.time()-start_time2))

            print(" done,  E step:",end='',flush=True)

            start_time2 = time.time()
            predictions_z = self.get_predictions_z(X_train) #p(z|x)
            predictions_g = self.get_predictions_g(A_train) #p(g|a)
            self.E_step(Y_ann_train, predictions_z, predictions_g)
            print("E step en %f sec"%(time.time()-start_time2))

            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL())
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])                    
                if relative:
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten().copy()         
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance):
                stop_c = True 
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self,X,Y_ann,A,batch_size=64,max_iter=50,tolerance=3e-2):
        """
            A stable schedule to train a model on this formulation
        """
        if not self.priors:
            self.define_priors('laplace') #needed..
        logL_hist = self.train(X,Y_ann,A, batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,relative=True)
        return logL_hist
    
    def multiples_run(self,Runs,X,Y_ann,A,batch_size=64,max_iter=50,tolerance=3e-2): 
        """
            Run multiples max_iter of EM algorithm, same start
        """
        if Runs==1:
            return self.stable_train(X,Y_ann,A,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance), 0

        #maybe lamda random here
        if not self.priors:
            self.define_priors('laplace') #needed!
            
        found_betas = []
        found_model_g = []
        found_model_z = []
        found_logL = []
        iter_conv = []
        clonable_model_z = keras.models.clone_model(self.base_model) #architecture to clone
        clonable_model_g = keras.models.clone_model(self.group_model) #architecture to clone
        for run in range(Runs):
            self.base_model = keras.models.clone_model(clonable_model_z) #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)
            
            self.group_model = keras.models.clone_model(clonable_model_g) #reset-weigths
            self.group_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,Y_ann,A,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,relative=True) #here the models get resets
            
            found_betas.append(self.betas.copy())
            found_model_g.append(self.group_model.get_weights())
            found_model_z.append(self.base_model.get_weights()) 
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            del self.base_model, self.group_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        if type(clonable_model_z.layers[0]) == keras.layers.InputLayer:
            self.base_model = keras.models.clone_model(clonable_model_z) #change
            self.group_model = keras.models.clone_model(clonable_model_g)
        else:
            it = keras.layers.Input(shape=X.shape[1:])
            self.base_model = keras.models.clone_model(clonable_model_z, input_tensors=it) 
            it = keras.layers.Input(shape=A.shape[1:])
            self.group_model = keras.models.clone_model(clonable_model_g, input_tensors=it) 
        self.base_model.set_weights(found_model_z[indexs_sort[0]])
        self.group_model.set_weights(found_model_g[indexs_sort[0]])
        self.E_step(Y_ann, self.get_predictions_z(X), self.get_predictions_g(A)) #to set up Q
        print("Multiples runs over Ours Individual, Epochs to converge= ",np.mean(iter_conv)) #maybe change name
        return found_logL,indexs_sort[0]

    def get_predictions_group(self,m,X):
        """ Predictions of group "m", p(y^o | xi, g=m) """
        p_z = self.get_predictions_z(X)
        p_y_m = np.zeros(p_z.shape)
        for i in range(self.N):
            p_y_m[i] = np.tensordot(p_z[i,:] ,self.betas[m,:,:],axes=[[0],[0]] ) # sum_z p(z|xi) * p(yo|z,g=m)
        return p_y_m 
    
    def get_predictions_groups(self,X,data=[]):
        """ Predictions of all groups , p(y^o | xi, g) """
        if len(data) != 0:
            p_z = data
        else:
            p_z = self.get_predictions_z(X)
        predictions_m = np.tensordot(p_z ,self.betas,axes=[[1],[1]] ) #sum_z p(z|xi) * p(yo|z,g)
        return predictions_m

    def calculate_extra_components(self,X, A, calculate_pred_annotator=True,p_z=[],p_g=[]):
        """
            Measure indirect probabilities through bayes and total probability of annotators
        """
        predictions_m = self.get_predictions_groups(X,data=p_z) #p(y^o|x,g=m)
        
        if len(p_g) != 0:
            prob_Gt = p_g
        else:
            prob_Gt = self.get_predictions_g(A)

        prob_Yzt = np.tensordot(prob_Gt, self.get_confusionM(),axes=[[1],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)
    
        prob_Yxt = None
        if calculate_pred_annotator:
            prob_Yxt = np.tensordot(predictions_m, prob_Gt, axes=[[1],[1]]).transpose(0,2,1) #p(y^o|x,t) = sum_g p(g|t) *p(yo|x,g)            
        return predictions_m, prob_Gt, prob_Yzt, prob_Yxt
    
    def calculate_Yz(self,prob_Gt):
        """ Calculate global confusion matrix"""
        alphas = np.mean(prob_Gt, axis=0)
        return np.sum(self.betas*alphas[:,None,None],axis=0)
    
    def get_annotator_reliability(self,X,a):
        """Get annotator reliability, based on his representation:"""
        if len(a.shape) ==1:
            a = [a]     
        prob_Gt = self.get_predictions_g(a)[0]
        prob_Yzt = np.tensordot(prob_Gt, self.get_confusionM(),axes=[[0],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)
        return prob_Yzt #do something with it
