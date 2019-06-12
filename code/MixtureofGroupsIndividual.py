import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import DBSCAN,AffinityPropagation, MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score
import gc, keras, time, sys

from .learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from .learning_models import default_CNN,default_RNN,default_RNNw_emb,CNN_simple, RNN_simple #deep learning
from .representation import *
from .utils import softmax


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
    def __init__(self,input_dim,Kl,M=2,epochs=1,optimizer='adam',pre_init=10,dtype_op='float32'): 
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
        self.priors=False #boolean of priors
        self.compile=False
        self.seted_alphainit = False
        self.lambda_random = False
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        """Get confusion matrices of every group p(y|g,z)"""  
        return self.betas.copy()
    def get_qestimation(self):
        """Get Q estimation param, this is Q_il(g,z) = p(g,z|x_i,a_il, r_il)"""
        return self.Qil_mgamma.copy()
        
    def define_model(self,tipo,start_units=1,deep=1,double=False,drop=0.0,embed=True,BatchN=True,h_units=128):
        """Define the base model and other structures"""
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
            #podria ser el maximo

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

        #if not (self.type == "keras_shallow" or self.type=="keras_perceptron"): 
        #    self.base_model = create_network(self.Kl,self.input_dim,tipo,info,infoextractor_network=aux_info,embedding_info=emb_info)
        #     future..
        self.base_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 
        self.compile = True
        
    def define_model_groups(self, input_dim, tipo, start_units=128,deep=1,drop=0.0, BatchN=True):
        #define model over annotators -- > p(g|a)
        self.group_model = None
        return
        
    def get_predictions_base(self,X):
        """Return the predictions of the model if is from sklearn or keras"""
        if "sklearn" in self.type:
            return self.base_model.predict_proba(X) 
        else:
            return self.base_model.predict(X,batch_size=self.batch_size)
        
    def get_predictions_group(self,A):
        """Return the predictions of the model if is from sklearn or keras"""
        if "sklearn" in self.type:
            return self.group_model.predict_proba(A) 
        else:
            return self.group_model.predict(A, batch_size=self.batch_size)
    
    def init_E(self,X,y_ann):
        """Realize the initialziation of the E step on the EM algorithm"""
        
        ########## RE DONE...
        
        #calcualte p(g|a)
        self.alpha_init = None
        
        
        #-------> init Majority voting --- in another way-- check how..
        self.mv_probs_j = majority_voting(r,repeats=True,probas=True) # soft -- p(y=j|xi)
        
        print("Pre-train network on %d epochs..."%(self.pre_init),end='',flush=True)
        if self.pre_init != 0:
            self.base_model.fit(X,self.mv_probs_j,batch_size=self.batch_size,epochs=self.pre_init,verbose=0)
            #reset optimizer but hold weights--necessary for stability 
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)
        print(" Done!")
        
        #-------> Initialize p(z=gamma|xi,y=j,g): Combination of mv and belive observable
        lambda_group = np.ones((self.M),dtype=self.DTYPE_OP)  #or zeros
        if self.lambda_random:
            for m in range(self.M):
                lambda_group[m] = np.random.beta(1,1)
        print("Lambda by group: ",lambda_group)
        Zijm = np.zeros((self.N,self.Kl,self.M,self.Kl),dtype=self.DTYPE_OP)
        for j_ob in range(self.Kl):
            onehot = np.tile(self.Keps, self.Kl)
            onehot[j_ob] = 1. #all belive in the observable
            for m in range(self.M):                
                Zijm[:,j_ob,m,:] = lambda_group[m]*self.mv_probs_j + (1-lambda_group[m])*onehot 
          
        #-------> init q_il
        self.Qil_mgamma = None #how to do shape?? self.alpha_init[:,:,:,None]*Zijm
        
        #-------> init betas
        self.betas = np.zeros((self.M,self.Kl,self.Kl),dtype=self.DTYPE_OP) 

        print("MV init: ",self.mv_probs_j.shape)
        print("Betas: ",self.betas.shape)
        print("Q estimate: ",self.Qil_mgamma.shape)
        del self.mv_probs_j
            
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
                print("Remember to prior alphas")
            else:
                print("Error on prior")
        self.priors = True
        
    def E_step(self, X, Y_ann, z_pred, g_pred):
        """ Realize the E step in matrix version"""   
        
        ########## RE DONE... in for way..
        
        #similar to raykar.. check

        self.Qil_mgamma = None
        self.aux_for_like = #sum over p(z, g,y)
        
    def M_step(self,X, Y_ann, A): 
        """ Realize the M step"""
        #-------> base model
        
        #redone in for way...
        Qil_gamma = self.Qil_mgamma.sum(axis=-2) #qil(gamma)
        z_estimate = Qil_gamma.sum(axis=1) #create the "estimate"/"ground truth"
        
        if "sklearn" in self.type:#train to learn p(z|x)
            self.base_model.fit(X, np.argmax(z_estimate,axis=1) ) 
        else:
            self.base_model.fit(X, z_estimate, batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
    
        #-------> alpha 
        
        #redone in for mode...
        Qil_m = self.Qil_mgamma.sum(axis=-1) #qil(m)
        
        self.group_model.fit(A, Qil_m, batch_size=self.batch_size,epochs=self.epochs,verbose=0)         
        
        #-------> beta
        Y_ann y q...
        for j_ob in range(self.Kl):
            self.betas[:,:,j_ob] = np.tensordot(self.Qij_mgamma[:,j_ob,:,:],r[:,j_ob], axes=[[0],[0]]) # ~p(yo=j|g,z)              
        if self.priors:
            self.betas += self.Bpriors #priors has to be shape: (M,Kl,Kl)--read define-prior functio
        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        """ Compute the log-likelihood of the optimization schedule"""
        #do in for mode..
        return np.sum( np.log(self.aux_for_like+self.Keps) )  #safe logarithm
                                                  
    def train(self,X_train,Y_ann_train, A_train ,batch_size=64,max_iter=500,relative=True,val=False,tolerance=3e-2):
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        print("Initializing new EM...")
        self.batch_size = batch_size
        self.N = X_train.shape[0]
        #self.T = #calculate in some way..
        start_time = time.time()
        self.init_E(X_train,Y_ann_train)
        self.init_exectime = time.time()-start_time
        
        logL = []
        stop_c = False
        tol,old_model,old_betas = np.inf,np.inf,np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train, Y_ann_train, A_train)
            print(" done,  E step:",end='',flush=True)
            predictions_z = self.get_predictions_base(X_train) #p(z|x)
            predictions_g = self.get_predictions_group(A_train) #p(g|a)
            self.E_step(X_train, Y_ann_train, predictions_z, predictions_g)
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL(r_train,predictions))
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])                    
                if relative:
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                tol3 = np.mean(np.abs(self.alphas-old_alphas)/(old_alphas+self.Keps)) #alphas
                print("Tol1: %.5f\tTol2: %.5f\tTol3: %.5f\t"%(tol,tol2,tol3),end='',flush=True)
            old_betas = self.betas.flatten().copy()         
            old_alphas = self.alphas.copy()
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance): #alphas fuera: and tol3<=tolerance
                stop_c = True 
        print("Finished training!")
        return np.asarray(logL)
    
    def annotator_2_group(self, annotator):
        """
            Map some representation for the annotator to a group p(g|a)
        """
        if len(annotator.shape) ==1:
            annotator = [annotator]
        return self.get_predictions_groups(annotator)
    
    def stable_train(self,X,Y_ann,A,batch_size=64,max_iter=50,tolerance=3e-2,cluster=True,bulk_annotators=[]):
        """
            A stable schedule to train a model on this formulation
        """
        if not self.priors:
            self.define_priors('laplace') #needed..
        
        if cluster: # do annotator clustering
            #if len(bulk_annotators) == 0:
            alphas_clusterized = clusterize_annotators(r,M=self.M,bulk=False,cluster_type='mv_close',data=X,model=self.base_model,DTYPE_OP=self.DTYPE_OP,BATCH_SIZE=batch_size) #clusteriza en base aloss
            #for REPEAT MODEL
            #elif len(bulk_annotators) == 1:
            #    alphas_clusterized = clusterize_annotators(bulk_annotators[0],M=self.M,no_label=-1)
            #else:
            #    alphas_clusterized = clusterize_annotators(bulk_annotators[0],M=self.M,no_label=-1,data=bulk_annotators[1])
            self.set_alpha(alphas_clusterized)

        logL_hist = self.train(X,Y_ann,A, batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,relative=True,val=False)
        return logL_hist
    
    def multiples_run(self,Runs,X,Y_ann,A,batch_size=64,max_iter=50,tolerance=3e-2,cluster=True,bulk_annotators=[]): 
        """
            Run multiples max_iter of EM algorithm, with random stars
        """
        if Runs==1:
            return self.stable_train(X,Y_ann,A,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,cluster=True,bulk_annotators=bulk_annotators), 0

        #maybe lamda random here
        if not self.priors:
            self.define_priors('laplace') #needed!
        
        if cluster: # do annotator clustering
            alphas_clusterized = clusterize_annotators(r,M=self.M,bulk=False,cluster_type='mv_close',data=X,model=self.base_model,DTYPE_OP=self.DTYPE_OP,BATCH_SIZE=batch_size) #clusteriza en base aloss -- mv_close
            self.set_alpha(alphas_clusterized)
            del alphas_clusterized
            
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
            
            del self.base_model, group_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        if type(clonable_model_z.layers[0]) == keras.layers.clonable_model_z:
            self.base_model = keras.models.clone_model(clonable_model_z) #change
            self.group_model = keras.models.clone_model(clonable_model_g)
        else:
            it = keras.layers.Input(shape=X.shape[1:])
            self.base_model = keras.models.clone_model(clonable_model_z, input_tensors=it) #change
            it = keras.layers.Input(shape=A.shape[1:])
            self.group_model = keras.models.clone_model(clonable_model_g, input_tensors=it) #change
        self.base_model.set_weights(found_model_z[indexs_sort[0]])
        self.group_model.set_weights(found_model_g[indexs_sort[0]])
        self.E_step(X, Y_ann, self.get_predictions_base(X), self.get_predictions_group(A)) #to set up Q
        print("Multiples runs over Ours Global, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]
    
    def get_predictions_group(self,m,X):
        """ Predictions of group "m", p(y^o | xi, g=m) """
        p_z = self.get_predictions(X)
        p_y_m = np.zeros(p_z.shape)
        for i in range(self.N):
            p_y_m[i] = np.tensordot(p_z[i,:] ,self.betas[m,:,:],axes=[[0],[0]] ) # sum_z p(z|xi) * p(yo|z,g=m)
        return p_y_m 
    
    def get_predictions_groups(self,X,data=[]):
        """ Predictions of all groups , p(y^o | xi, g) """
        if len(data) != 0:
            p_z = data
        else:
            p_z = self.get_predictions(X)
        predictions_m = np.tensordot(p_z ,self.betas,axes=[[1],[1]] ) #sum_z p(z|xi) * p(yo|z,g)
        return predictions_m#.transpose(1,0,2)

    def calculate_extra_components(self,X, A, T=0, calculate_pred_annotator=True,p_z=[]):
        """
            Measure indirect probabilities through bayes and total probability of annotators
        """
        predictions_m = self.get_predictions_groups(X,data=p_z) #p(y^o|x,g=m)
        
        #prob_Gt = np.zeros((T,self.M)) #p(g|t)
        #for t in range(t):
        #    prob_Gt[t] = self.annotator_2_group(A[t],pred=predictions_m) 
        prob_Gt = self.get_predictions_group(A)

        prob_Yzt = np.tensordot(prob_Gt, self.get_confusionM(),axes=[[1],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)
  
        if calculate_pred_annotator:
            prob_Yxt = np.tensordot(predictions_m, prob_Gt, axes=[[1],[1]]).transpose(0,2,1) #p(y^o|x,t) = sum_g p(g|t) *p(yo|x,g)
        else:
            prob_Yxt = None
        gc.collect()
        return predictions_m, prob_Gt, prob_Yzt, prob_Yxt
    
    def calculate_Yz(self,prob_Gt):
        """ Calculate global confusion matrix"""
        alphas = np.mean(prob_Gt, axis=0)
        return np.sum(self.betas*alphas[:,None,None],axis=0)
    
    def get_annotator_reliability(self,X,a):
        """Get annotator reliability, based on his representation:"""        
        prob_Gt = annotator_2_group(self,a)
        prob_Yzt = np.tensordot(prob_Gt, self.get_confusionM(),axes=[[0],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)
        return prob_Yzt #do something with it
