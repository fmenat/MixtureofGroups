import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import KernelPCA, PCA
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

def aux_clusterize_annotators(data_to_cluster,M,DTYPE_OP='float32',option="fuzzy",l=0.005):
    """ Get p(g=m|t)  based on a proyection of the annotator "t"  """
    std = StandardScaler()
    data_to_cluster = std.fit_transform(data_to_cluster) #sacar
        
    kmeans = MiniBatchKMeans(n_clusters=M, random_state=0,init='k-means++',batch_size=128)
    #KMeans(M,init='k-means++', n_jobs=-1,random_state=0)
    kmeans.fit(data_to_cluster)
    distances = kmeans.transform(data_to_cluster)

    if option=="fuzzy":
        probas_t = np.zeros_like(distances,dtype=DTYPE_OP)
        for t in range(probas_t.shape[0]):
            for m in range(probas_t.shape[1]):
                m_fuzzy = 1.1
                probas_t[t,m] = 1/(np.sum( np.power((distances[t,m]/(distances[t,:]+keras.backend.epsilon())), 2/(m_fuzzy-1)) ) + keras.backend.epsilon())
    elif option == "softmax":
        probas_t = softmax(-(distances+keras.backend.epsilon())/l).astype(DTYPE_OP)
    elif option == "softmax inv":
        probas_t = softmax(1/(l*distances+keras.backend.epsilon())).astype(DTYPE_OP)
    elif option == 'hard':
        probas_t = keras.utils.to_categorical(kmeans.labels_)
    #another option??
        #model = GaussianMixture(n_components=M)
        #model.fit(data_to_cluster)
        #probas_t = model.predict_proba(data_to_cluster).astype(DTYPE_OP)
    return probas_t
            
def clusterize_annotators(y_o,M,no_label=-1,bulk=True,cluster_type='loss',data=[],model=None,DTYPE_OP='float32',BATCH_SIZE=64,option="fuzzy",l=0.005):
    if bulk: 
        if len(y_o.shape) == 2:
            M_itj = categorical_representation(y_o,no_label =no_label)
        else:
            M_itj = y_o.copy()
        mask_nan = M_itj.sum(axis=1,keepdims=True) == 0
        mask_nan = np.tile(mask_nan,(1,M_itj.shape[1],1))
        M_itj[mask_nan] = 1
        M_itj_norm = M_itj/np.sum(M_itj,axis=1)[:,None,:]
        M_itj_norm = M_itj_norm.astype(DTYPE_OP)
        
        if len(data) != 0:
            data_to_cluster = data.copy() #annotators_pca
        else:
            data_to_cluster = M_itj.transpose(1,0,2).reshape(M_itj.shape[1],M_itj.shape[0]*M_itj.shape[2])
            
        probas_t = aux_clusterize_annotators(data_to_cluster,M,DTYPE_OP,option,l) #0.05 is close to one-hot
        print("Clustering Done!")
        alphas_init = np.tensordot(M_itj_norm,probas_t, axes=[[1],[0]]) 
    else: #sirve como auxiliar: y_o: is repeats
        if len(y_o.shape) == 2: 
            mv_hard = majority_voting(y_o,repeats=True,probas=False) 
        else:
            mv_hard = majority_voting(y_o,repeats=False,probas=False) 
        aux_model = keras.models.clone_model(model)
        aux_model.compile(loss='categorical_crossentropy',optimizer=model.optimizer)
        aux_model.fit(data, mv_hard, batch_size=BATCH_SIZE,epochs=30,verbose=0)
        predicted = aux_model.predict(data)
        data_to_cluster = []
        if cluster_type=='loss': #cluster respecto to loss function
            for i in range(mv_hard.shape[0]):
                for j in range(mv_hard.shape[1]):
                    ob = np.tile(keras.backend.epsilon(), mv_hard.shape[1])
                    ob[j] = 1
                    true = np.clip(predicted[i],keras.backend.epsilon(),1.)        
                    loss = np.sum(true*np.log(ob)) 
                    data_to_cluster.append([loss])
        data_to_cluster = np.asarray(data_to_cluster)
        probas_t = aux_clusterize_annotators(data_to_cluster,M,DTYPE_OP,option,l)
        print("Clustering Done!")
        alphas_init = probas_t.reshape(mv_hard.shape[0],mv_hard.shape[1],M)
    return alphas_init

def project_and_cluster(y_o,M_to_try=20,anothers_visions=True,DTYPE_OP='float32',printed=True):
    ###another way to cluster..
    if len(y_o.shape) == 2:
        M_itj = categorical_representation(y_o,no_label =-1)
    else:
        M_itj = y_o.copy()
    K = np.max(y_o)+1    
    data_to_cluster = M_itj.transpose(1,0,2).reshape(M_itj.shape[1],M_itj.shape[0]*M_itj.shape[2])

    #kpca_model = KernelPCA(n_components=4, kernel='rbf', n_jobs=-1) #componentes a proyectar?
    kpca_model = PCA(n_components=4)
    plot_data = kpca_model.fit_transform(data_to_cluster).astype(DTYPE_OP)
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


class GroupMixtureOpt(object): #optimized version
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
        """Get confusion matrices of every group p(yo|g,z)"""  
        return self.betas.copy()
    def get_alpha(self):
        """Get alpha param, p(g) globally"""
        return self.alphas.copy()
    def set_alpha(self,alpha_init):
        """set alpha param with a previosuly method"""
        self.alpha_init = alpha_init.copy()
        self.seted_alphainit = True
    def get_qestimation(self):
        """Get Q estimation param, this is Q_ij(g,z) = p(g,z|xi,y=j)"""
        return self.Qij_mgamma.copy()
        
    def define_model(self,tipo,start_units=1,deep=1,double=False,drop=0.0,embed=True,BatchN=True):
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
            self.base_model = CNN_simple(self.input_dim,self.Kl,start_units,deep,double=double,BN=BatchN,drop=drop)
        
        elif self.type=='simplernn' or self.type=='simple rnn' or 'rnn' in self.type:
            print("Needed params (units,deep,drop,embed?)")
            self.base_model = RNN_simple(self.input_dim,self.Kl,start_units,deep,drop=drop,embed=embed,len=0,out=start_units*2)
            #and what is with embedd

        #if not (self.type == "keras_shallow" or self.type=="keras_perceptron"): 
        #    self.base_model = create_network(self.Kl,self.input_dim,tipo,info,infoextractor_network=aux_info,embedding_info=emb_info)
        #     future..
        self.base_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 
        self.compile = True
        
    def get_predictions(self,X):
        """Return the predictions of the model if is from sklearn or keras"""
        if "sklearn" in self.type:
            return self.base_model.predict_proba(X) 
        else:
            return self.base_model.predict(X,batch_size=self.batch_size)
    
    def init_E(self,X,r):
        """Realize the initialziation of the E step on the EM algorithm"""
        #-------> init alpha
        if not self.seted_alphainit: #random is the worst option
            self.alpha_init = np.random.dirichlet(np.ones(self.M)/50,size=(self.N,self.Kl))
        
        #-------> init Majority voting        
        self.mv_probs_j = majority_voting(r,repeats=True,probas=True) # soft -- p(y=j|xi)
        
        print("Pre-train network on %d epochs..."%(self.pre_init),end='',flush=True)
        mv_probs = keras.utils.to_categorical(self.mv_probs_j.argmax(axis=1)) # one-hot
        self.base_model.fit(X,mv_probs,batch_size=self.batch_size,epochs=self.pre_init,verbose=0)
        print(" Done!")
        mv_probs = self.base_model.predict(X,verbose=0) 
        #reset optimizer but hold weights--necessary for stability 
        self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

        #-------> Initialize p(z=gamma|xi,y=j,g): Combination of mv and belive observable
        lambda_group = np.ones((self.M),dtype=self.DTYPE_OP) 
        if self.lambda_random:
            for m in range(self.M):
                lambda_group[m] = np.random.beta(1,1)
        print("Lambda by group: ",lambda_group)
        Zijm = np.zeros((self.N,self.Kl,self.M,self.Kl),dtype=self.DTYPE_OP)
        for j_ob in range(self.Kl):
            onehot = np.tile(self.Keps, self.Kl)
            onehot[j_ob] = 1. #all belive in the observable
            for m in range(self.M):                
                Zijm[:,j_ob,m,:] = lambda_group[m]*mv_probs + (1-lambda_group[m])*onehot 
          
        #-------> init q_ij      
        self.Qij_mgamma = self.alpha_init[:,:,:,None]*Zijm
        
        #-------> init betas
        self.betas = np.zeros((self.M,self.Kl,self.Kl),dtype=self.DTYPE_OP) 

        #-------> init alphas
        self.alphas = np.zeros((self.M),dtype=self.DTYPE_OP)
     
        print("Alphas: ",self.alphas.shape)
        print("MV init: ",mv_probs.shape)
        print("Betas: ",self.betas.shape)
        print("Q estimate: ",self.Qij_mgamma.shape)
            
    def define_priors(self,priors):
        """
            Priors with shape: (M,K,K), need counts for every group and every pair (k,k) ir global (M,K)
            The group m, given a class "k" is probably that say some class
            it is recomended that has full of ones
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
        
    def E_step(self,X,predictions):
        """ Realize the E step in matrix version"""       
        p_new = np.clip(predictions[:,None,None,:] , self.Keps,1.) #safe logarithmn
        a_new = np.clip(self.alphas[None,None,:,None] , self.Keps,1.) #safe logarithmn
        b_new = np.clip((self.betas[None,:,:,:]).transpose(0,3,1,2) , self.Keps,1.) #safe logarithmn
        
        self.Qij_mgamma = np.exp(np.log(p_new) + np.log(a_new) + np.log(b_new)) 
        self.aux_for_like = np.sum(np.sum(self.Qij_mgamma,axis=-1),axis=-1) #p(y=j|x) --marginalized
        self.Qij_mgamma = self.Qij_mgamma/self.aux_for_like[:,:,None,None] #normalize
        
    def M_step(self,X,r): 
        """ Realize the M step"""
        #-------> base model
        Qij_gamma = np.sum(self.Qij_mgamma,axis=-2) #qij(gamma)
        r_estimate = np.zeros((self.N,self.Kl),dtype=self.DTYPE_OP) #create the repeat "estimate"/"ground truth"
        for i in range(self.N):
            r_estimate[i] = np.tensordot(Qij_gamma[i],r[i],axes=[[0],[0]])
        if "sklearn" in self.type:#train to learn p(z|x)
            self.base_model.fit(X, np.argmax(r_estimate,axis=1) ) 
        else:
            self.base_model.fit(X,r_estimate,batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
    
        #-------> alpha 
        Qij_m = np.sum(self.Qij_mgamma,axis=-1) #qij(m)
        self.alphas = np.tensordot(Qij_m, r , axes=[[0,1],[0,1]]) # sum_ij r_ij(g) = Qij_m[i]*r[i] 
        self.alphas = self.alphas.astype(self.DTYPE_OP) #necessary
        self.alphas = self.alphas/np.sum(self.alphas,axis=-1) #p(g) -- normalize
        
        #-------> beta
        for j_ob in range(self.Kl):
            self.betas[:,:,j_ob] = np.tensordot(self.Qij_mgamma[:,j_ob,:,:],r[:,j_ob], axes=[[0],[0]]) # ~p(yo=j|g,z)              
        if self.priors:
            self.betas += self.Mpriors #priors has to be shape: (M,Kl,Kl)--read define-prior functio
        self.betas = self.betas/np.sum(self.betas,axis=-1)[:,:,None] #normalize (=p(yo|g,z))

    def compute_logL(self,r,predictions):
        """ Compute the log-likelihood of the optimization schedule"""
        self.aux_for_like = np.clip(self.aux_for_like, self.Keps, 1.) #safe logarithm
        return np.tensordot(r , np.log(self.aux_for_like))+0.
                                                  
    def train(self,X_train,r_train,batch_size=64,max_iter=500,relative=True,val=False,tolerance=1e-2):
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        print("Initializing new EM...")
        self.batch_size = batch_size
        self.N = X_train.shape[0]
        self.init_E(X_train,r_train)
        
        logL = []
        stop_c = False
        tol,old_model,old_betas,old_alphas = np.inf,np.inf,np.inf,np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train,r_train)
            print(" done,  E step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x) 
            self.E_step(X_train,predictions)
            print(" done //  (in %.2f sec)\t"%(time.time()-start_time),end='',flush=True)
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
            #if val:
            #    print("F1: %.4f"%(f1_score(Z_train, predictions.argmax(axis=1),average='micro')),end='',flush=True)
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance and tol3<=tolerance):
                stop_c = True 
        print("Finished training!")
        gc.collect()
        return np.asarray(logL)
    
    def annotations_2_group(self,annotations,data=[],no_label_sym = -1):
        """
            Map some annotations to some group model by the confusion matrices, p(g| {x_l,y_l})
        """
        if data.shape[0] == self.M: #if prediction_m is passed
            predictions_m = data
        elif data.shape[0] != self.M: #if data is passed
            X = data
            predictions_m = self.get_predictions_groups(X)
            
        result = np.log(self.get_alpha()+self.Keps)
        aux_annotations = [(i,annotation) for i, annotation in enumerate(annotations) if annotation != no_label_sym]
        for i, annotation in aux_annotations:
            if annotation != no_label_sym: #if label it
                for m in range(self.M):
                    result[m] += np.log(predictions_m[m,i,annotation]+self.Keps)
        result = np.exp(result - result.max(axis=-1, keepdims=True) ) #invert logarithm in safe way
        return result/np.sum(result)
    
    def stable_train(self,X,r,batch_size=64,max_iter=50,tolerance=1e-2,cluster=True,bulk_annotators=[]):
        """
            A stable schedule to train a model on this formulation
        """
        self.lambda_random = False #lambda=1
        self.define_priors('laplace')
        
        if cluster: # do annotator clustering
            if len(bulk_annotators) == 0:
                alphas_clusterized = clusterize_annotators(r,M=self.M,bulk=False,cluster_type='loss',data=X,model=self.base_model) #clusteriza en base aloss
            else:
                alphas_clusterized = clusterize_annotators(bulk_annotators[0],M=self.M,no_label=-1,data=bulk_annotators[1])
            self.set_alpha(alphas_clusterized)

        logL_hist = self.train(X,r,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,relative=True,val=False)
        return logL_hist
    
    #and multiples runs with lambda random false?
    def multiples_run(self,Runs,X,r,batch_size=64,max_iter=50,tolerance=1e-2,cluster=True,bulk_annotators=[]): 
        """
            Run multiples max_iter of EM algorithm, with random stars
        """
        self.define_priors('laplace')
        
        if cluster: # do annotator clustering
            if len(bulk_annotators) == 0:
                alphas_clusterized = clusterize_annotators(r,M=self.M,bulk=False,cluster_type='loss',data=X,model=self.base_model,DTYPE_OP=self.DTYPE_OP,BATCH_SIZE=batch_size) #clusteriza en base aloss
            else:
                alphas_clusterized = clusterize_annotators(bulk_annotators[0],M=self.M,no_label=-1,data=bulk_annotators[1],DTYPE_OP=self.DTYPE_OP)
            self.set_alpha(alphas_clusterized)
            
        found_betas = []
        found_alphas = []
        found_model = []
        found_logL = []
        for run in range(Runs):
            self.base_model = keras.models.clone_model(self.base_model) #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,r,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,relative=True) #here the models get resets
            
            found_betas.append(self.betas.copy())
            found_alphas.append(self.alphas.copy())
            found_model.append(self.base_model) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        self.alphas = found_alphas[indexs_sort[0]].copy()
        self.base_model = found_model[indexs_sort[0]]
        self.E_step(X,self.get_predictions(X)) #to set up Q
        return found_logL,indexs_sort[0]
    
    def get_predictions_group(self,m,X):
        """ Predictions of group "m", p(y^o | xi, g=m) """
        p_z = self.get_predictions(X)
        p_y_m = np.zeros(p_z.shape)
        for i in range(self.N):
            p_y_m[i] = np.tensordot(p_z[i,:] ,self.betas[m,:,:],axes=[[0],[0]] ) # sum_z p(z|xi) * p(yo|z,g=m)
        return p_y_m 
    
    def get_predictions_groups(self,X):
        """ Predictions of all groups , p(y^o | xi, g) """
        p_z = self.get_predictions(X)
        predictions_m = np.tensordot(p_z ,self.betas,axes=[[1],[1]] ) #sum_z p(z|xi) * p(yo|z,g)
        return predictions_m.transpose(1,0,2)

    def calculate_extra_components(self,X,y_o,T,calculate_pred_annotator=True):
        """
            Measure indirect probabilities through bayes and total probability of annotators
        """
        predictions_m = self.get_predictions_groups(X) #p(y^o|x,g=m)
        
        prob_Gt = np.zeros((T,self.M)) #p(g|t)
        for t in range(T):
            prob_Gt[t] = self.annotations_2_group(y_o[:,t],predictions_m) 

        prob_Yzt = np.tensordot(prob_Gt, self.get_confusionM(),axes=[[1],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)
  
        if calculate_pred_annotator:
            prob_Yxt = np.tensordot(prob_Gt, predictions_m,axes=[[1],[0]]) #p(y^o|x,t) = sum_g p(g|t) *p(yo|x,g)
        else:
            prob_Yxt = None
        gc.collect()
        return predictions_m, prob_Gt, prob_Yzt, prob_Yxt
    
    def get_annotator_reliability(self,y_o,X,t):
        """Get annotator reliability, based on his annotations:"""        
        prob_Gt = annotations_2_group(self,y_o[:,t],data=X)
        
        prob_Yzt = np.tensordot(prob_Gt, self.get_confusionM(),axes=[[0],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)
        return prob_Yzt #do something with it
