"""
	*Old model by fors!-- more efficient in RAM terms
"""
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
class GroupMixture(object):
    def __init__(self,input_dim,Kl=2,M=2,epochs=1,optimizer='adam',pre_init=30): 
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.input_dim = input_dim
        self.Kl = Kl #number of classes of the problem
        self.Keps = keras.backend.epsilon()
        self.priors=False #boolean of priors
        #params
        self.M = M #groups of annotators
        self.epochs = epochs
        self.optimizer = optimizer
        self.compile=False
        self.pre_init = pre_init

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
        #f self.data_dependence:#self.alphas = np.tile(1./self.M,(self.N,self.M))  #uniform start p(g|x)=1/M
        if not self.seted_alphainit: #random is the worst option
            self.alpha_init = np.random.dirichlet(np.ones(self.M)/50,size=(self.N,self.Kl))
        
        #-------> init Majority voting        
        self.mv_probs_j = majority_voting(r,repeats=True,probas=True) # soft -- p(y=j|xi)
        
        print("Pre-train network on %d epochs..."%(self.pre_init),end='',flush=True) #5 with complex
        mv_probs = keras.utils.to_categorical(np.argmax(self.mv_probs_j,axis=1)) # one-hot
        self.base_model.fit(X,mv_probs,batch_size=self.batch_size,epochs=self.pre_init,verbose=0)
        print(" Done!")
        #if trained for a few epochs use soft-MV and this: 
        #mv_probs = mv_probs_j.copy()
        #else
        mv_probs = self.base_model.predict(X,verbose=0) 
        #reset optimizer--not necessary
        #self.base_model .compile(loss='categorical_crossentropy',optimizer=self.optimizer)

        #-------> Initialize p(z=gamma|xi,y=j,g): Combination of mv and belive observable
        if self.lambda_random:
            lambda_group = [np.random.beta(1,1) for m in range(self.M)]
        else:
            lambda_group = np.ones((self.M)) 
        print("Lambda by group: ",lambda_group)
        Zijm = np.zeros((self.N,self.Kl,self.M,self.Kl))
        for j_ob in range(self.Kl):
            onehot = np.zeros(self.Kl)
            onehot[j_ob] = 1 #all belive in the observable
            for m in range(self.M):                
                Zijm[:,j_ob,m,:] = lambda_group[m]*mv_probs + (1-lambda_group[m])*onehot 
          
        #-------> init q_ij
        self.Qij_mgamma = np.zeros((self.N,self.Kl,self.M,self.Kl)) 
        for i in range(self.N):
            for j_ob in range(self.Kl):
                for m in range(self.M):
                    self.Qij_mgamma[i,j_ob,m,:] = self.alpha_init[i,j_ob,m]*Zijm[i,j_ob,m]
        
        #-------> init betas
        self.betas = np.zeros((self.M,self.Kl,self.Kl)) 
        
        #-------> init alphas
        self.alphas = np.zeros((self.M))
     
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
        if len(priors.shape)==2:
            priors=np.expand_dims(priors,axis=2)
        self.Mpriors = priors
        self.priors = True
        
    def E_step(self,X,predictions):
        """ Realize the E step"""       
        self.aux_for_like = np.zeros((self.N,self.Kl)) #sum over unnormalized  Q
        for i in range(self.N):
            for j_ob in range(self.Kl):
                for m in range(self.M):
                    self.Qij_mgamma[i,j_ob,m,:] = np.log(predictions[i,:]+self.Keps)+np.log(self.alphas[m]+self.Keps)+np.log(self.betas[m,:,j_ob]+self.Keps)
                self.Qij_mgamma[i,j_ob] = np.exp(self.Qij_mgamma[i,j_ob])
                self.aux_for_like[i,j_ob] = np.sum(np.sum(self.Qij_mgamma[i,j_ob],axis=-1),axis=-1) #p(y=j|x) --marginalized
                self.Qij_mgamma[i,j_ob] = self.Qij_mgamma[i,j_ob]/self.aux_for_like[i,j_ob] #normalize q
        
    def M_step(self,X,r): 
        """ Realize the M step"""
        #-------> base model
        #create the repeat "estimate"/"ground truth"
        Qij_gamma = np.sum(self.Qij_mgamma,axis=-2) #qij(gamma)
        r_estimate = np.zeros((self.N,self.Kl)) #repeat estimate
        for i in range(self.N):
            r_estimate[i] = np.tensordot(Qij_gamma[i],r[i],axes=[[0],[0]])
        if "sklearn" in self.type:#train to learn p(z|x)
            self.base_model.fit(X, np.argmax(r_estimate,axis=1) ) 
        else:
            history = self.base_model.fit(X,r_estimate,batch_size=self.batch_size,epochs=self.epochs,verbose=0) 

        #-------> alpha --maneja caso global y normal..
        Qij_m = np.sum(self.Qij_mgamma,axis=-1) #qij(m)
        g_estimate = np.zeros((self.N,self.M))
        for i in range(self.N):
            g_estimate[i] = np.tensordot(Qij_m[i],r[i],axes=[[0],[0]]) #Qij_m[i]*r[i] = r_ij(g)
        g_estimate = np.sum(g_estimate,axis=0) 
        self.alphas = g_estimate/np.sum(g_estimate,axis=-1) #p(g) 
        
        #-------> beta
        for m in range(self.M):
            for j_z in range(self.Kl):
                for j_ob in range(self.Kl):
                    self.betas[m,j_z,j_ob] = np.tensordot(self.Qij_mgamma[:,j_ob,m,j_z],r[:,j_ob], axes=[[0],[0]]) # ~p(yo|g,z)
        if self.priors:
            self.betas += self.Mpriors #priors has to be shape: (M,Kl,Kl)--read define-prior function
        self.betas = self.betas/np.sum(self.betas,axis=-1)[:,:,None] #normalize (=p(yo|g,z))

    def compute_logL(self,r,predictions):
        """ Compute the log-likelihood of the optimization schedule"""
        return np.tensordot(r , np.log(self.aux_for_like+self.Keps))+0
                                                  
    def train(self,X_train,r_train,batch_size=32,iterations=250,relative=False,val=False,tolerance=0): #0.0001  
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        print("Initializing new EM...")
        self.batch_size = batch_size
        self.N = X_train.shape[0]
        self.init_E(X_train,r_train)
        
        logL = []
        tol = 100
        self.current_iter = 1
        while(self.current_iter <= iterations and tol>tolerance):
            print("Iter %d/%d\nM step:"%(self.current_iter,iterations),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train,r_train)
            print(" done\tE step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x) 
            self.E_step(X_train,predictions)
            print(" done, (in %.2f sec)\t"%(time.time()-start_time),end='',flush=True)
            logL.append(self.compute_logL(r_train,predictions))
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])                    
                if relative:
                    tol = tol/np.abs(logL[-2])
                print("Tol: %.5f\t"%(tol),end='',flush=True)
            if val:
                print("F1: %.4f"%(calculate_f1_keras(self.base_model, X_train, Z_train)),end='',flush=True)
            self.current_iter+=1
            print("")
        print("Finished training!")
        return np.asarray(logL)
    
    def annotations_2_group(self,annotations,data=[]):
        """
            Map some annotations to some group model by the confusion matrices, p(g| {x_l,y_l})
        """
        no_label_sym = -1 #no label symbol
        if data.shape[0] == self.M: #if prediction_m is passed
            predictions_m = data
        elif data.shape[0] != self.M: #if data is passed
            X = data
            predictions_m = np.zeros((self.M,self.N,self.Kl))
            for m in range(self.M):
                predictions_m[m] = self.get_predictions_group(m,X) 
            
        result = np.log(self.get_alpha()+self.Keps)
        aux_annotations = [(i,annotation) for i, annotation in enumerate(annotations) if annotation != no_label_sym]
        np.random.shuffle(aux_annotations)
        for i, annotation in aux_annotations:
            if annotation != no_label_sym and np.min(result) > -730: #if label it and avoid zero values
                #only a subset of annotations randomly..
                for m in range(self.M):
                    result[m] += np.log(predictions_m[m,i,annotation]+self.Keps)
        result = np.exp(result)#invert logarithm
        return result/np.sum(result)
    
    def get_predictions_group(self,m,X):
        """ Predictions of group "m", p(y^o | xi, g=m) """
        p_z = self.get_predictions(X)
        p_y_m = np.zeros(p_z.shape)
        for i in range(self.N):
            p_y_m[i] = np.tensordot(p_z[i,:] ,self.betas[m,:,:],axes=[[0],[0]] )
        return p_y_m 
    
    def get_predictions_groups(self,X):
        """ Predictions of all groups , p(y^o | xi, g) """
        predictions_m = np.zeros((self.M,self.N,self.Kl))
        for m in range(self.M):
            predictions_m[m] = gMixture.get_predictions_group(m,X) 
        return predictions_m

    def stable_train(self,X,r,cluster=True,bulk_annotators=[]):
        """
            A stable schedule to train a model on this formulation
        """
        self.lambda_random = False #lambda=1
        self.define_priors(np.ones((self.M,self.Kl)))
        
        if cluster: # do annotator clustering
            if len(bulk_annotators) == 0:
                alphas_clusterized = clusterize_annotators(r,M=self.M,bulk=False,cluster_type='loss',data=X,model=self.base_model) #clusteriza en base aloss
            else:
                alphas_clusterized = clusterize_annotators(bulk_annotators[0],M=self.M,no_label=-1,data=bulk_annotators[1])
            self.set_alpha(alphas_clusterized)

        logL_hist = self.train(X,r,batch_size=BATCH_SIZE,iterations=50,relative=True,val=True) #maybe more iterations
        return logL_hist
    
    def multiples_run(self,Runs,X,r,cluster=True,bulk_annotators=[]): 
        """
            Run multiples iterations of EM algorithm, with random stars
        """
        #with data dependent set outside
        self.lambda_random = True     
        self.define_priors(np.ones((self.M,self.Kl))) #need priors!!!
        
        if cluster: # do annotator clustering
            if len(bulk_annotators) == 0:
                alphas_clusterized = clusterize_annotators(r,M=self.M,bulk=False,cluster_type='loss',data=X,model=self.base_model) #clusteriza en base aloss
            else:
                alphas_clusterized = clusterize_annotators(bulk_annotators[0],M=self.M,no_label=-1,data=bulk_annotators[1])
            self.set_alpha(alphas_clusterized)
            
        found_betas = []
        found_alphas = []
        found_model = []
        found_logL = []
        for run in range(Runs):
            self.base_model = clone_model(self.base_model) #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,r,batch_size=BATCH_SIZE,iterations=50,relative=True) #here the models get resets
            
            found_betas.append(self.betas.copy())
            found_alphas.append(self.alphas.copy())
            found_model.append(self.base_model) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        self.alphas = found_alphas[indexs_sort[0]].copy()
        self.base_model = found_model[indexs_sort[0]]
        self.E_step(X,self.get_predictions(X)) #to set up Q
        return found_logL
    
    def calculate_extra_components(self,X,y_o,T,calculate_pred_annotator=True):
        """
            Measure indirect probabilities through bayes and total probability of annotators
        """
        predictions_m = np.zeros((self.M,self.N,self.Kl)) 
        for m in range(self.M):
            predictions_m[m] = self.get_predictions_group(m,X) #p(y^o|x,g=m)
        
        prob_Gt = [] #p(g|t)
        for t in range(T):
            prob_Gt.append( self.annotations_2_group(y_o[:,t],predictions_m) )
        prob_Gt = np.asarray(prob_Gt)

        prob_Yzt = np.zeros((T,self.Kl,self.Kl)) #p(y^o|z,t)
        for t in range(T):
            for z in range(self.Kl):
                prob_Yzt[t,z] = np.tensordot(prob_Gt[t], self.get_confusionM()[:,z,:],axes=[[0],[0]] )
        
        if calculate_pred_annotator:
            prob_Yxt = np.zeros((T,self.N,self.Kl))
            for t in range(T):
                for i in range(self.N):
                    prob_Yxt[t,i] = np.tensordot(prob_Gt[t], predictions_m[:,i,:],axes=[[0],[0]] )
        else:
            prob_Yxt = None
        return predictions_m, prob_Gt, prob_Yzt, prob_Yxt
