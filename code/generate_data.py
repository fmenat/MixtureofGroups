import numpy as np
import pickle

class SinteticData(object):
    def __init__(self,state=None):
        self.probas = False #not yet
        self.Random_num = np.random.RandomState(None)
        if type(state) ==str:
            with open(state, 'rb') as handle:
                aux = pickle.load(handle)  #read from file
                self.Random_num.set_state(aux)
        elif type(state) == tuple or type(state) == int:
            self.Random_num.set_state(state)
            
        #init state:
        self.init_state = self.Random_num.get_state() #to replicate

    def set_probas(self,asfile = True,file_matrix = "matrix.csv",file_groups='groups.csv'):
        """
            * conf_matrix: All confusion matrices of the differentes groups
            * prob_groups: probabilities of the groups in the data
        """
        if asfile:
            load_matrix = np.loadtxt(file_matrix,delimiter=',')
            rows,Kl = load_matrix.shape
            self.conf_matrix = []
            for j in np.arange(Kl,load_matrix.shape[0]+1,Kl):
                self.conf_matrix.append(load_matrix[j-Kl:j])
            self.conf_matrix = np.asarray(self.conf_matrix)

            self.prob_groups = np.loadtxt(file_groups,delimiter=',')
        else:
            #se entrega el archivo directamente
            self.conf_matrix = np.asarray(file_matrix) 
            self.prob_groups = np.asarray(file_groups)
        self.probas = True


    def sintetic_annotate_data(self,Y,Tmax,T_data,deterministic,hard=True):
        """ARGS:
            * N: number of data
            * Tmax: number of annotators in all the data
            * T_data: Expected value of number of annotators by every data      
        """
        if not self.probas:
            self.set_probas()

        N = Y.shape[0]
        #sample group for every annotator:
        sintetic_annotators_group = []
        for t in range(Tmax):
            if hard:
                S_t = 1
            else: #soft
                S_t = max([self.Random_num.poisson(self.prob_groups.shape[0]+1),1]) 

            grupo = self.Random_num.multinomial(S_t,self.prob_groups)
            
            if hard:
                grupo = [np.argmax(grupo)]
            else:
                grupo = grupo/np.sum(grupo) #soft
            sintetic_annotators_group.append(grupo)
        sintetic_annotators_group = np.asarray(sintetic_annotators_group)
        
        sintetic_annotators = -1*np.ones((N,Tmax),dtype='int32')

        prob = T_data/float(Tmax) #probability that she annotates
        for i in range(N):
            #get ground truth of data 
            if Y[i].shape != ():
                z = int(np.argmax(Y[i]))
            else:
                z = int(Y[i])

            if deterministic:
                Ti = self.Random_num.choice(np.arange(Tmax), size=T_data, replace=False) #multinomial of index
                for t in Ti: #index of annotators
                    #get group of annotators
                    g = sintetic_annotators_group[t] #in discrete value, g {0,1,...,M}
                    if hard:
                        sample_prob = self.conf_matrix[g[0],z,:]
                    else: #soft
                        sample_prob = np.tensordot(g[:], self.conf_matrix[:,z,:], axes=[[0],[0]]) #mixture
                    #sample trough confusion matrix 
                    yo = np.argmax( self.Random_num.multinomial(1, sample_prob) )
                    sintetic_annotators[i,t] = yo
            else:
                for t in range(Tmax):
                    if self.Random_num.rand() <= prob: #if she label the data i
                        #get group of annotators
                        g = sintetic_annotators_group[t] #in discrete value, g {0,1,...,M}
                        if hard:
                            sample_prob = self.conf_matrix[g[0],z,:]
                        else: #soft
                            sample_prob = np.tensordot(g[:], self.conf_matrix[:,z,:], axes=[[0],[0]]) #mixture
                        #sample trough confusion matrix 
                        yo = np.argmax( self.Random_num.multinomial(1, sample_prob) )
                        sintetic_annotators[i,t] = yo
                        
            if np.sum( sintetic_annotators[i,:] != -1)  == 0: #avoid data not labeled
                t_rand = self.Random_num.randint(0,Tmax)
                g = sintetic_annotators_group[t_rand] #in discrete value, g {0,1,...,M}
                if hard:
                    sample_prob = self.conf_matrix[g[0],z,:]
                else: #soft
                    sample_prob = np.tensordot(g[:], self.conf_matrix[:,z,:], axes=[[0],[0]]) #mixture

                sintetic_annotators[i,t_rand] = np.argmax( self.Random_num.multinomial(1, sample_prob) )
                
        #clean the annotators that do not label
        mask_label = np.where(np.sum(sintetic_annotators,axis=0) != sintetic_annotators.shape[0]*-1)[0]
        sintetic_annotators = sintetic_annotators[:,mask_label]
        
        self.annotations = sintetic_annotators
        self.annotations_group = sintetic_annotators_group
        return sintetic_annotators,sintetic_annotators_group[mask_label,:]
        
    def save_sintetic(self,file_name='annotations',npy=True):
        if npy:
            np.save(file_name+'.npy',self.annotations.astype('int32')) 
        else:
            np.savetxt(file_name+'.csv',self.annotations,delimiter=',',fmt='%d') 

    def save_groups(self,file_name='annotations_group',npy=True):
        if npy:
            np.save(file_name+'.npy',self.annotations_group) 
        else:
            if self.annotations_group.shape[1] ==1:
                np.savetxt(file_name+'.csv',self.annotations_group,delimiter=',',fmt='%d') 
            else:
                np.savetxt(file_name+'.csv',self.annotations_group,delimiter=',',fmt='%.8f') 

    def save_generation_files(self,name_dataset,npy=False):
        if npy:
            np.save('groups_'+name_dataset+'.npy',self.prob_groups.reshape(1,-1))
            to_save = self.conf_matrix.reshape(self.conf_matrix.shape[0]*self.conf_matrix.shape[1],self.conf_matrix.shape[2])
            np.save('matrix_'+name_dataset+'.npy',to_save)
        else:
            np.savetxt('groups_'+name_dataset+'.csv',self.prob_groups.reshape(1,-1), delimiter=',',fmt='%.8f')
            to_save = self.conf_matrix.reshape(self.conf_matrix.shape[0]*self.conf_matrix.shape[1],self.conf_matrix.shape[2])
            np.savetxt('matrix_'+name_dataset+'.csv',to_save, delimiter=',',fmt='%.5f')