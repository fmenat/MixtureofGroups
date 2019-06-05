import numpy as np
import keras
def categorical_representation(obs,no_label =-1):
	## desirable represnetation
	N,T = obs.shape
	K = int(np.max(obs)+1) # assuming that are indexed in order
	y_obs_categorical = np.zeros((N,T,K),dtype='int8') #solo 0 o 1
	for i in range(N):
	    for t in range(T):
	        observado = obs[i,t]
	        if observado != no_label: #no_label symbol
	            y_obs_categorical[i,t,observado] = 1
	#if annotator do not annotate a data her one-hot is full of zeroes
	return y_obs_categorical

def categorical_masked_representation(obs, no_label=-1):
    ## desirable represnetation
    if len(obs.shape)!=3:
        y_obs_catmasked = categorical_representation(obs,no_label)
    else:
        y_obs_catmasked = obs
    mask =  np.sum(y_obs_catmasked,axis=-1)  == 0
    #if annotator do not annotate a data her one-hot is full of -1
    y_obs_catmasked[mask] = -1
    return y_obs_catmasked.transpose(0,2,1)

##transformar a repeat
def annotations2repeat(annotations):
    """
    assuming that annotations is a 3-dimensional array and with one hot vectors, and annotators
    that does not annotate a data have a one hot vectors of zeroes --> sum over annotators axis
    """
    if len(annotations.shape) ==2:
    	annotations = categorical_representation(annotations)
    return np.sum(annotations,axis=1,dtype='int32')

def annotations2repeat_efficient(obs,no_label=-1):
	"""
	Used when memory error is through over normal "annotations2repeat" function
	"""
	if len(obs.shape) ==2:
		N,T = obs.shape
		K = int(np.max(obs)+1) # assuming that are indexed in order
		repeats_obs = np.zeros((N,K),dtype='int32')
		for i in range(N):
			for t in range(T):
				observado = obs[i,t]
				if observado != no_label: #no_label symbol
					repeats_obs[i,observado] +=1
	else:
		repeats_obs = np.sum(obs,axis=1,dtype='int32')
	return repeats_obs

def DS_representation(obs):
	#representation for D&S
	N,T = obs.shape
	annotations = {i+1:{} for i in range(N)}
	for i in range(N):
	    for t in range(T):
	        observado = obs[i,t]
	        if observado != -1:
	            annotations[i+1][t+1] = [np.int16(observado)]
	return annotations

"""
Original/Based representation: (N,T)
Example: X = [  [1, 2,-1, 2,-1]
				[1,-1, 2,-1, 1] ]

Raykar need as one-hot: (N,T,K)
SET:  set_representation(X,needed="onehot")


Dawid and Skene need a dictionary: 
SET:  set_representation(X,needed="dawid")

Group-based Model need repeats: (N,K)
SET:  set_representation(X,needed="repeat")
"""

def set_representation(obs,needed="onehot"):
	if needed.lower()=="onehot" or needed.lower()=="one-hot":
		return categorical_representation(obs)
	elif needed.lower()=="repeat":
		return annotations2repeat_efficient(obs)
	elif needed.lower()=="dawid":
		return DS_representation(obs)
	elif needed.lower()=='onehotmasked' or needed.lower()=='rodriguesmasked':
		return categorical_masked_representation(obs)
    


#perform majority voting
def majority_voting(observed,repeats=True,onehot=True,probas=False):
    """
    Args:
        * repeats mean if the observed data come's as a repeat vector (counting the annotators labels)
        * onehot mean if the returned array come as a one hot of the classes
        * probas mean that a probability version of majority voting is returned
    """
    if not repeats:
        #r_obs = annotations2repeat(observed)
        r_obs = annotations2repeat_efficient(observed)
    else:
        r_obs = observed
        
    if probas:
        return r_obs/np.expand_dims(np.sum(r_obs,axis=-1,dtype='float32'),axis=1)

    mv = r_obs.argmax(axis=1) #over classes axis
    if onehot: 
        mv = keras.utils.to_categorical(mv)
    return mv