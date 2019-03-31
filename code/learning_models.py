#logistic regression by sklearn
from sklearn.linear_model import LogisticRegression
def LogisticRegression_Sklearn(epochs):
    """
        solver: Conjugate gradient (divided by hessian) as original Raykar paper
        warm_start set True to incremental fit (training): save last params on fit
    """
    return LogisticRegression(C=1., max_iter=epochs,fit_intercept=True
                       ,solver='newton-cg',multi_class='multinomial',warm_start=True,n_jobs=-1)
    #for sgd solver used "sag"

from keras.models import Sequential,Model
from keras.layers import *
def LogisticRegression_Keras(input_dim,output_dim):
    model = Sequential() 
    model.add(Dense(output_dim, input_shape=input_dim, activation='softmax')) 
    return model

#MLP Simple
def MLP_Keras(input_dim,output_dim,units,hidden_deep,BN=False,drop=0.0):
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    for i in range(hidden_deep): #all the deep layers
        model.add(Dense(units,activation='relu'))
        if BN:
            model.add(BatchNormalization())
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
    model.add(Dense(output_dim, activation='softmax'))     
    return model

def default_CNN(input_dim,output_dim): #quizas cambiara  CP,CP,CP 
    #weight_decay = 1e-4
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    model.add(Conv2D(32,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    
    #another layer?-yes
    model.add(Conv2D(128,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25)) #maybe more 512 y 0.5 d dropa
    model.add(Dense(output_dim, activation='softmax'))      
    return model

def default_RNN(input_dim,output_dim):
    #revisar la red de Rodrigues
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    model.add(CuDNNGRU(64,return_sequences=True))
    model.add(CuDNNGRU(32,return_sequences=False))
    model.add(Dense(output_dim, activation='softmax'))     
    return model

def default_RNNw_emb(input_dim,output_dim,len): #len is the length of the vocabulary
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    model.add(Embedding(input_dim=len,output_dim=128,input_length=input_dim[0])) #si son muy pocos datos en texto inicializar embedding con Glove
    model.add(CuDNNGRU(64,return_sequences=True))
    model.add(CuDNNGRU(32,return_sequences=False))
    model.add(Dense(output_dim, activation='softmax'))     
    return model


#### idea: tener representaciones de modelos neuronales bases para tipos de problemas
def CNN_simple(input_dim,output_dim,units,hidden_deep,double=False,BN=False,drop=0.0): #CP
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    start_unit = units
    for i in range(hidden_deep): #all the deep layers
        model.add(Conv2D(start_unit,(3,3),strides=1,padding='same',activation='relu'))
        if BN:
            model.add(BatchNormalization())
        if double:
            model.add(Conv2D(start_unit,(3,3),strides=1,padding='same',activation='relu'))
            if BN:
                model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
        start_unit = start_unit*2
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    if BN:
        model.add(BatchNormalization())
    if drop!= 0 and drop != None and drop != False:
        model.add(Dropout(drop))
    model.add(Dense(output_dim, activation='softmax')) 
    return model


def RNN_simple(input_dim,output_dim,units,hidden_deep,drop=0.0,embed=False,len=0,out=0):
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    if embed:
        model.add(Embedding(input_dim=len,output_dim=out,input_length=input_dim[0]))
    start_unit = units
    for i in range(hidden_deep): #all the deep layers
        if i == hidden_deep-1:
            model.add(CuDNNGRU(start_unit,return_sequences=False)) #a.k.a flatten
        else:
            model.add(CuDNNGRU(start_unit,return_sequences=True))
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
        start_unit = start_unit/2 #o mantener
    model.add(Dense(output_dim, activation='softmax')) 
    return model


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as p16
def through_VGG(X,pooling_mode=None):
    """
        Pass data X through VGG 16
        * pooling_mode: as keras say could be None, 'avg' or 'max' (in order to reduce dimensionality)
    """
    X_vgg = p16(X)
    input_tensor=Input(shape=X_vgg.shape[1:])
    modelVGG = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling=pooling_mode ) # LOAD PRETRAINED MODEL 
    return_value = modelVGG.predict(X_vgg)
    return return_value#.reshape(return_value.shape[0],np.prod(return_value.shape[1:]))

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as pIncept
def through_InceptionV3(X):
    """
        Pass data X through Inception V3
    """
    X_incept = pIncept(X)
    input_tensor=Input(shape=X_incept.shape[1:])
    modelInception = InceptionV3(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling=None ) # LOAD PRETRAINED MODEL 
    return modelInception.predict(X_incept)
