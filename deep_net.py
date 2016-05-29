import os   
#os.environ['THEANO_FLAGS'] = "device=gpu0"
#os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu0,lib.cnmem=1,allow_gc=False"
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu0,allow_gc=False"

import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

from myKerasLayer_new import MyLayer
from myOptimizers import SGD
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
#from keras.optimizers import SGD, Adadelta, Adagrad
from keras import models
#from keras.layers import containers, AutoEncoder
from collections import defaultdict


from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
#import theano
from sklearn import svm
import numpy as np
import parse_data
import time
import math
import pickle
def load_data(landmark=True):
    sample_label_dict=defaultdict(lambda: None)
    gene_sample_expression_dict=defaultdict(lambda: 0)
    sample_label_dict,gene_sample_expression_dict=parse_data.load_data(sample_label_dict,gene_sample_expression_dict,'mouse_10_expr_RPKM.txt','mouse_10_label.txt',landmark)
    #sample_label_dict,gene_sample_expression_dict=parse_data.load_data(sample_label_dict,gene_sample_expression_dict,'mouse_4_expr_RPKM.txt','mouse_4_label.txt')
    data,label=parse_data.gen_nn_data(sample_label_dict,gene_sample_expression_dict)
    return data,label

    
def nn_class_model(input_dim,output_dim, hidden_layer_size,num_hidden_layer):

    #input_dim = data.shape[1]
    #print list(set(sample_label_dict.values()))
    #print 'input_dim: ',input_dim
    #output_dim = len(list(set(sample_label_dict.values())))
    #output_dim=max(label)+1
    #label = np_utils.to_categorical(label, output_dim)

    #print 'output_dim: ',output_dim
    #hidden_layer_size=100
    #print 'hidden_layer_size: ',hidden_layer_size

    #num_hidden_layer=3

    model = Sequential()
    #model.add(Dense(output_dim=output_dim, input_dim=input_dim, init="glorot_uniform"))
    #model.add(Activation("tanh"))
    if hidden_layer_size>0:
        model.add(Dense(hidden_layer_size, init='normal',input_dim=input_dim))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        for i in range(num_hidden_layer-1):
            model.add(Dense(hidden_layer_size, init='normal'))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
        model.add(Dense(output_dim, init='normal'))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(output_dim=output_dim, input_dim=input_dim,init='normal'))
        model.add(Activation("softmax"))
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)

    #sgd = SGD(l2=0.0,lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")

    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model

def nn_denoising_autoencoder_model(input_dim, hidden_layer_size,num_hidden_layer,drop_out_rate):
    model = Sequential()
    
    model.add(Dense(hidden_layer_size, init='normal',input_dim=input_dim))
    model.add(Activation('tanh'))
    model.add(Dropout(drop_out_rate))
    for i in range(num_hidden_layer-1):
        model.add(Dense(hidden_layer_size, init='normal'))
        model.add(Activation('tanh'))
        model.add(Dropout(drop_out_rate))
    model.add(Dense(input_dim, init='normal'))
    model.add(Activation('tanh'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model
def keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,drop_out_rate,activation_func):
    print 'hidden_layer_size = ',hidden_layer_size
    model = models.Sequential()
    model.add(Dense(hidden_layer_size,input_dim=input_dim,activation=activation_func))
    model.add(Dense(input_dim,activation=activation_func))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model
    
def keras_denoising_autoencoder_model_1layer_new(input_dim, hidden_layer_size,drop_out_rate,activation_func,group_gene_mat,group_gene_dict):

    print 'hidden_layer_size = ',hidden_layer_size
    #print group_gene_index_dict
    model = models.Sequential()
    #model.add(MyLayer(hidden_layer_size,input_dim=input_dim,activation=activation_func))
    #model.add(MyLayer(input_dim,activation=activation_func))
    model.add(MyLayer(hidden_layer_size,input_dim=input_dim,activation=activation_func,input_output_mat=group_gene_mat.transpose(),group_gene_dict=group_gene_dict))
    model.add(MyLayer(input_dim,activation=activation_func, input_output_mat=group_gene_mat,group_gene_dict=group_gene_dict))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model
def keras_denoising_autoencoder_model(input_dim, hidden_layer_size,drop_out_rate,activation_func):
    # input shape: (nb_samples, input_dim)
    #middle_layer_size=int(math.sqrt(input_dim*hidden_layer_size))
    middle_layer_size=int(math.sqrt(input_dim/hidden_layer_size)*hidden_layer_size)
    
    #middle_layer_size=2*hidden_layer_size
    print 'hidden_layer_size = ',hidden_layer_size
    print 'middle_layer_size = ',middle_layer_size
    '''
    encoder = containers.Sequential()
    encoder.add(Dense(middle_layer_size,input_dim=input_dim,activation=activation_func))
    #encoder.add(Dropout(drop_out_rate))
    encoder.add(Dense(hidden_layer_size,activation=activation_func))
    #encoder.add(Dropout(drop_out_rate))
    
    decoder = containers.Sequential()
    decoder.add(Dense(middle_layer_size, input_dim=hidden_layer_size,activation=activation_func))
    #decoder.add(Dropout(drop_out_rate))
    decoder.add(Dense(input_dim,activation=activation_func))
    #decoder.add(Dropout(drop_out_rate))
    
    #encoder = containers.Sequential([Dense(middle_layer_size, input_dim=input_dim,activation=activation_func), Dense(hidden_layer_size,activation=activation_func)])
    #decoder = containers.Sequential([Dense(middle_layer_size, input_dim=hidden_layer_size,activation=activation_func), Dense(input_dim,activation=activation_func)])

    

    
    autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
    '''
    model = models.Sequential()
    model.add(Dense(middle_layer_size,input_dim=input_dim,activation=activation_func))
    model.add(Dense(hidden_layer_size,activation=activation_func))
    model.add(Dense(middle_layer_size, input_dim=hidden_layer_size,activation=activation_func))
    model.add(Dense(input_dim,activation=activation_func))
    
    #model.add(autoencoder)
    #model.layers[0].build() 

    # training the autoencoder:
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    #model.compile(optimizer='sgd', loss='mse')
    
    '''
    model.fit(X_train, X_train, nb_epoch=10)

    # predicting compressed representations of inputs:
    autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
    model.compile(optimizer='sgd', loss='mse')
    representations = model.predict(X_test)

    # the model is still trainable, although it now expects compressed representations as targets:
    model.fit(X_test, representations, nb_epoch=1)  # in this case the loss will be 0, so it's useless

    # to keep training against the original inputs, just switch back output_reconstruction to True:
    autoencoder.output_reconstruction = True
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X_train, X_train, nb_epoch=10)
    '''
    return model
def get_code(model,autoencoder, data):
    autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
    model.compile(optimizer='sgd', loss='mse')
    representations = model.predict(data)
    return data

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, batch_size, nb_epoch):
    #model.fit(data, label, nb_epoch=1000, batch_size=30,validation_split=0.1,shuffle=True,verbose=1,show_accuracy=True)
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size,verbose=1)
    print score
    return score



#model.fit(data, label, nb_epoch=1000, batch_size=30,validation_split=0.1,shuffle=True,verbose=1,show_accuracy=True)

def cross_validation(n_folds=5,nb_epoch=1000,batch_size=32,hidden_layer_size=100,num_hidden_layer=3):
    #n_folds = 5
    data, labels = load_data()
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
    #print len(skf)
    input_dim = data.shape[1]
    output_dim=max(labels)+1
    print 'input_dim: ',input_dim
    print 'output_dim: ',output_dim
    #nb_epoch=1000
    #batch_size=32
    #hidden_layer_size=100
    #num_hidden_layer=3
    performance=0
    for i, (train, test) in enumerate(skf):
        print "Running Fold", i+1, "/", n_folds
        #print train
        #print test
        model = None # Clearing the NN.
        model = nn_class_model(input_dim,output_dim, hidden_layer_size,num_hidden_layer)
        lab_train = np_utils.to_categorical(labels[train], output_dim)
        lab_test = np_utils.to_categorical(labels[test], output_dim)
        performance+=train_and_evaluate_model(model, data[train], lab_train, data[test], lab_test,batch_size, nb_epoch)
    performance/=n_folds
    print 'mean performance:', performance
    return performance
def try_epoch_nh_CV():
    n_folds=10
    #nb_epoch=1000
    batch_size=32
    hidden_layer_size=100
    #num_hidden_layer=2
    
    for num_hidden_layer in range(4):
        X=[]
        Y=[]    
        for nb_epoch in range(1000,1600,100):
            X.append(nb_epoch)
            performance=cross_validation(n_folds,nb_epoch,batch_size,hidden_layer_size,num_hidden_layer)
            Y.append(performance)
        plt.plot(X,Y,label='nh='+str(num_hidden_layer))
    plt.legend(loc=4)
    plt.ylabel('performamce')
    plt.xlabel('nb_epoch')
    plt.show()
#def get_output(model, layer, data):
#   get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
#   activations = get_activations(data) # same result as above
#   return activations
def get_output(model, layer, data):
    #get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    get_activations = theano.function([model.layers[0].input], model.layers[layer].output)
    activations = get_activations(data) # same result as above
    return activations
    
if __name__=='__main__':
    #test_nn_things()
    if 1:
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_1_4_6_7_8_10_16.txt')
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_4_6_7_10.txt', landmark=True)  
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True )   
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data_try.txt',sample_normalize=True,gene_normalize=True )   
        group_gene_index_dict, sorted_group_names, group_gene_mat = parse_data.load_group_gene_index_dict(gene_names,'group_try.txt')
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True )   
        #group_gene_index_dict, sorted_group_names, group_gene_mat = parse_data.load_group_gene_index_dict(gene_names,'ppi_tf_merge_cluster.txt')
        
        '''
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True,ref_gene_file='cluster_genes.txt')   
        group_gene_index_dict, sorted_group_names, group_gene_mat= parse_data.load_group_gene_index_dict(gene_names,'ppi_tf_merge_cluster.txt')
        '''
        data=all_data
        #dict_weights={}
        #print(len(all_weights))
        #print(len(labeled_weights))
        #print(len(unlabeled_weights))
        #for i,w in enumerate(unlabeled_weights):
        #   dict_weights[i]=w

        #data=np.float32(data)
        print data.shape
        input_dim = data.shape[1]
        print 'input_dim= ', input_dim
        hidden_layer_size=100
        drop_out_rate=1
        batch_size=32
        epoch_step=10
        max_iter=20
        print 'hidden_layer_size= ', hidden_layer_size
        print 'drop_out_rate= ',drop_out_rate
        print 'batch_size= ',batch_size
        print 'epoch_step= ',epoch_step
        now_iter=0
        #model_name='model/NN100Code3StackLandmark'
        #model_name='model/NN100Code1layerLandmark'
        #model_name='model/NN100Code1layerAllgene'
        #model_name='model/NN100Code3StackAllgene'
       
        #model_name='model/NN100Code1layerAllgeneData010406070810'
        #model_name='model/NN100Code3StackAllgeneData010406070810'
        
        #model_name='model/NN100Code1layerAllgeneData01040607081016unlabeled'
        #model_name='model/NN100Code1layerAllgeneData01040607081016'
        
        #model_name='model/NN100Code1layerAllgeneNData01040607081016linear'
        #model_name='model/NN100Code1layerAllgeneNDataDX01040607081016linear'
        #activation_func='linear'
        #model_name='model/NN100Code1layerAllgeneNData01040607081016_tanh'
        model_name='model/NN100Code1layerAllgeneNDataDX01040607081016_tanh'
        activation_func='tanh'
        #model_name='model/NN100Code1layerAllgeneNData01040607081016_relu'
        #model_name='model/NN100Code1layerAllgeneNDataDX01040607081016_relu'
        #activation_func='relu'
        #model_name='model/NN100Code1layerAllgeneNData01040607081016_sigmoid'
        #model_name='model/NN100Code1layerAllgeneNDataDX01040607081016_sigmoid'
        #activation_func='sigmoid'
        
        print 'activation_func= ',activation_func
        #model_name='model/NN100Code3StackAllgeneData01040607081016unlabeled'
        #model_name='model/NN100Code3StackAllgeneData01040607081016'
        
        #model_name='JustATest'
        if now_iter==0:
            #model=keras_denoising_autoencoder_model(input_dim, hidden_layer_size,drop_out_rate,activation_func)
            #model=keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,drop_out_rate,activation_func)
            model=keras_denoising_autoencoder_model_1layer_new(input_dim, hidden_layer_size,drop_out_rate,activation_func,group_gene_mat,group_gene_index_dict)
            json_string = model.to_json()
            f=open(model_name+'.json', 'w')
            f.write(json_string)
            #model.save_weights(model_name+'_'+str(0)+'.h5', overwrite=True)
            f.close()
        else:
            model = model_from_json(open(model_name+'.json').read())
            model.load_weights(model_name+'_'+str(now_iter)+'.h5')
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)
        print 'start train' 
        '''
        for layer in model.layers:
                weights = layer.get_weights()
                print weights
        '''
        while now_iter<max_iter:
            print 'now_iter= ',now_iter
            #model.fit(labeled_data, labeled_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
            model.fit(all_data, all_data, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            #model.fit(unlabeled_data, unlabeled_data, sample_weight=unlabeled_weights, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            now_iter+=epoch_step
            #model.save_weights(model_name+'_'+str(now_iter)+'.h5', overwrite=True)
        print 'after train'
        
        for layer in model.layers:
                weights = layer.get_weights()
                print weights[0].todense()[:3,:3]
                layer.set_weights(weights)
                weights = layer.get_weights()
                print weights[0].todense()[:3,:3]

                #print weights[0]
                #print type(weights)
        
