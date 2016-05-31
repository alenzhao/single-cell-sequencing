import os    
#os.environ['THEANO_FLAGS'] = "device=gpu0"
#os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu0,lib.cnmem=1,allow_gc=False"
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu0,allow_gc=False"

import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'


from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras import models
#from keras.layers import containers, AutoEncoder
from collections import defaultdict

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
#import theano
from sklearn import svm
import numpy as np
import parse_data
import time
import math
import pickle
from deep_net import *
if __name__=='__main__':
    if 1:
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True)   
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True,ref_gene_file='cluster_genes.txt')   
        group_gene_index_dict, sorted_group_names, group_gene_mat= parse_data.load_group_gene_index_dict(gene_names,'ppi_tf_merge_cluster.txt')
        #print all_data[:20,:20]
        
        data=all_data
        print data.shape
        input_dim = data.shape[1]
        print 'input_dim= ', input_dim
        hidden_layer_size=100
        drop_out_rate=1
        batch_size=32
        epoch_step=10
        max_iter=10000
        print 'hidden_layer_size= ', hidden_layer_size
        print 'drop_out_rate= ',drop_out_rate
        print 'batch_size= ',batch_size
        print 'epoch_step= ',epoch_step
        now_iter=0
        valid_set_index=np.random.choice(all_data.shape[0],size=all_data.shape[0]/10,replace=False)
        #print valid_set_index
        train_set_index=[x for x in range(all_data.shape[0]) if x not in valid_set_index]
        #print train_set_index
        
        train_data=all_data[train_set_index,:]
        valid_data=all_data[valid_set_index,:]
        #train_data=unlabeled_data
        #valid_data=labeled_data
        print train_data.shape
        print valid_data.shape
        #model_name='model/NN100Code3StackLandmark'
        #model_name='model/NN100Code1layerLandmark'
        #model_name='model/NN100Code1layerAllgene'
        #model_name='model/NN100Code3StackAllgene'
       
        #model_name='model/NN100Code1layerAllgeneData010406070810'
        #model_name='model/NN100Code3StackAllgeneData010406070810'
        
        #model_name='model/NN100Code1layerAllgeneData01040607081016unlabeled'
        #model_name='model/NN100Code1layerAllgeneData01040607081016'
        #model_name='model/NN100Code3StackAllgeneData01040607081016unlabeled'
        #model_name='model/NN100Code3StackAllgeneData01040607081016'
        
        #model_name='JustATest'
        #if now_iter==0:
        #    json_string = model.to_json()
        #    f=open(model_name+'.json', 'w')
        #    f.write(json_string)
        #    model.save_weights(model_name+'_'+str(0)+'.h5', overwrite=True)
        #    f.close()
        #else:
        #    model = model_from_json(open(model_name+'.json').read())
        #    model.load_weights(model_name+'_'+str(now_iter)+'.h5')
        #    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #    model.compile(loss='mean_squared_error', optimizer=sgd)
        #model_try={}
        #model_name='TPM_whitened_data_1layer_PPITF_linear'
        #activation_func='linear'
        #model_name='TPM_whitened_data_1layer_PPITF_relu'
        #activation_func='relu'
        #model_name='TPM_whitened_data_1layer_PPITF_tanh'
        #activation_func='tanh'
        model_name='TPM_whitened_data_1layer_PPITF_sigmoid'
        activation_func='sigmoid'
        #model_name='TPM_whitened_data_1layer_sidmoid'
        #activation_func='relu'
        #model_name='TPM_whitened_data_1layer_relu'
        #activation_func='tanh'
        #model_name='TPM_whitened_data_1layer_tanh'
        print 'activation_func= ',activation_func
        #model=keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,drop_out_rate,activation_func)
        model=keras_denoising_autoencoder_model_1layer_new(input_dim, hidden_layer_size,drop_out_rate,activation_func,group_gene_mat,group_gene_index_dict)
        
        #model_name='TPM_whitened_data_3layer'
        #model=keras_denoising_autoencoder_model(input_dim, hidden_layer_size,drop_out_rate,activation_func)
        #for model_name,model in model_try.items():
        f_output=open(model_name+'_deep_loss_check.txt','w') 
        while now_iter<max_iter:
            print 'now_iter= ',now_iter
            reconstructed_train=model.predict(train_data)
            train_mse=mean_squared_error(train_data,reconstructed_train)
            print 'train_predict_MSE: ', train_mse
            reconstructed_valid=model.predict(valid_data)
            valid_mse=mean_squared_error(valid_data,reconstructed_valid)
            print 'valid_predict_MSE: ', valid_mse
            #model.fit(labeled_data, labeled_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
            model.fit(train_data, train_data, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            #model.fit(unlabeled_data, unlabeled_data, sample_weight=unlabeled_weights, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            now_iter+=epoch_step
            #model.save_weights(model_name+'_'+str(now_iter)+'.h5', overwrite=True)
            f_output.write(str(now_iter)+"\t"+str(train_mse)+'\t'+str(valid_mse)+"\n")
        f_output.close()
