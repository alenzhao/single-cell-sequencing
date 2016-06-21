import numpy as np

import os   
#os.environ['THEANO_FLAGS'] = "device=gpu0"
#os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu0,lib.cnmem=1,allow_gc=False"
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu0,allow_gc=False"

import theano
import argparse
from myKerasLayer_new import MyLayer
from myOptimizers import SGD
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
#from keras.optimizers import SGD, Adadelta, Adagrad
from keras import models
from collections import defaultdict


from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import svm
import parse_data
import time
import math
import pickle
def keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,activation_func,label=None,output_dim=None):
    print 'hidden_layer_size = ',hidden_layer_size
    model = models.Sequential()
    model.add(Dense(hidden_layer_size,input_dim=input_dim,activation=activation_func))
    if label is None:
        model.add(Dense(input_dim,activation=activation_func))
    else:
        model.add(Dense(output_dim,activation='softmax'))
    return model
    
def keras_denoising_autoencoder_model_1layer_new(input_dim, hidden_layer_size,activation_func,group_gene_mat,group_gene_dict,label=None,output_dim=None):

    #print 'hidden_layer_size = ',hidden_layer_size
    model = models.Sequential()
    model.add(MyLayer(hidden_layer_size,input_dim=input_dim,activation=activation_func,input_output_mat=group_gene_mat.transpose(),group_gene_dict=group_gene_dict))
    if label is None:
        model.add(MyLayer(input_dim,activation=activation_func, input_output_mat=group_gene_mat,group_gene_dict=group_gene_dict))
    else:
        model.add(Dense(output_dim,activation='softmax'))

    return model
def keras_denoising_autoencoder_model_3layer_new(input_dim, hidden_layer_size,activation_func,group_gene_mat,group_gene_dict,label=None,output_dim=None):

    print 'hidden_layer_size = ',hidden_layer_size
    model = models.Sequential()
    model.add(MyLayer(hidden_layer_size,input_dim=input_dim,activation=activation_func,input_output_mat=group_gene_mat.transpose(),group_gene_dict=group_gene_dict))
    model.add(Dense(hidden_layer_size,activation=activation_func)) 
    if label is None:
        model.add(Dense(group_gene_mat.shape[0],activation=activation_func)) 
        model.add(MyLayer(input_dim,activation=activation_func, input_output_mat=group_gene_mat,group_gene_dict=group_gene_dict))
    else:
        model.add(Dense(output_dim,activation='softmax'))

    return model

def keras_denoising_autoencoder_model_3layer(input_dim, hidden_layer_size,activation_func,middle_layer_size=None,label=None,output_dim=None):
    if middle_layer_size is None:
        middle_layer_size=int(math.sqrt(input_dim/hidden_layer_size)*hidden_layer_size)
    print 'hidden_layer_size = ',hidden_layer_size
    print 'middle_layer_size = ',middle_layer_size
    model = models.Sequential()
    model.add(Dense(middle_layer_size,input_dim=input_dim,activation=activation_func))
    model.add(Dense(hidden_layer_size,activation=activation_func))
    if label is None:
        model.add(Dense(middle_layer_size, input_dim=hidden_layer_size,activation=activation_func))
        model.add(Dense(input_dim,activation=activation_func))
    else:
        model.add(Dense(output_dim,activation='softmax'))

    return model

def get_output(model, layer, data):
    #get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    get_activations = theano.function([model.layers[0].input], model.layers[layer].output)
    activations = get_activations(data) # same result as above
    return activations

def save_model_weight_to_pickle(model,file_name):
    print 'saving weights'
    weight_list=[]
    for layer in model.layers:
            weights = layer.get_weights()
            weight_list.append(weights)
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

def load_model_weight_from_pickle(model,file_name):
    print 'loading weights'
    weight_list=[]
    with open(file_name, 'rb') as handle:
        weight_list = pickle.load(handle)
    for layer,weights in zip(model.layers,weight_list):
            layer.set_weights(weights)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('architecture')
    
    parser.add_argument('-p',"--PPI_TF_cluster_file", help="specify the cluster file for layer connection, if not specified then Dense layer will be used")
    parser.add_argument('-r',"--reference_gene_file", help="specify the file that contatins the genes to be kept, if not specified then all the genes in training data will be used")
    parser.add_argument('-d',"--data_file", help="specify the data file, if not specified then a default training data file will be used")
    parser.add_argument('-mn',"--model_name", help="specify the model_name, if not specified then a default model_name will be used")
    
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument('-sm',"--store_output_model", help="specify whether to store output models, default is 1 (means True)",choices=[0,1], type=int,default=1)
    #parser.add_argument('-sl',"--store_output_loss", help="specify whether to store loss in each iteration, default is 0 (means False)",choices=[0,1], type=int,default=0)
    #parser.add_argument('-tl',"--training_loss", help="specify whether to load training loss in each iteration, default is 0 (means False)",choices=[0,1], type=int,default=0)
    parser.add_argument('-vp',"--validation_percent", help="specify the percentage of validation set, default is 0", type=float,default=0.0)
    parser.add_argument('-vct',"--validation_cell_types", help="specify the number of validation cell types, default is 0", type=int,default=0)
    parser.add_argument('-seed',"--random_seed", help="specify the random seed, default is 0", type=int,default=0)
    
    parser.add_argument('-stc',"--supervised_training_classifier", help="specify whether to use supervised training classifier, default is 0 (means False)",choices=[0,1], type=int,default=0)
   

    parser.add_argument('-a',"--architecture", help="specify the deep learning architecture", choices=['1layer','3layer'],default='1layer')
    parser.add_argument('-m',"--max_iteration", help="specify the maximum training iteration, default is 100", type=int,default=100)
    parser.add_argument('-s',"--step_size", help="specify the training step size, default is 10", type=int,default=10)
    parser.add_argument('-hls',"--hidden_layer_size", help="specify the hidden layer size, default is 100", type=int,default=100)
    parser.add_argument('-mls',"--middle_layer_size", help="specify the middle layer size, default is 696", type=int,default=696)
    parser.add_argument('-si',"--starting_iteration", help="specify the starting iteration (resume from the last training, -sm is need to store model), default is 0", type=int,default=0)
    parser.add_argument('-b',"--batch_size", help="specify the training batch size, default is 32", type=int,default=32)
    parser.add_argument('-act',"--activation_function", help="specify the training activation_function, default is tanh",choices=['tanh','relu','sigmoid','linear'],default='tanh')
    
    parser.add_argument('-sgd_lr',"--SGD_learning_rate", help="specify the SGD learning rate, default is 0.1", type=float,default=0.1)
    parser.add_argument('-sgd_m',"--SGD_momentum", help="specify the SGD momentum, default is 0.9", type=float,default=0.9)
    parser.add_argument('-sgd_d',"--SGD_decay", help="specify the SGD decay, default is 1e-6", type=float,default=1e-6)
    parser.add_argument('-sgd_n',"--SGD_nesterov", help="specify the SGD nesterov, default is 1 (means True)",choices=[0,1], type=int,default=1)
    
    
    parser.add_argument('-sn',"--sample_normalize", help="specify whether to normalize each sample (divide by total reads), default is 1 (means True)",choices=[0,1], type=int,default=1)
    parser.add_argument('-gn',"--gene_normalize", help="specify whether to normalize each gene (mean 0 std 1), default is 1 (means True)",choices=[0,1], type=int,default=1)
    
    args=parser.parse_args()
    np.random.seed(args.random_seed)
    
    #if args.store_output_loss or args.training_loss:
    #    args.store_output_model=0
    '''
    print args.PPI_TF_cluster_file
    print args.reference_gene_file
    
    print args.data_file
    print args.model_name
    
    print args.store_output_model
    print args.store_output_loss    
    
    print args.architecture
    print args.max_iteration
    print args.starting_iteration
    print args.step_size
    print args.batch_size
    print args.activation_function
    print args.SGD_learning_rate
    print args.SGD_momentum
    print args.SGD_decay
    print args.SGD_nesterov
    print args.sample_normalize
    print args.gene_normalize
    '''
    data_file_name='../data/TPM_mouse_1_4_6_7_8_10_16.txt'
    #data_file_name='../data_try.txt'

    if args.data_file is not None:
        data_file_name=args.data_file
    model_name=args.architecture+"_SN"+str(args.sample_normalize)+"_GN"+str(args.gene_normalize)+"_BS"+str(args.batch_size)
    model_name+="_hls"+str(args.hidden_layer_size)+"_mls"+str(args.middle_layer_size)
    model_name+='_seed'+str(args.random_seed)
    if args.PPI_TF_cluster_file is not None:
        model_name+='_PPITF'
    if args.supervised_training_classifier:
        model_name+='_classifier'
    if args.validation_cell_types>0:
        model_name+='_vct'+str(args.validation_cell_types)
    model_name+="_"+args.activation_function
    if args.model_name is not None:
        model_name=args.model_name
    #print model_name
    if 1:
        
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data(data_file_name,sample_normalize=args.sample_normalize,gene_normalize=args.gene_normalize,ref_gene_file=args.reference_gene_file)#'cluster_genes.txt'   
        
        if args.PPI_TF_cluster_file is not None:
            group_gene_index_dict, sorted_group_names, group_gene_mat= parse_data.load_group_gene_index_dict(gene_names,args.PPI_TF_cluster_file)#"PPI_TF_merge_cluster.txt"
        
        train_data=all_data
        train_valid=train_data
        nn_label=None
        output_dim=None
        if args.supervised_training_classifier:
             
            output_dim=max(labeled_label)+1
            nn_label=np_utils.to_categorical(labeled_label, output_dim)
            #print 'nn_label!'
            #print nn_label
            train_data=labeled_data
            train_valid=nn_label
            valid_data=train_data
            valid_valid=train_valid
        if args.validation_percent>0:
            s=int(train_data.shape[0]*args.validation_percent)
            if s==0:
                s=1
            valid_set_index=np.random.choice(train_data.shape[0],size=s,replace=False)
            train_set_index=[x for x in range(train_data.shape[0]) if x not in valid_set_index]
            
            valid_data=train_data[valid_set_index,:] 
            valid_valid=train_valid[valid_set_index,:]
            train_data=train_data[train_set_index,:]
            train_valid=train_valid[train_set_index,:]
            

            #print 'saving weights'
            output_dict={}
            output_dict['train_X']=train_data
            output_dict['test_X']=valid_data
            output_dict['train_Y']=train_valid
            output_dict['test_Y']=valid_valid
            output_dict['test_Y_array']=labeled_label[valid_set_index]
            with open('seed'+str(args.random_seed)+'_vp'+str(validation_percent)+'.pickle', 'wb') as handle:
                pickle.dump(output_dict, handle)
        if args.validation_cell_types>0:
            #s=int(train_data.shape[0]*args.validation_percent)
            #if s==0:
            #    s=1
            valid_cell_types=np.random.choice(np.arange(1,max(labeled_label)+1),size=args.validation_cell_types,replace=False)
            print 'valid cell types:'
            for ct in valid_cell_types:
                print ct, label_unique_list[ct]
            valid_set_index=[]
            print labeled_label
            for index,lab in enumerate(labeled_label):
                if lab in valid_cell_types:
                    valid_set_index.append(index)
            print valid_set_index
            #valid_set_index=np.random.choice(train_data.shape[0],size=s,replace=False)
            train_set_index=[x for x in range(train_data.shape[0]) if x not in valid_set_index]
            
            valid_data=train_data[valid_set_index,:] 
            valid_valid=train_valid[valid_set_index,:]
            train_data=train_data[train_set_index,:]
            train_valid=train_valid[train_set_index,:]
            

            #print 'saving weights'
            output_dict={}
            output_dict['train_X']=train_data
            output_dict['test_X']=valid_data
            output_dict['train_Y']=train_valid
            output_dict['test_Y']=valid_valid
            output_dict['test_Y_array']=labeled_label[valid_set_index]
            with open('seed'+str(args.random_seed)+'_vct'+str(args.validation_cell_types)+'.pickle', 'wb') as handle:
                pickle.dump(output_dict, handle)
        
        validation_data=(valid_data,valid_valid)
            
        
        #train_data=unlabeled_data
        #valid_data=labeled_data
        
        print train_data.shape
        input_dim = train_data.shape[1]
        print 'input_dim= ', input_dim
        hidden_layer_size=args.hidden_layer_size
        #drop_out_rate=1
        batch_size=args.batch_size
        epoch_step=args.step_size
        max_iter=args.max_iteration
        activation_function=args.activation_function
        print 'batch_size= ',batch_size
        print 'epoch_step= ',epoch_step
        print 'activation_function= ',activation_function
        now_iter=args.starting_iteration

        custom_obj={}
        custom_obj['MyLayer']=MyLayer
        if now_iter==0:# and args.training_loss==0:
            print 'initializing model'
            
            if args.PPI_TF_cluster_file is not None:
                if args.architecture =='1layer':
                    print 'cluster 1 layer'
                    model=keras_denoising_autoencoder_model_1layer_new(input_dim, hidden_layer_size,activation_function,group_gene_mat,group_gene_index_dict,output_dim=output_dim,label=nn_label)
                elif args.architecture =='3layer':
                    print 'cluster 3 layer'
                    model=keras_denoising_autoencoder_model_3layer_new(input_dim, hidden_layer_size,activation_function,group_gene_mat,group_gene_index_dict,output_dim=output_dim,label=nn_label)
            else:
                if args.architecture =='1layer':
                    print 'Dense 1layer'
                    #model=keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,activation_function)
                    model=keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,activation_function,output_dim=output_dim,label=nn_label)
                elif args.architecture =='3layer':
                    print 'Dense 3layer'
                    model=keras_denoising_autoencoder_model_3layer(input_dim, hidden_layer_size,activation_function,middle_layer_size=args.middle_layer_size,output_dim=output_dim,label=nn_label)
            if args.store_output_model:
                json_string = model.to_json()
                f=open('model/'+model_name+'.json', 'w')
                f.write(json_string)
                save_model_weight_to_pickle(model,'model/'+model_name+'_'+str(0)+'.pickle')
                f.close()
        else:
            model = model_from_json(open('model/'+model_name+'.json').read(),custom_objects=custom_obj)
            load_model_weight_from_pickle(model,'model/'+model_name+'_'+str(now_iter)+'.pickle')
        
        sgd = SGD(lr=args.SGD_learning_rate, decay=args.SGD_decay, momentum=args.SGD_decay, nesterov=args.SGD_nesterov)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        if args.supervised_training_classifier:
            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            #model.compile(loss='categorical_crossentropy', optimizer='adagrad',metrics=['accuracy'])
        
        print 'train_data.shape: ',train_data.shape
        print 'valid_data.shape: ',valid_data.shape
        print 'train_valid.shape: ',train_valid.shape
        print 'valid_valid.shape: ',valid_valid.shape

        #print   args.store_output_model
        #print   args.store_output_loss
        #print   args.training_loss
       # if args.store_output_model:
        while now_iter<max_iter:
            print 'now_iter= ',now_iter
            predicted_train=model.predict(train_data)
            #print predicted_train
            #train_mse=mean_squared_error(train_valid,predicted_train)
            #score = model.evaluate(X_test, y_test, batch_size=32)
            #print 'train_predict_MSE: ', train_mse
            #model.fit(labeled_data, labeled_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
            #model.fit(all_data, all_data, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            #model.fit(train_data, train_valid, batch_size=batch_size, nb_epoch=epoch_step,verbose=2)
            model.fit(train_data, train_valid, batch_size=batch_size, nb_epoch=epoch_step,verbose=2,validation_data=validation_data)
            #model.fit(unlabeled_data, unlabeled_data, sample_weight=unlabeled_weights, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            now_iter+=epoch_step
            if args.store_output_model:
                save_model_weight_to_pickle(model,'model/'+model_name+'_'+str(now_iter)+'.pickle')
        '''
        if args.store_output_loss:
            f_output=open('loss/'+model_name+'_deep_loss_check.txt','w') 
            while now_iter<max_iter:
                print 'now_iter= ',now_iter
                #reconstructed_train=model.predict(train_data)
                #train_mse=mean_squared_error(train_data,reconstructed_train)
                predicted_train=model.predict(train_data)
                train_mse=mean_squared_error(train_valid,predicted_train)
                print 'train_predict_MSE: ', train_mse
                reconstructed_valid=model.predict(valid_data)
                valid_mse=mean_squared_error(valid_data,reconstructed_valid)
                print 'valid_predict_MSE: ', valid_mse
                model.fit(train_data, train_data, batch_size=batch_size, nb_epoch=epoch_step,verbose=0)
                now_iter+=epoch_step
                f_output.write(str(now_iter)+"\t"+str(train_mse)+'\t'+str(valid_mse)+"\n")
                f_output.flush()
                os.fsync(f_output.fileno())
            f_output.close()
        '''
        '''
        if args.training_loss:
            f_output=open('loss/'+model_name+'_deep_train_loss.txt','w') 
            while now_iter<max_iter:
                print 'now_iter= ',now_iter
                load_model_weight_from_pickle(model,'model/'+model_name+'_'+str(now_iter)+'.pickle')
                #reconstructed_train=model.predict(train_data)
                #train_mse=mean_squared_error(train_data,reconstructed_train)
                predicted_train=model.predict(train_data)
                train_mse=mean_squared_error(train_valid,predicted_train)
                print 'train_predict_MSE: ', train_mse
                #reconstructed_valid=model.predict(valid_data)
                #valid_mse=mean_squared_error(valid_data,reconstructed_valid)
                #print 'valid_predict_MSE: ', valid_mse
                #model.fit(train_data, train_data, batch_size=batch_size, nb_epoch=epoch_step,verbose=0)
                now_iter+=epoch_step
                f_output.write(str(now_iter)+"\t"+str(train_mse)+"\n")
                f_output.flush()
                os.fsync(f_output.fileno())
            f_output.close()
        '''

