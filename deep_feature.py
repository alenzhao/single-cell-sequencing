
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras import models
from collections import defaultdict
from sklearn.metrics import mean_squared_error


from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import theano
from sklearn import svm
import numpy as np
import parse_data
import time
import math
import pickle
from deep_net import *

if __name__=='__main__':
	#test_nn_things()
    #all_data_landmark, labeled_data_landmark,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_1_4_6_7_8_10_16.txt', landmark=True)
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_1_4_6_7_8_10_16.txt', landmark=False)
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True, gene_normalize=True)   
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=False, gene_normalize=True)   
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True,ref_gene_file='cluster_genes.txt')   
    #group_gene_index_dict, sorted_group_names, group_gene_mat= parse_data.load_group_gene_index_dict(gene_names,'ppi_tf_merge_cluster.txt')
    #data_landmark=[all_data_landmark, labeled_data_landmark]
    #data_allgene=[all_data, labeled_data]
    #all_data_list=[all_data_landmark, all_data]
    #labeled_data_list=[labeled_data_landmark,labeled_data]
    #data=all_data
    #print data.shape
    #data=all_data_landmark
    #print data.shape
    #data=labeled_data
    #print data.shape
    #data=labeled_data_landmark
    #print data.shape
    #model_100Code3StackLandmark='NN100Code3StackLandmark'
    #model_100Code3StackAllgene='NN100Code3StackAllgene'
    #model_100Code1LayerAllgene='NN100Code1layerAllgene'
    #model_100Code1LayerLandmark='NN100Code1layerLandmark'
    
    #model_100Code3StackAllgene='NN100Code3StackAllgeneData01040607081016'
    #model_100Code1LayerAllgene='NN100Code1layerAllgeneData01040607081016'
    #model_100Code3StackAllgene='NN100Code3StackAllgeneData01040607081016unlabeled'
    #model_100Code1LayerAllgene='NN100Code1layerAllgeneData01040607081016unlabeled'
    model_names=[]
    train_iter={}
    #train_iter.append(20500)
    #train_iter.append(10000)
    #train_iter.append(13000)
    #train_iter.append(4800)
    #train_iter=[0,100,500,1000]
    train_iter=[i*100 for i in range(5)]
    train_iter.extend([i*500 for i in range(1,5)])
    #train_iter=[i*10 for i in range(11)]
    #train_iter[model_100Code3StackAllgene]=0
    #train_iter[model_100Code1LayerAllgene]=0
    #train_iter.append(1000)
    #model_names.append(model_100Code1LayerLandmark)
    #model_names.append(model_100Code1LayerAllgene)
    #model_names.append(model_100Code3StackLandmark)
    #model_names.append(model_100Code3StackAllgene)
    
    
    #model_names.append('NN100Code1layerAllgeneNData01040607081016relu')
    #model_names.append('NN100Code1layerAllgeneNData01040607081016tanh')
    #model_names.append('NN100Code1layerAllgeneNData01040607081016sigmoid')
    #model_names.append('NN100Code1layerAllgeneNDataDX01040607081016relu')
    #model_names.append('NN100Code1layerAllgeneNDataDX01040607081016tanh')
    #model_names.append('NN100Code1layerAllgeneNDataDX01040607081016sigmoid')
        
    model_name='NN100Code1layerAllgeneSNGNData01040607081016_PPITF_tanh'
    model_names.append(model_name)
    model_name='NN100Code1layerAllgeneSNGNData01040607081016_PPITF_relu'
    model_names.append(model_name)
    model_name='NN100Code1layerAllgeneSNGNData01040607081016_PPITF_sigmoid'
    model_names.append(model_name)
    model_name='NN100Code1layerAllgeneSNGNData01040607081016_PPITF_linear'
    model_names.append(model_name)
    
    output_dict={}
    custom_obj={}
    custom_obj['MyLayer']=MyLayer
    #output_dict["all_data_landmark"]=all_data_landmark
    #output_dict["all_data"]=all_data
    #output_dict["labeled_data_landmark"]=labeled_data_landmark
    #output_dict["labeled_data"]=labeled_data
    #output_dict["labeled_label"]=labeled_label
    #output_dict["all_label"]=all_label
    print 'all data shape:', all_data.shape
    print 'labeled data shape:', labeled_data.shape
    for iteration in train_iter:
        for model_name in model_names:
            print  'model_name: ', model_name
            #model = model_from_json(open('model/'+model_name+'.json').read())
            model = model_from_json(open('model/'+model_name+'.json').read(),custom_objects=custom_obj)
            #model.load_weights('model/'+model_name+'_'+str(iteration)+'.h5')
            load_model_weight_from_pickle(model,'model/'+model_name+'_'+str(iteration)+'.pickle')
            model.compile(optimizer='sgd', loss='mse')
            recovered_all_data = model.predict(all_data)
            print 'reconstruct error for all data:'
            print mean_squared_error(all_data, recovered_all_data)
            
            nlayer= len(model.layers)
            layer=nlayer/2-1
            print 'layer: ',layer
            code_labeled= get_output(model,layer,labeled_data)
            print 'labeled code shape: ', code_labeled.shape
            name=model_name+"_codelayer_"+str(code_labeled.shape[1])+"_labeled_data"
            print 'store as: ',name
            #output_dict[name]=code_labeled
            output_dict[name+"_"+str(iteration)]=code_labeled
            
            code_all= get_output(model,layer,all_data)
            print code_all.shape
            name=model_name+"_codelayer_"+str(code_all.shape[1])+"_all_data"
            print name
            output_dict[name+"_"+str(iteration)]=code_all
            if 'Stack' in model_name:
                layer=-2
                code_labeled= get_output(model,layer,labeled_data)
                print code_labeled.shape
                name=model_name+"_lastlayer_"+str(code_labeled.shape[1])+"_labeled_data"
                print name
                #output_dict[name]=code_labeled
                output_dict[name+"_"+str(iteration)]=code_labeled
                code_all= get_output(model,layer,all_data)
                print code_all.shape
                name=model_name+"_lastlayer_"+str(code_all.shape[1])+"_all_data"
                print name
                output_dict[name+"_"+str(iteration)]=code_all
    for key, val in output_dict.items():
        print 'save data and shape: ',key, val.shape
    with open('deep_feature.pickle', 'wb') as handle:
        print 'saving codes'
        pickle.dump(output_dict, handle)
