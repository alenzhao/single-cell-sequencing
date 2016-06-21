
import argparse


from sklearn.decomposition import PCA,KernelPCA
import parse_data
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans,SpectralClustering
import clustering
import random
import parse_data
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras import models
from collections import defaultdict
from sklearn.metrics import mean_squared_error


import numpy as np
import parse_data
import time
import math
import pickle
from deep_net import *
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t',"--transform", help="specify the transform set, default is all", choices=['all','labeled','unlabeled'],default='all')
    parser.add_argument('-ori',"--original", help="specify whether to do clustering on original data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-nn',"--neural_network", help="specify whether to do clustering on nn data, and please provide nn model")
    parser.add_argument('-f',"--fit", help="specify the fitting set, default is all", choices=['all','labeled','unlabeled'],default='all')
    parser.add_argument('-n',"--n_component", help="specify the number of components, default is 0, means all", type=int,default=0)
    parser.add_argument('-ni',"--nn_iteration", help="specify the number of nn iterations, default is 10", type=int,default=10)
    parser.add_argument('-c',"--n_cluster", help="specify the number of clusters, default is the number of labels", type=int,default=0)
    parser.add_argument('-i',"--n_iteration", help="specify the number of iteration, default is 10", type=int,default=10)
    parser.add_argument('-l',"--loss", help="specify whether to show loss or not, default is 1",choices=[0,1], type=int,default=1)
    parser.add_argument('-vdc',"--validation_data_classifier", help="specify whether to use validation data from pickle, default is 0",choices=[0,1], type=int,default=0)
    parser.add_argument('-vct',"--validation_cell_types", help="specify the number of validation cell types, default is 0", type=int,default=0)
    parser.add_argument('-seed',"--random_seed", help="specify the random seed, default is 0", type=int,default=0)
    parser.add_argument('-fs',"--fitself", help="specify whether to use validation data from pickle and fit the test set, default is 0",choices=[0,1], type=int,default=0)
    
    parser.add_argument('-r',"--reference_gene_file", help="specify the file that contatins the genes to be kept, if not specified then all the genes in training data will be used")
    parser.add_argument('-d',"--data_file", help="specify the data file, if not specified then a default training data file will be used")
    parser.add_argument('-sn',"--sample_normalize", help="specify whether to normalize each sample (divide by total reads), default is 1 (means True)",choices=[0,1], type=int,default=1)
    parser.add_argument('-gn',"--gene_normalize", help="specify whether to normalize each gene (mean 0 std 1), default is 1 (means True)",choices=[0,1], type=int,default=1)
    
    args=parser.parse_args()
    '''
    print args.transform
    print args.fit
    print args.n_component
    print args.n_cluster
    print args.n_iteration
    print args.loss
    '''
    data_file_name='../data/TPM_mouse_1_4_6_7_8_10_16.txt'
    if args.data_file is not None:
        data_file_name=args.data_file
    
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data(data_file_name,sample_normalize=args.sample_normalize,gene_normalize=args.gene_normalize,ref_gene_file=args.reference_gene_file)#'cluster_genes.txt'   
    fit_data=all_data
    transform_data=all_data

    #print 'all_data.shape: ', all_data.shape	
    
    #all_data=all_data[:100,:200]


    if args.fit=='labeled':
        fit_data=labeled_data
    elif args.fit=='unlabeled':
        fit_data=unlabeled_data
    if args.transform=='labeled':
        transform_data=labeled_data
    elif args.transform=='unlabeled':
        transform_data=unlabeled_data
    if args.validation_data_classifier==1 or args.validation_cell_types>0:
        if args.validation_data_classifier==1:
            with open('seed'+str(args.random_seed)+'_vp'+str(args.validation_percent)+'.pickle', 'rb') as handle:
                output_dict = pickle.load(handle)
        if args.validation_cell_types>0:
            with open('seed'+str(args.random_seed)+'_vct'+str(args.validation_cell_types)+'.pickle', 'rb') as handle:
                output_dict = pickle.load(handle)
            #fit_data=output_dict['train_X']
        transform_data=output_dict['test_X']
        fit_data=output_dict['train_X']
        if args.fitself:
            fit_data=output_dict['test_X']
        valid_Y=output_dict['test_Y_array']
    print 'fit_data.shape: ', fit_data.shape	
    print 'transform_data.shape: ', transform_data.shape	
    
    if args.original==1:
        code=transform_data
    elif args.neural_network is not None:
        custom_obj={}
        custom_obj['MyLayer']=MyLayer
        model_name=args.neural_network
        iteration=args.nn_iteration
        model = model_from_json(open('model/'+model_name+'.json').read(),custom_objects=custom_obj)
        load_model_weight_from_pickle(model,'model/'+model_name+'_'+str(iteration)+'.pickle')
        model.compile(optimizer='sgd', loss='mse')
        nlayer= len(model.layers)
        if args.validation_data_classifier:
            layer=nlayer-1
        else:
            layer=nlayer/2-1 
        code=get_output(model,layer,transform_data)
        print 'layer: ',layer
        print 'code shape: ', code.shape
    else:
        if args.n_component==0:
            pca=PCA()
        else:
            pca=PCA(args.n_component)
        print 'fitting data:' + args.fit
        pca.fit(fit_data)
        print 'transforming data: '+args.transform
        code=pca.transform(transform_data)
        print 'inverse transforming'
        recovered_data=pca.inverse_transform(code)

        print 'pca recover MSE:'
        print mean_squared_error(transform_data, recovered_data)
    
    useAllData=False
    if args.validation_data_classifier or args.validation_cell_types>0:
        Label=valid_Y
    else:
        Label=labeled_label
    print Label
    n_cluster=args.n_cluster
    if n_cluster==0:
        #n_cluster=max(Label)
        n_cluster=len(set(Label))
    print n_cluster

    est=KMeans(init='k-means++', n_clusters=n_cluster, n_init=args.n_iteration)
    if args.transform=='all' and args.validation_data_classifier==0 and args.validation_cell_types==0:
        useAllData=True
        Label=all_label
    print 'time\t ARI'
    clustering.bench_clustering(est,name='kmeans++', data=code, labels=Label,useAllData=useAllData)
