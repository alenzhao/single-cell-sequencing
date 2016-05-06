
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
def load_data(landmark=True):
	sample_label_dict=defaultdict(lambda: None)
	gene_sample_expression_dict=defaultdict(lambda: 0)
	sample_label_dict,gene_sample_expression_dict=parse_data.load_data(sample_label_dict,gene_sample_expression_dict,'mouse_10_expr_RPKM.txt','mouse_10_label.txt',landmark)
	#sample_label_dict,gene_sample_expression_dict=parse_data.load_data(sample_label_dict,gene_sample_expression_dict,'mouse_4_expr_RPKM.txt','mouse_4_label.txt')
	data,label=parse_data.gen_nn_data(sample_label_dict,gene_sample_expression_dict)
	return data,label

def keras_denoising_autoencoder_model(input_dim, hidden_layer_size,drop_out_rate,activation_func):
	# input shape: (nb_samples, input_dim)
	#middle_layer_size=int(math.sqrt(input_dim*hidden_layer_size))
	middle_layer_size=int(math.sqrt(input_dim/hidden_layer_size)*hidden_layer_size)
	
	#middle_layer_size=2*hidden_layer_size
	print 'hidden_layer_size = ',hidden_layer_size
	print 'middle_layer_size = ',middle_layer_size
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
	return model

def get_output(model, layer, data):
    #get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    get_activations = theano.function([model.layers[0].input], model.layers[layer].output)
    activations = get_activations(data) # same result as above
    return activations
	

if __name__=='__main__':
	#test_nn_things()
    #all_data_landmark, labeled_data_landmark,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_1_4_6_7_8_10_16.txt', landmark=True)
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_1_4_6_7_8_10_16.txt', landmark=False)
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',whitening=True)   
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
    model_100Code3StackAllgene='NN100Code3StackAllgeneData01040607081016unlabeled'
    model_100Code1LayerAllgene='NN100Code1layerAllgeneData01040607081016unlabeled'
    model_names=[]
    train_iter={}
    #train_iter.append(20500)
    #train_iter.append(10000)
    #train_iter.append(13000)
    #train_iter.append(4800)
    #train_iter=[0,100,500,1000]
    train_iter=[i*10 for i in range(11)]
    #train_iter[model_100Code3StackAllgene]=0
    #train_iter[model_100Code1LayerAllgene]=0
    #train_iter.append(1000)
    #model_names.append(model_100Code1LayerLandmark)
    model_names.append(model_100Code1LayerAllgene)
    #model_names.append(model_100Code3StackLandmark)
    model_names.append(model_100Code3StackAllgene)
    output_dict={}
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
            model = model_from_json(open('model/'+model_name+'.json').read())
            model.load_weights('model/'+model_name+'_'+str(iteration)+'.h5')
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
