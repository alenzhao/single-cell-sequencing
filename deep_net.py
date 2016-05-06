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
def get_output(model, layer, data):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(data) # same result as above
    return activations
    
def test_nn_things():
    start_time=time.time()
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_4_6_7_10.txt')
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_4_6_7_10.txt', landmark=True)
    #data, labels = load_data(False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print 'loading data elapsed time: ',elapsed_time
    data=all_data
    print data.shape
    print label_unique_list
    #print data
    #print data[1]
    #print data[1,:]
    #print data[:,1]
    input_dim = data.shape[1]
    print 'input_dim= ', input_dim
    hidden_layer_size=100
    #num_hidden_layer=3
    drop_out_rate=0.5
    batch_size=32
    nb_epoch=1
    #activation_func='relu'
    activation_func='relu'
    print 'hidden_layer_size= ', hidden_layer_size
    #print 'num_hidden_layer= ', num_hidden_layer
    #print 'drop_out_rate= ',drop_out_rate
    print 'batch_size= ',batch_size
    print 'nb_epoch= ',nb_epoch
    print 'activation_func= ',activation_func
    model=keras_denoising_autoencoder_model(input_dim, hidden_layer_size,drop_out_rate,activation_func)
    #model=nn_denoising_autoencoder_model(input_dim,hidden_layer_size,num_hidden_layer,drop_out_rate)
    #sss = StratifiedShuffleSplit(all_label, 1, test_size=0.1, random_state=0)
    
    

    
    start_time=time.time()
    
    #model.fit(labeled_data, labeled_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    model.fit(all_data, all_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    
    
    json_string = model.to_json()
    f=open('my_model_architecture.json', 'w')
    f.write(json_string)
    f.close()
    
    representations = get_output(model, 1, labeled_data)
    print representations
    print representations.shape
    end_time = time.time()
    elapsed_time = end_time - start_time
    print 'training elapsed time: ',elapsed_time
    
    output_dict={}
    output_dict['data']=labeled_data
    output_dict['compressed_data']=representations
    output_dict['label']=labeled_label
    output_dict['label_unique_list']=label_unique_list
    #print labeled_label
    
    with open('compressed_data_label.pickle', 'wb') as handle:
        pickle.dump(output_dict, handle)
    
    
    model.save_weights('my_model_weights.h5', overwrite=True)
    
    model2 = model_from_json(open('my_model_architecture.json').read())
    #model2.compile(loss='mean_squared_error', optimizer='sgd')
    model2.load_weights('my_model_weights.h5')
    representations =  get_output(model2, 1, labeled_data)
    print representations
    print representations.shape
    
    #with open('compressed_data_label.pickle', 'rb') as handle:
    #   b = pickle.load(handle)
    #print b
    #code=get_code(model,autoencoder,labeled_data)
    #print code.shape
    
    '''
    code2=get_output(model, 1, data)
    print code2.shape
    code2=get_output(model, 2, data)
    print code2.shape
    code2=get_output(model, 3, data)
    print code2.shape
    '''
    #score2 = model.evaluate(data[test_index],data[test_index], batch_size,verbose=1)
    #print score2
    
    #print len(model.layers[0].get_weights())
    #print model.layers[0].get_weights()
    #print len(model.layers[0].get_weights()[0])
    #print len(model.layers[0].get_weights()[1])
    
    #model2 = Sequential()
    #model2.add(Dense(input_dim, hidden_layer_size, weights=model.layers[0].get_weights()))
    #model2.add(Activation('tanh'))
    #model2.compile()
    
    #activations = model2.predict(data)
    #print activations
    #activations = get_output(model, 1, data)
    #print activations
    '''
    clf = svm.SVC()
    clf.fit(activations, labels)
    print clf
    print labels
    print clf.predict(activations)
    '''
    #print len(activations)
    #print len(activations[0])
    #print len(data)
    #get_3rd_layer_output = K.function([model.layers[0].input],
    #                              [model.layers[3].get_output(train=False)])
    #layer_output = get_3rd_layer_output([X])[0]
    
    #print model.layers[1].get_weights()
    #print model.layers[2].get_weights()
    #model2 = Sequential()
    #model2.add(Dense(20, 64, weights=model.layers[0].get_weights()))
    #model2.add(Activation('tanh'))

    #activations = model2._predict(X_batch) 

if __name__=='__main__':
    #test_nn_things()
    if 1:
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_1_4_6_7_8_10_16.txt')
        #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/mouse_4_6_7_10.txt', landmark=True)  
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',whitening=True)   
        data=all_data
        #dict_weights={}
        #print(len(all_weights))
        #print(len(labeled_weights))
        #print(len(unlabeled_weights))
        #for i,w in enumerate(unlabeled_weights):
        #    dict_weights[i]=w

        #data=np.float32(data)
        print data.shape
        input_dim = data.shape[1]
        print 'input_dim= ', input_dim
        hidden_layer_size=100
        drop_out_rate=1
        batch_size=32
        epoch_step=10
        max_iter=1000
        activation_func='relu'
        print 'hidden_layer_size= ', hidden_layer_size
        print 'drop_out_rate= ',drop_out_rate
        print 'batch_size= ',batch_size
        print 'epoch_step= ',epoch_step
        print 'activation_func= ',activation_func
        now_iter=0
        #model_name='model/NN100Code3StackLandmark'
        #model_name='model/NN100Code1layerLandmark'
        #model_name='model/NN100Code1layerAllgene'
        #model_name='model/NN100Code3StackAllgene'
       
        #model_name='model/NN100Code1layerAllgeneData010406070810'
        #model_name='model/NN100Code3StackAllgeneData010406070810'
        
        #model_name='model/NN100Code1layerAllgeneData01040607081016unlabeled'
        #model_name='model/NN100Code1layerAllgeneData01040607081016'
        model_name='model/NN100Code3StackAllgeneData01040607081016unlabeled'
        #model_name='model/NN100Code3StackAllgeneData01040607081016'
        
        #model_name='JustATest'
        if now_iter==0:
            model=keras_denoising_autoencoder_model(input_dim, hidden_layer_size,drop_out_rate,activation_func)
            #model=keras_denoising_autoencoder_model_1layer(input_dim, hidden_layer_size,drop_out_rate,activation_func)
            json_string = model.to_json()
            f=open(model_name+'.json', 'w')
            f.write(json_string)
            model.save_weights(model_name+'_'+str(0)+'.h5', overwrite=True)
            f.close()
        else:
            model = model_from_json(open(model_name+'.json').read())
            model.load_weights(model_name+'_'+str(now_iter)+'.h5')
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=sgd)
        while now_iter<max_iter:
            print 'now_iter= ',now_iter
            #model.fit(labeled_data, labeled_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
            #model.fit(all_data, all_data, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            model.fit(unlabeled_data, unlabeled_data, sample_weight=unlabeled_weights, batch_size=batch_size, nb_epoch=epoch_step,verbose=1)
            now_iter+=epoch_step
            model.save_weights(model_name+'_'+str(now_iter)+'.h5', overwrite=True)
        
