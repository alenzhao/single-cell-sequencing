import argparse

#python pca_clustering.py -nn 1layer_SN0_GN1_BS32_linear -t labeled  \
#	-ni 100\
#	> clustering_log/pca_ori_all_Cdefault.log 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn',"--model_name", help="specify the model_name, if not specified then a default model_name will be used")
    parser.add_argument('-t',"--transform_data", help="specify the transform_dataset",choices=['all','labeled','unlabeled'],default='all')
    parser.add_argument('-m',"--max_iteration", help="specify the maximum training iteration, default is 100", type=int,default=100)
    parser.add_argument('-s',"--step_size", help="specify the training step size, default is 10", type=int,default=10)
    parser.add_argument('-si',"--starting_iteration", help="specify the starting iteration , default is 0", type=int,default=0)
    parser.add_argument('-c',"--n_cluster", help="specify the number of clusters, default is the number of labels", type=int,default=0)
    parser.add_argument('-vdc',"--validation_data_classifier", help="specify whether to use validation data from pickle, default is 0",choices=[0,1], type=int,default=0)
    #parser.add_argument('-r',"--reference_gene_file", help="specify the file that contatins the genes to be kept, if not specified then all the genes in training data will be used")
    parser.add_argument('-vct',"--validation_cell_types", help="specify the number of validation cell types, default is 0", type=int,default=0)
    parser.add_argument('-seed',"--random_seed", help="specify the random seed, default is 0", type=int,default=0)
    
    
    args=parser.parse_args()
    step=args.step_size
    start=args.starting_iteration
    end=args.max_iteration
    splits=args.model_name.split('_')
    while start<=end:

        print 'python pca_clustering.py -nn '+args.model_name +' -t '+args.transform_data+" \\"
        print '-ni '+str(start)+" \\"
        print '-seed '+str(args.random_seed)+' \\'
        if args.n_cluster>0:
            print '-c '+str(args.n_cluster)+' \\'
        if splits[1][-1]=='0':
            print '-sn 0 \\'
        if args.validation_data_classifier:
            print '-vdc 1 \\'
        if args.validation_cell_types>0:
            print '-vct '+str(args.validation_cell_types)+' \\'
        if 'PPITF' in args.model_name:
            print '-r ../cluster_genes.txt \\'
        print '> clustering_log/nn_'+args.model_name+"_"+args.transform_data+"_c"+str(args.n_cluster)+"_"+str(start)+'.log'
        start+=step
