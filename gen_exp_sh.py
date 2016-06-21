
ref_gene_file='../cluster_genes.txt'
ppi_tf_cluster_file='../ppi_tf_merge_cluster.txt'

#epi_tf_ref_gene_tuple=ppi_tf_cluster_file,ref_gene_file

ppi_tf_files=['',ppi_tf_cluster_file]
#ppi_tf_files=['']

activation=['tanh', 'relu','sigmoid','linear']
max_iteration=100
step_size=10
middle_layer_size=696
hidden_layer_size=100
#hidden_layer_size=696

#validation_percent=0.5
validation_percent=0
sm=1
sn=1
gn=1
stc=1
architecture=['1layer', '3layer']
#architecture=['1layer']
#vct=0
vct=8
seed=0
count=0

gene_train=1

for fp in ppi_tf_files:
    for act in activation:
        for archi in architecture:
            count+=1
            hls=hidden_layer_size
            if archi=='1layer' and fp=='':
                #print 'Dense 1layer 696!!!!!!!!!!!'
                hls=696
            log_file='Dense'
            if fp!='':
                log_file='PPITF'
            log_file+='_'+archi
            if stc==1:
                log_file+='_classifier'
            else:
                log_file+='_autoencoder'
            
            log_file+='_hls'+str(hls)
            log_file+='_mls'+str(middle_layer_size)
            log_file+='_SN'+str(sn)
            log_file+='_GN'+str(gn)
            log_file+='_sm'+str(sm)
            log_file+='_seed'+str(seed)
            log_file+='_vct'+str(vct)
            log_file+='_VP'+str(validation_percent)
            log_file+='_'+str(act)
            log_file+='.log'
            if gene_train==1:
                print 'echo '+str(count)
                print 'python deep_net.py \\'
                if fp!='':
                    print '-p '+fp+' -r '+ref_gene_file+' \\'
                else:
                    print ' -r '+ref_gene_file+' \\'

                print '-m '+ str(max_iteration)+' -hls '+str(hls)+" -mls " + str(middle_layer_size)+' -s '+str(step_size)+ ' \\'
                print '-a '+archi+" \\"
                print '-vp '+ str(validation_percent)+" \\"
                print '-vct '+ str(vct)+" \\"
                print '-seed '+ str(seed)+" \\"
                print '-act '+act + ' \\'
                print '-stc '+str(stc)+' \\'
                print '-sm '+ str(sm)+' \\'
                print '-sn '+ str(sn)+' \\'
                print '-gn '+ str(gn)+' \\'
                print '> log/'+log_file
            else:
                
                #python gen_nn_cluster_sh.py -mn 1layer_SN1_GN1_BS32_PPITF_linear -m 2000 -s 100 -si 0 -t labeled  > run_nn_clustering_2.sh
                #sh run_nn_clustering_2.sh 
#-rw-r--r-- 1 z8 users       667 Jun 14 02:51 model/1layer_SN1_GN1_BS32_hls100_mls100_classifier_tanh.json
#-rw-r--r-- 1 z8 users   1741328 Jun 14 02:44 model/1layer_SN1_GN1_BS32_hls100_mls100_PPITF_classifier_linear_0.pickle
#-rw-r--r-- 1 z8 users   1737479 Jun 14 02:44 model/1layer_SN1_GN1_BS32_hls100_mls100_PPITF_classifier_linear_100.pickle

#-rw-r--r-- 1 z8 users   2401445 Jun 19 13:25 3layer_SN1_GN1_BS32_hls100_mls696_seed0_PPITF_classifier_vct4_linear_100.pickle
#-rw-r--r-- 1 z8 users   2400983 Jun 19 13:25 3layer_SN1_GN1_BS32_hls100_mls696_seed0_PPITF_classifier_vct4_linear_90.pickle
                mn=archi
                mn+='_SN'+str(sn)
                mn+='_GN'+str(gn)
                mn+='_BS32'
                mn+='_hls'+str(hls)
                mn+='_mls'+str(middle_layer_size)
                mn+='_seed'+ str(seed)
                if fp!='':
                    mn+='_PPITF'
                mn+='_classifier'
                mn+='_vct'+ str(vct)
                mn+='_'+act
                if validation_percent>0:
                    print 'python gen_nn_cluster_sh.py -mn '+mn+' -m '+str(max_iteration)+' -s '+str(step_size)+ ' -si 0 '+ ' -t labeled -vdc 1  > run_nn_clustering_temp.sh'
                else:
                    print 'python gen_nn_cluster_sh.py -mn '+mn+' -m '+str(max_iteration)+' -s '+str(step_size)+ ' -si 0 '+ ' -seed '+str(seed)+' -vct '+str(vct)+ ' -t labeled > run_nn_clustering_temp.sh'

                print 'sh run_nn_clustering_temp.sh'
