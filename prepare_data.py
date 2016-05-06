from os import listdir
from os.path import isfile, join
from collections import defaultdict
import numpy as np
def load_gene_length():
    gene_length_dict={}
    lines=open('gene_length.txt').readlines()
    #print data_lines
    for line in lines[1:]:
        sp=line.replace('\n','').split('\t')
        #print sp
        length=int(sp[1])-int(sp[0])+1
        if not sp[2].startswith('\n'):
            gene=sp[2].lower()
            #print gene, length
            gene_length_dict[gene]=length
    return gene_length_dict
def parse_data(type,index,seq):
    dir=type+'/'+index+'/'
    #gene_length_dict_keys=load_gene_length().keys()
    if type=='mouse' and index=='1':
        expr_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith('.txt') and f.startswith('mgisymb')]
        out_ID='data/'+type+'_'+index
        out_label=open(out_ID+'_label.txt','w')
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        sample_ID=['Sample ID']
        sample_type=['type']
        data=[]
        gene=[]
        for index,fn in enumerate(expr_files):
            print fn
            fn_splits=fn.split('_')
            type=fn_splits[3]
            data_lines1=open(dir+fn).readlines()
            sps=data_lines1[0].replace('\n','').split('\t')
            sample_ID.extend(sps[1:])
            gene_append=False
            #sample_type.extend(['mES'+type]*len(sps[1:]))
            #sample_type.extend(['mES']*len(sps[1:]))
            sample_type.extend(['None']*len(sps[1:]))
            for i,line in enumerate(data_lines1[1:]):
                line=line.replace('\n','').split('\t')
                #if line[0] not in gene_length_dict_keys:
                #   continue
                if not gene_append and index == 0:
                    data.append([line[0]])
                data[i].extend(line[1:])
            gene_append=True
            
        out_data.write('\t'.join(sample_ID)+'\n')
        for d in data:
            out_data.write('\t'.join(d)+"\n")
        #sample_ID.append('\n')
        #sample_type.append('\n')
        out_label.write('\t'.join(sample_ID)+"\n")
        out_label.write('\t'.join(sample_type)+"\n")
        
    if type=='mouse' and index=='4':
        data=[]
        data_lines=open(dir+'GSE71585_RefSeq_RPKM.csv').readlines()
        label_lines=open(dir+'sample_label.csv').readlines()
        out_ID='data/'+type+'_'+index
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        out_label=open(out_ID+'_label.txt','w')
        for i,line in enumerate(data_lines):
            data_lines[i]='\t'.join(line.replace('\n','').replace('"','').split(','))
        sample_ID=['Sample ID']
        sample_type=['type']
        sample_type2=['type2']
        for i,line in enumerate(label_lines[1:]):
            splits=line.replace('\n','').split('\t')
            sample_ID.append(splits[1])
            sample_type.append(splits[0].split(' ')[0])
            sample_type2.append(splits[0])
        #sample_ID.append('\n')
        #sample_type.append('\n')
        out_data.write('\n'.join(data_lines))
        out_label.write('\t'.join(sample_ID)+"\n")
        #out_label.write('\t'.join(sample_type))
        out_label.write('\t'.join(sample_type2)+"\n")
    if type=='mouse' and index=='6':
        data=[]
        data_lines=open(dir+'GSE59739_DataTable.txt').readlines()
        label_lines=open(dir+'sample_label.csv').readlines()
        out_ID='data/'+type+'_'+index
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        out_label=open(out_ID+'_label.txt','w')
        for i,line in enumerate(data_lines):
            data_lines[i]=line.replace('\n','')
        for i,line in enumerate(label_lines):
            label_lines[i]=line.replace('\n','')
        data.append(data_lines[0])
        data.extend(data_lines[5:])
        out_data.write('\n'.join(data))
        out_label.write('\n'.join(label_lines))
    if type=='mouse' and index=='7':
        data=[]
        data_lines=open(dir+'GSE41265_allGenesTPM.txt').readlines()
        #label_lines=open(dir+'sample_label.csv').readlines()
        out_ID='data/'+type+'_'+index
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        out_label=open(out_ID+'_label.txt','w')
        for i,line in enumerate(data_lines):
            data_lines[i]='\t'.join(line.replace('\n','').split('\t')[:-3])
        sample_ID=['Sample ID']
        sample_type=['type']
        for i,line in enumerate(data_lines[0].split('\t')[1:]):
            #splits=line.replace('\n','').split('\t')
            sample_ID.append(line)
            if line.startswith('S'):
                sample_type.append('BMDC')
        #sample_ID.append('\n')
        #sample_type.append('\n')
        out_data.write('\n'.join(data_lines))
        out_label.write('\t'.join(sample_ID)+"\n")
        out_label.write('\t'.join(sample_type)+"\n")
    if type=='mouse' and index=='8':
        #data=[]
        dir+='1036556/'
        expr_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith('.txt')]
        out_ID='data/'+type+'_'+index
        out_label=open(out_ID+'_label.txt','w')
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        sample_ID=['Sample ID']
        sample_type1=['type1']
        sample_type2=['type2']
        data=[]
        gene=[]
        gene_append=False
        for fn in expr_files:
            print fn
            fn_splits=fn.split('_')
            sample_ID.append(fn_splits[0])
            #sample_type.append('_'.join(fn_splits[1:-2]))
            type=fn_splits[1]
            if type.startswith('E'):
                sample_type1.append("ES")
            else:
                sample_type1.append("PrE")
            sample_type2.append(type)
            data_lines1=open(dir+fn).readlines()
            for i,line in enumerate(data_lines1[1:]):
                line=line.replace('\n','').split('\t')
                if not gene_append:
                    data.append([line[1]])
                data[i].append(line[2])
            gene_append=True
        out_data.write('\t'.join(sample_ID)+'\n')
        for d in data:
            out_data.write('\t'.join(d)+"\n")
        #sample_ID.append('\n')
        #sample_type1.append('\n')
        #sample_type2.append('\n')
        out_label.write('\t'.join(sample_ID)+'\n')
        out_label.write('\t'.join(sample_type1)+'\n')
        out_label.write('\t'.join(sample_type2)+'\n')
    if type=='mouse' and index=='10':
        #data=[]
        dir+='45719/'
        expr_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith('.txt')]
        out_ID='data/'+type+'_'+index
        out_label=open(out_ID+'_label.txt','w')
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        sample_ID=['Sample ID']
        sample_type=['type']
        data=[]
        gene=[]
        gene_append=False
        for fn in expr_files:
            print fn
            fn_splits=fn.split('_')
            sample_ID.append(fn_splits[0])
            #sample_type.append('_'.join(fn_splits[1:-2]))
            if fn_splits[1].startswith('zy'):
                sample_type.append("zy")
            else:
                sample_type.append(fn_splits[1])
            data_lines1=open(dir+fn).readlines()
            for i,line in enumerate(data_lines1[1:]):
                line=line.replace('\n','').split('\t')
                if not gene_append:
                    data.append([line[0]])
                data[i].append(line[2])
            gene_append=True
        out_data.write('\t'.join(sample_ID)+'\n')
        for d in data:
            out_data.write('\t'.join(d)+"\n")
        #sample_ID.append('\n')
        #sample_type.append('\n')
        out_label.write('\t'.join(sample_ID)+'\n')
        out_label.write('\t'.join(sample_type)+'\n')

    if type=='mouse' and index=='16':
        data=[]
        data_lines=open(dir+'GSE48968_allgenesTPM_GSM1189042_GSM1190902.txt').readlines()
        out_ID='data/'+type+'_'+index
        for i,line in enumerate(data_lines):
            data_lines[i]=line.replace('\n','')
        sample_ID=['Sample ID']
        sample_type=['type']
        #samples=data_lines[0].split('\t')
        #exps=data_lines[1].split('\t')
        #print samples
        for i,line in enumerate(data_lines[0].split('\t')):
            sample_ID.append(line)
            print 'sample_ID: ',line
            #sample_type.append('BMDC')
            sample_type.append('None')
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        out_label=open(out_ID+'_label.txt','w')
        out_data.write('\t'.join(sample_ID)+'\n')
        out_data.write('\n'.join(data_lines[1:]))
        out_label.write('\t'.join(sample_ID)+'\n')
        out_label.write('\t'.join(sample_type)+'\n')
        
'''
if type=='human' and index=='1':
        data=[]
        data_lines=open(dir+'nsmb.2660-S2-modified.csv').readlines()
        #label_lines=open(dir+'sample_label.csv').readlines()
        out_ID='data/'+type+'_'+index
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        out_label=open(out_ID+'_label.txt','w')
        for i,line in enumerate(data_lines):
            data_lines[i]=line.replace('\n','')
        sample_ID=['Sample ID']
        sample_type=['type']
        for i,line in enumerate(data_lines[0].split('\t')[1:]):
            #splits=line.replace('\n','').split('\t')
            sample_ID.append(line)
            sample_type.append(line.split('#')[0])
            #if line.startswith('S'):
            #   sample_type.append('BDMC')
            #elif line.startswith('P'):
            #   sample_type.append('10000 population')
        sample_ID.append('\n')
        sample_type.append('\n')
        out_data.write('\n'.join(data_lines))
        out_label.write('\t'.join(sample_ID))
        out_label.write('\t'.join(sample_type))
    if 0 and type=='human' and index=='3':
        dir='human/Human and Mouse Data 3/GSE38495/'
        expr_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith('.txt')]
        out_ID='data/'+type+'_'+index
        #out_label=open(out_ID+'_label.txt','w')
        out_data=open(out_ID+'_expr_'+seq+'.txt','w')
        sample_ID=['Sample ID']
        #sample_type1=['type1']
        #sample_type2=['type2']
        data=[]
        gene=[]
        gene_append=False
        for fn in expr_files:
            print fn
            fn_splits=fn.split('_')
            sample_ID.append(fn_splits[0])
            #sample_type.append('_'.join(fn_splits[1:-2]))
            type=fn_splits[1]
        #   if type.startswith('E'):
        #      sample_type1.append("ES")
        #   else:
        #      sample_type1.append("PrE")
        #   sample_type2.append(type)
            data_lines1=open(dir+fn).readlines()
            for i,line in enumerate(data_lines1[2:]):
                line=line.replace('\n','').split('\t')
                print line
                if not gene_append:
                    data.append([line[0]])
                data[i].append(line[2])
            gene_append=True
        out_data.write('\t'.join(sample_ID)+'\n')
        for d in data:
            out_data.write('\t'.join(d)+"\n")
        sample_ID.append('\n')
        #sample_type1.append('\n')
        #sample_type2.append('\n')
        out_label.write('\t'.join(sample_ID))
        #out_label.write('\t'.join(sample_type1))
        #out_label.write('\t'.join(sample_type2))'''
def TPM_normalize_data(type,data_file_name,label_file_name, output_file_name,uselabel=True):
    #RPKM or FPKM to TPM and linearly normalize each gene to 0-1 scale 
    print 'TPM normalizing:',data_file_name
    gene_length_dict=load_gene_length()
    sample_label_dict=defaultdict(lambda: None)
    data_file_lines=open(data_file_name).readlines()
    label_file_lines=open(label_file_name).readlines()
    IDs=label_file_lines[0].replace('\n','').split('\t')[1:]
    if uselabel:
        sample_label=label_file_lines[1].replace('\n','').split('\t')[1:]
        for index,ID in enumerate(IDs):
            sample_label_dict[ID]=sample_label[index]
    print 'number of labels: ',len(sample_label_dict.keys())
    IDs_expr=data_file_lines[0].replace('\n','').split('\t')[1:]
    #print 'number of samples: ', len(IDs_expr)
    Expr_matrix=[]
    genes=[]
    genes_set=set()
    gene_feature_dict=defaultdict(lambda: [])
    for line in data_file_lines[1:]:
        splits = line.replace('\n','').split('\t')
        gene=splits[0].lower()
        row=map(float,splits[1:])
        #if max(row)-min(row)==0:
        #   continue
        #Expr_matrix.append(row)
        #genes.append(gene)
        genes_set.add(gene)
        gene_feature_dict[gene].append(row)
        
    for gene in genes_set:
        row=gene_feature_dict[gene][0]
        for i in range(1,len(gene_feature_dict[gene])):
            row=[a + b for a, b in zip(row, gene_feature_dict[gene][i])]
        row=[x / len(gene_feature_dict[gene]) for x in row]
        Expr_matrix.append(row)
        genes.append(gene)
    Expr_matrix=np.array(Expr_matrix)
    
    
    #print Expr_matrix
    nrow=Expr_matrix.shape[0]
    ncol=Expr_matrix.shape[1]
    #print "nrow=", nrow
    #print "ncol=", ncol
    good_genes=[]
    if type =='RPKM' or type == 'FPKM':
        for k in range(ncol):
            s=sum(Expr_matrix[:,k])
            Expr_matrix[:,k]=Expr_matrix[:,k]/s *1000000
    if type == 'RAW':
        for k in range(nrow):
            if genes[k] not in gene_length_dict.keys():
                print 'skip no length genes: ',genes[k]
            else:
                good_genes.append(k)
                Expr_matrix[k,:]=Expr_matrix[k,:]/gene_length_dict[genes[k]]
        Expr_matrix=Expr_matrix[good_genes]
        genes=[genes[x] for x in good_genes]
        nrow=Expr_matrix.shape[0]
        ncol=Expr_matrix.shape[1]
        for k in range(ncol):
            s=sum(Expr_matrix[:,k])
            Expr_matrix[:,k]=Expr_matrix[:,k]/s *1000000
    '''
    for k in range(nrow):
        row=Expr_matrix[k,:]
        upper=max(row)
        lower=min(row)
        ran=upper-lower
        if ran>0:
            Expr_matrix[k,:]=(Expr_matrix[k,:]-lower)/ran
    '''
    #print Expr_matrix
    IDs_expr=data_file_lines[0].replace('\n','').split('\t')[1:]
    f_output=open(output_file_name,'w')
    f_output.write('Sample')
    for ID in IDs_expr:
        #print ID, sample_label_dict[ID]
        f_output.write('\t'+ID)
    f_output.write('\nLabel')
    for ID in IDs_expr:
        #print ID, sample_label_dict[ID]
        f_output.write('\t'+str(sample_label_dict[ID]))
    f_output.write('\n')
    Expr_matrix=Expr_matrix.tolist()
    #print '\t'.join(Expr_matrix[k])
    for k in range(nrow):
        row=map(str,Expr_matrix[k])
        f_output.write(genes[k]+'\t')
        f_output.write('\t'.join(row)+'\n')
    f_output.close()
def integrate_datasets(filenames,output_file_name):
    gene_dict=defaultdict(lambda: 0)
    file_lines=[]
    for fn in filenames:
        file_lines.append(open(fn).readlines())
    for lines in file_lines:
        for index, line in enumerate(lines):
            splits=line.split('\t')
            gene=splits[0].lower()
            gene_dict[gene]+=1
    gene_output_dict=defaultdict(lambda: [])
    for lines in file_lines:
        l=float(len(lines[0].split('\t'))-1)
        gene_output_dict['weight'].extend([str(1/l)]*int(l))
        for index, line in enumerate(lines):
            splits=line.replace('\n','').split('\t')
            gene=splits[0].lower()
            if gene_dict[gene] < len(filenames):
                continue
            gene_output_dict[gene].extend(splits[1:])
            #if len(gene_output_dict[gene])>350:
            #   print gene, len(gene_output_dict[gene])
    
    #print len(gene_output_dict['label'])
    #print len(gene_output_dict['sample'])
    #print len(gene_output_dict['weight'])
    #print gene_output_dict['sample'].index('')
    
    f_output=open(output_file_name,'w')
    f_output.write('Sample'+'\t'+'\t'.join(gene_output_dict['sample'])+'\n')
    f_output.write('Label'+'\t'+'\t'.join(gene_output_dict['label'])+'\n')
    f_output.write('Weight'+'\t'+'\t'.join(gene_output_dict['weight'])+'\n')
    remain=0
    skip=0
    for key,val in gene_output_dict.items():
        if key=='sample' or key=='label' or key =='weight':
            continue
        row=map(float,val)
        if max(row)-min(row)==0:
            print "skip all 0 intersection gene: "+key 
            skip+=1
            continue
        remain+=1
        f_output.write(key+'\t'+'\t'.join(gene_output_dict[key])+'\n')
    print 'remain: ',remain
    print 'skip: ',skip
    #print gene_dict
    f_output.close()
    
if __name__=="__main__":
    
    parse_data('mouse','6','RPKM')
    parse_data('mouse','4','RPKM')
    parse_data('mouse','7','TPM')
    parse_data('mouse','10','RPKM')
    parse_data('mouse','8','FPKM')
    parse_data('human','1','RPKM')
    parse_data('mouse','1','RAW')
    parse_data('mouse','16','TPM')
    
    TPM_normalize_data('RPKM','data/mouse_10_expr_RPKM.txt','data/mouse_10_label.txt','data/mouse_10_TPMnormalized_expr.txt')
    TPM_normalize_data('RPKM','data/mouse_4_expr_RPKM.txt','data/mouse_4_label.txt','data/mouse_4_TPMnormalized_expr.txt',False)
    TPM_normalize_data('RPKM','data/mouse_6_expr_RPKM.txt','data/mouse_6_label.txt','data/mouse_6_TPMnormalized_expr.txt',False)
    TPM_normalize_data('TPM','data/mouse_7_expr_TPM.txt','data/mouse_7_label.txt','data/mouse_7_TPMnormalized_expr.txt')
    TPM_normalize_data('FPKM','data/mouse_8_expr_FPKM.txt','data/mouse_8_label.txt','data/mouse_8_TPMnormalized_expr.txt')
    TPM_normalize_data('RAW','data/mouse_1_expr_RAW.txt','data/mouse_1_label.txt','data/mouse_1_TPMnormalized_expr.txt')
    TPM_normalize_data('TPM','data/mouse_16_expr_TPM.txt','data/mouse_16_label.txt','data/mouse_16_TPMnormalized_expr.txt')
    integrate_set=[]
    #integrate_set=['data/mouse_10_normalized_expr.txt','data/mouse_4_normalized_expr.txt','data/mouse_7_normalized_expr.txt','data/mouse_6_normalized_expr.txt']
    
    integrate_set.append('data/mouse_4_TPMnormalized_expr.txt')
    integrate_set.append('data/mouse_6_TPMnormalized_expr.txt')
    integrate_set.append('data/mouse_7_TPMnormalized_expr.txt')
    integrate_set.append('data/mouse_8_TPMnormalized_expr.txt')
    integrate_set.append('data/mouse_10_TPMnormalized_expr.txt')
    integrate_set.append('data/mouse_1_TPMnormalized_expr.txt')
    integrate_set.append('data/mouse_16_TPMnormalized_expr.txt')
    integrate_datasets(integrate_set, "data/TPM_mouse_1_4_6_7_8_10_16.txt")
    
    #integrate_set=['data/mouse_10_normalized_expr.txt','data/mouse_7_normalized_expr.txt']
    #integrate_datasets(integrate_set, "data/mouse_7_10.txt")
    #generate_gene_map('data/mouse_8_normalized_expr.txt','data/mouse_8_gene_map.txt')
