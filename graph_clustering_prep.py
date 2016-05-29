from collections import *

def read_ppi_net(filename):
    network_fn=filename
    net_lines=open(network_fn).readlines() 
    G=nx.DiGraph()
    for line in net_lines:#[:10]:
        splits= line.replace('\n','').split('\t')
        a=splits[0].lower()
        b=splits[2].lower()
        score=float(splits[3])
        if splits[1]=='(pp)':
            G.add_edge(a,b,weight=score)
            G.add_edge(b,a,weight=score)
        if splits[1]=='(ptm)':
            G.add_edge(a,b,weight=score)
    #elarge=[(u,v,d) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
    #print elarge[:10]
    
    #get integer number of label
    #G=nx.convert_node_labels_to_integers(G)
    return G
def read_tf_net(filename):
    network_fn=filename
    net_lines=open(network_fn).readlines() 
    G=nx.DiGraph()
    for line in net_lines:#[:10]:
        splits= line.replace('\n','').split('\t')
        a=splits[0].lower()
        b=splits[1].lower()
        G.add_edge(a,b)
    return G

def gen_ppi_mcl_abc(filename, output_file,ref_file=None,ref_gene=False):
    ref_gene_list=[]
    if ref_gene:
        ref_lines=open(ref_file).readlines()
        for line in ref_lines[3:]:
            ref_gene_list.append(line.split('\t')[0].lower())
    network_fn=filename
    net_lines=open(network_fn).readlines() 
    f_output=open(output_file,'w')
    for line in net_lines:#[:10]:
        splits= line.replace('\n','').split('\t')
        a=splits[0].lower()
        b=splits[2].lower()
        score=float(splits[3])
        if splits[1]=='(pd)':
            continue
        #if  splits[1]=='(ptm)':
        #    continue
        if ref_gene and a not in ref_gene_list:
            continue
        if ref_gene and b not in ref_gene_list:
            continue
        f_output.write(a+"\t"+b+'\t'+str(score)+"\n")
def gen_tf_groups(filename,output_file,ref_file=None,ref_gene=False):
    ref_gene_list=[]
    if ref_gene:
        ref_lines=open(ref_file).readlines()
        for line in ref_lines[3:]:
            ref_gene_list.append(line.split('\t')[0].lower())
    network_fn=filename
    net_lines=open(network_fn).readlines() 
    f_output=open(output_file,'w')
    tf_groups=defaultdict(lambda:[])
    for line in net_lines[1:]:
        splits= line.replace('\n','').split('\t')
        a=splits[0].lower()
        b=splits[1].lower()
        #if a not in ref_gene_list:
        #    continue
        if ref_gene and b not in ref_gene_list:
            continue
        tf_groups[a].append(b)   
    
    s=0
    ma=1
    mi=100000
    for key,val in tf_groups.items():
        l=len(val)
        s+=l   
        ma=max(ma,l)
        mi=min(mi,l)
        f_output.write('TF: '+key+"\t"+"\t".join(val)+"\n")
    #print s/len(tf_groups.keys())
    #print ma
    #print mi
    #f_output.write(a+"\t"+b+'\t'+str(score)+"\n")
def merge_ppi_tf_clusters(ppi_file,tf_file,out_cluster_file, out_gene_file):
    ppi_lines=open(ppi_file).readlines() 
    tf_lines=open(tf_file).readlines() 
    f_out_cluster=open(out_cluster_file,'w')
    f_out_gene=open(out_gene_file,'w')
    gene_set=set()
    clusters={}
    for line in tf_lines:
        splits= line.replace('\n','').split('\t')
        gn=splits[0]
        ge=splits[1:]
        gene_set|=set(ge)
        clusters[gn]=ge
    for index,line in enumerate(ppi_lines):
        splits= line.replace('\n','').split('\t')
        gn='ppi_'+str(index+1)
        ge=splits
        gene_set|=set(ge)
        clusters[gn]=ge
        #print gn
        #print ge
    for key,val in clusters.items():
        f_out_cluster.write(key+"\t"+'\t'.join(val)+"\n")
    print '#genes=', len(gene_set)
    gene_list=list(gene_set)
    #print gene_list
    f_out_gene.write('\t'.join(gene_list))
    print len(gene_set)
    
    

if __name__=='__main__':
    
    #gen_ppi_mcl_abc('ppi_ptm_pd_hgnc.txt','ppi.abc','data/TPM_mouse_1_4_6_7_8_10_16.txt',ref_gene=False)
    #gen_ppi_mcl_abc('ppi_ptm_pd_hgnc.txt','ppi_pd_ptm_ref.abc','data/TPM_mouse_1_4_6_7_8_10_16.txt',ref_gene=True)
    #gen_ppi_mcl_abc('ppi_ptm_pd_hgnc.txt','ppi_ptm_ref.abc','data/TPM_mouse_1_4_6_7_8_10_16.txt',ref_gene=True)
    #gen_tf_groups('encode_100_tf_gene.txt','tf_group_ref.txt','data/TPM_mouse_1_4_6_7_8_10_16.txt',ref_gene=True)
    #gen_tf_groups('encode_100_tf_gene.txt','tf_group.txt')

    merge_ppi_tf_clusters('ppi_ptm_ref_clusterone_min11_d03088.txt','tf_group_ref.txt','ppi_tf_merge_cluster.txt', 'cluster_genes.txt')
    
