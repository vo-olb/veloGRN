import numpy as np
import pandas as pd
import torch

def RNA_prep(args):
    #TODO: 质控、norm/log、hvg、cluster、按伪时间排序，输出n_cell*n_gene
    expr = pd.read_csv(args.expr_path, index_col=0) # n_gene*n_cell
    expr = expr.T.reset_index().rename(columns={'index':'cell'})

    args.preprocess = args.preprocess.split(',')
    if args.preprocess == ['None']:
        pass
    else:
        raise NotImplementedError
    
    time_info = pd.read_csv(args.time_info)
    time_info = time_info.rename(columns={'Unnamed: 0':'cell'})
    expr = pd.merge(expr, time_info, on='cell', how='inner').sort_values(by='PseudoTime')
    return expr.drop(columns=['cell','PseudoTime']).reset_index(drop=True)

def Network_prep(args):
    #TODO: 用给定网络或者用ATAC生成网络
    prior_network = pd.read_csv(args.network_path)
    prior_network = prior_network[['from','to']]
    prior_network = prior_network.rename(columns={'from':'Gene1','to':'Gene2'})
    prior_network = prior_network.drop_duplicates(keep='first').reset_index(drop=True)
    return prior_network

def preprocess(args):
    #加载表达矩阵
    expr = RNA_prep(args)  # 输出n_cell*n_gene
    gene_list = expr.columns

    #加载初始网络，并将初始网络中的gene都限制在基因表达矩阵中
    prior_network = Network_prep(args)
    prior_network = prior_network.loc[prior_network['Gene1'].isin(gene_list) & prior_network['Gene2'].isin(gene_list), :].reset_index(drop=True)

    #将初始网络转换为edge_list的形式
    idx_GeneName_map = pd.DataFrame({'idx': range(len(gene_list))}, index=gene_list)
    edgelist = [idx_GeneName_map.loc[prior_network['Gene1'].tolist(), 'idx'].tolist(), 
                idx_GeneName_map.loc[prior_network['Gene2'].tolist(), 'idx'].tolist()]
    edgelist = torch.from_numpy(np.array(edgelist))

    return expr, edgelist, idx_GeneName_map
