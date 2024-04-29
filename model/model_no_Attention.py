import os
import math
from math import sqrt
import numpy as np
from logging import Logger
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    to_dense_adj,
    dense_to_sparse,
    degree
)
import torch.nn.functional as F
from torch_geometric.data import Data
import pytorch_warmup as warmup
import pandas as pd
from utils import EarlyPrec,EarlyStopping,computeScores
import matplotlib.pyplot as plt
from data import Dataset_MTS
import warnings
warnings.filterwarnings('ignore')


from torch.nn.modules.loss import _Loss

class myRMSELoss(_Loss):
    def __init__(self, mask: bool = False) -> None:
        super().__init__()
        self.mask = mask
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mask:
            input[target == 0] = torch.nan
        return torch.sqrt(torch.nanmean((input-target)**2))
        
class myNRMSELoss(_Loss):
    def __init__(self, mask: bool = False) -> None:
        super().__init__()
        self.mask = mask
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mask:
            input[target == 0] = torch.nan
        a = torch.sqrt(torch.nanmean((input-target)**2, dim=(0,1)))
        b = torch.sqrt(torch.nanmean(target**2, dim=(0,1)) / torch.nanmean(target, dim=(0,1))**2)
        res = a/b
        res[torch.isinf(res)] = torch.nan
        return torch.nanmean(res)

def metric(pred, true, mask):
    def MAE(pred, true):
        return np.nanmean(np.abs(pred-true))

    def MSE(pred, true):
        return np.nanmean((pred-true)**2)

    def NRMSE(pred, true):
        a = np.sqrt(np.nanmean((pred-true)**2, axis=(0,1)))
        b = np.sqrt(np.nanmean(true**2, axis=(0,1)) / np.nanmean(true, axis=(0,1))**2)
        res = a/b
        res[np.isinf(res)] = np.nan
        return np.nanmean(res)

    if mask:
        pred[true == 0] = np.nan
    mae = MAE(pred, true)
    rmse = np.sqrt(MSE(pred, true))
    nrmse = NRMSE(pred, true)
    return mae,rmse,nrmse

class DSW_embedding(nn.Module): # 将t个时间点分成seg_num个长度为seg_len的segment变换成长度为d_model
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> b d seg_num seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        return x_embed  # batch_size, gene_num, seg_num, d_model

class PositionalEncoding(nn.Module):
    def __init__(self, seg_num, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seg_num, d_model)
        position = torch.arange(0, seg_num, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe)
    
    def forward(self, x):
        return x + self.pe

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels,flow: str='source_to_target',**kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', flow)
        super(GCNConv, self).__init__(**kwargs)

        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):   # 对x变换，然后传递信息
        x = self.lin(x)

        row, col = edge_index
        in_deg, out_deg = degree(col, x.size(0), dtype=x.dtype), degree(row, x.size(0), dtype=x.dtype)
        in_deg_inv_sqrt, out_deg_inv_sqrt = in_deg.pow(-0.5), out_deg.pow(-0.5)
        norm = out_deg_inv_sqrt[row] * in_deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j

class Graph_encoder(nn.Module):
    def __init__(self,input_dim,output_dim,dropout):
        super(Graph_encoder, self).__init__()
        
        self.act = nn.GELU()
        self.layers = nn.ModuleList([])
        dims = [input_dim, input_dim]  # 2 layers are used
        for l in range(len(dims)):
            last_dim = input_dim if l < len(dims) - 1 else output_dim
            self.layers.append(nn.ModuleList([
                # in-coming
                GCNConv(dims[l], dims[l], flow='source_to_target'),
                # out-going
                GCNConv(dims[l], dims[l], flow='target_to_source'),
                nn.Sequential(
                    nn.Linear(dims[l] * 2, input_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, last_dim),
                ),
            ]))
    
    def forward(self,x,edge_index):
        for gcn_in, gcn_out, ffn in self.layers:
            x_in = gcn_in(x, edge_index)
            x_out = gcn_out(x, edge_index)
            x = ffn(torch.cat((self.act(x_in), self.act(x_out)), 1))

        return x

class Graph(nn.Module):
    def __init__(self, d_model, seg_num, dropout) -> None:
        super(Graph,self).__init__()
        
        #用于提取空间信息
        input_dim = seg_num * d_model
        self.encoder = Graph_encoder(input_dim, input_dim, dropout=dropout)
        self.graph_encoder = DeepGraphInfomax(hidden_channels=input_dim,
                                              encoder=self.encoder,
                                              summary=self.__summary,
                                              corruption=self.__corruption)
    @staticmethod
    def __corruption(x,edge_index) -> Data:
        x = x[torch.randperm(x.size(0))]
        edge_index = edge_index
        return (x , edge_index)

    @staticmethod
    def __summary(z, *args, **kwargs) -> torch.Tensor:
        # return torch.sigmoid(z.mean(dim=0))
        return torch.sigmoid(torch.cat((3 * z.mean(dim=0).unsqueeze(0),
                                        z.max(dim=0)[0].unsqueeze(0),
                                        z.min(dim=0)[0].unsqueeze(0),
                                        2 * z.median(dim=0)[0].unsqueeze(0),
                                        ), dim=0))
    
    def forward(self,x,edge_list):
        batch,gene_num,feature = x.shape

        #将数据整理成图神经网络的输入
        x = x.view(-1, feature) #feature = input_seg_num * d_model
        edge_list = remove_self_loops(edge_list)[0]
        edge_list = add_self_loops(edge_list,num_nodes=gene_num)[0] #每个基因加上自回边
        each_adj = to_dense_adj(edge_list).squeeze()                #将邻接矩阵转换为矩阵形式
        batch_adj = [each_adj for _ in range(batch)]
        stack_adj = torch.block_diag(*batch_adj)                    # 合成一个大的邻接矩阵，batch内的不同细胞的基因之间没有关系
        batch_edge_list = dense_to_sparse(stack_adj)[0].to(device=x.device,dtype=torch.int64)

        pos_z, neg_z, summary = self.graph_encoder(x,batch_edge_list)
        graph_loss = self.graph_encoder.loss(pos_z, neg_z, summary)
        
        return pos_z, graph_loss

class Encoder(nn.Module):
    def __init__(self, in_len, seg_len, d_model, d_ff, n_heads, dropout, router):
        super(Encoder, self).__init__()
        input_seg_num = int(in_len / seg_len)

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = PositionalEncoding(input_seg_num, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        
        # # Graph Encoder
        self.graph = Graph(d_model, input_seg_num, dropout)
    
    def forward(self, x_seq, edge_list):
        batch_size = x_seq.shape[0]
        x_seq = self.enc_value_embedding(x_seq) # batch_size, seg_num*seg_len, gene_num -> batch_size, gene_num, seg_num, d_model
        x_seq = self.enc_pos_embedding(x_seq) # shape不变，加上时间顺序信息
        x_seq = self.pre_norm(x_seq)    # shape不变，对最后一维进行归一化
        
        enc = x_seq
        enc = rearrange(enc, 'b gene_num seg_num d_model -> b gene_num (seg_num d_model)')
        # 用原始图和打乱图分别对基因特征进行编码，得到pos_z和neg_z，然后与summary比较得到graph_loss
        enc, graph_loss  = self.graph(enc, edge_list)      # (b * gene_num) (input_seg_num * d_model)
        enc = rearrange(enc, '(b  gene_num) feature -> b gene_num feature', b = batch_size)

        return graph_loss, enc

class NextStep(nn.Module):
    def __init__(self, gene_num, pred_method='linear'):
        super(NextStep, self).__init__()
        self.pred_method = pred_method
        if pred_method == 'linear':
            self.lin = nn.Linear(gene_num, gene_num)
        elif pred_method == 'ode':  #TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, x):   # b,g,f -> b,g,f
        if self.pred_method == 'linear':
            x = rearrange(x, 'b g f -> b f g')
            x = self.lin(x)
            x = rearrange(x, 'b f g -> b g f')
            return x
        elif self.pred_method == 'ode':
            raise NotImplementedError
        else:
            raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, gene_num, in_dim, out_dim, hidden, dropout=0.1, batch_norm=False):
        super(Decoder, self).__init__()
        self.batch_norm = batch_norm
        self.in_dp = nn.Dropout(dropout)
        hidden = [in_dim] + [hidden] if isinstance(hidden, int) else [in_dim] + hidden
        self.layers = nn.ModuleList([])
        for i in range(len(hidden) - 1):
            self.layers.append(nn.ModuleList([]))
            self.layers[-1].append(nn.Linear(hidden[i], hidden[i+1]))
            self.layers[-1].append(nn.Linear(gene_num, gene_num))
            if batch_norm:
                self.layers[-1].append(nn.BatchNorm1d(gene_num*hidden[i+1]))
            self.layers[-1].append(nn.GELU())
            self.layers[-1].append(nn.Dropout(dropout))
        self.out = nn.Linear(hidden[-1], out_dim)
    def forward(self, x):
        b, g, _ = x.shape
        x = self.in_dp(x)
        for layer in self.layers:
            x = layer[0](x)
            x = rearrange(x, 'b g f -> b f g')
            x = layer[1](x)
            x = rearrange(x, 'b f g -> b g f')
            if self.batch_norm:
                x = x.view(b, -1)
                x = layer[2](x)
                x = x.view(b, g, -1)
            x = layer[-2](x)
            x = layer[-1](x)
        x = self.out(x)
        return rearrange(x, 'b g t -> b t g')

class STGRN(nn.Module):
    def __init__(self, gene_num, in_len, out_len, seg_len, d_model, d_ff, n_heads, dropout, router, next_step, hidden, bn, device):
        super(STGRN, self).__init__()
        input_seg_num = int(in_len / seg_len)
        if input_seg_num*seg_len != in_len:
            raise ValueError
        self.next_step = next_step
        self.in_len = in_len
        self.out_len = out_len
        self.gene_num = gene_num

        # Encoder
        self.encoder = Encoder(in_len, seg_len, d_model, d_ff, n_heads, dropout, router)

        # GRN learner
        self.nextstep = NextStep(gene_num, next_step)

        # Decoder
        self.decoder = Decoder(gene_num, input_seg_num*d_model, in_len, hidden=hidden, dropout=dropout, batch_norm=bn)
        
    def forward(self, x_seq, edge_list, mode='all'):    # x_seq = i~i+in_len时间点的表达矩阵；edge_list = 先验网络
                                                            # x_seq.shape = (batch_size, timepoints, gene_nums)
        graph_loss, enc = self.encoder(x_seq, edge_list)    # b, t, g -> b, g, f=in_len*d_model/seg_len

        enc = enc if mode == 'encode' else self.nextstep(enc)  # b,g,f -> b,g,f

        # Decoder先对dec_transform_embedding进行TwoStageAttentionLayer变换，然后作为query查询编码的状态，最后线性变换。
        dec = self.decoder(enc) if mode == 'encode' else self.decoder(enc)[:, -self.out_len:, :] #b,g,f -> b,t,g

        return graph_loss, dec

    def grn(self, method='weight', test_loader=None, edge_list=None, device='cuda:0', logger=None):
        if method == 'weight':
            if self.next_step == 'linear':
                return self.nextstep.lin.weight
        
        elif method[:9] == 'embedding':
            self.eval()
            with torch.no_grad():
                grns = []
                for batch_x, _ in test_loader:
                    batch_x = batch_x.float().to(device)
                    _, emb = self.encoder(batch_x, edge_list)
                    grn = emb @ rearrange(emb, 'b g f -> b f g')
                    if 'norm' in method:
                        diag = torch.diagonal(grn, dim1=-2, dim2=-1)
                        norm = diag.pow(-0.5).unsqueeze(-1).expand_as(grn)
                        grn = norm * grn * norm.transpose(-2,-1)
                    grns.append(grn)
                mean_grn = torch.cat(grns, dim=0).mean(dim=0)
                if 'weight' in method:
                    mean_grn = mean_grn * self.nextstep.lin.weight
            return mean_grn
        
        elif method == 'derivative':
            self.eval()
            grns = []
            p, cnt = 0.1, 0
            try:
                for batch_x, _ in test_loader:
                    if np.random.random() > p:
                        logger.info('skip one batch to save time')
                        continue
                    cnt += 1
                    logger.info(f'batch #{cnt} grn calculating...')
                    batch_x = batch_x.float().to(device).requires_grad_(True)
                    b, t, g = batch_x.shape
                    _, batch_y = self.forward(batch_x, edge_list)
                    grn = []
                    for j in range(g):
                        batch_y[:, 0, j].backward(torch.ones(b, device=device), retain_graph=True)
                        grn.append(batch_x.grad[:, -1, :])  # 第j个target gene对各个source gene的导数
                        batch_x.grad.zero_() # 清空梯度，因为每个target gene对source的导数需要分开计算
                    grn = torch.stack(grn, dim=1) # 第1到第g个target gene (dim=1) 对source gene (dim=2) 的导数（可以看作受到的调控作用）
                    grns.append(grn)
            except KeyboardInterrupt:
                logger.info('Stopped by user.')
            mean_grn = torch.cat(grns, dim=0).mean(dim=0)
            return mean_grn

    def set_trainee(self, param='all'):
        if param == 'all':
            self.encoder.requires_grad_(True)
            self.nextstep.requires_grad_(True)
            self.decoder.requires_grad_(True)
        elif param == 'encode':
            self.encoder.requires_grad_(True)
            self.nextstep.requires_grad_(False)
            self.decoder.requires_grad_(True)
        elif param == 'predict':
            self.encoder.requires_grad_(False)
            self.nextstep.requires_grad_(True)
            self.decoder.requires_grad_(False)

class Entry_STGRN(object):
    def __init__(self, data, args, logger:Logger):
        super(Entry_STGRN, self).__init__()
        self.data = data
        self.args = args
        self.logger = logger
        self.device = self._acquire_device()
        self.loss_gamma = [float(x) for x in args.loss_gamma.split(',')]
        
        self.model = self._build_model(data[0].shape[1]).to(self.device)
        if self.args.train_mode == 'predict' and self.args.load_model == 'None':
            self.logger.warning('train_mode=predict，但是load_model=None。请提供encode阶段的模型！')
        if self.args.load_model != 'None':
            self.logger.info('加载已存模型...')
            path = os.path.join(self.args.load_model, 'model')
            ckpt = torch.load(path+'/'+'checkpoint.pth')
            self.model.load_state_dict(ckpt)
    
    def _build_model(self, gene_num):        
        model = STGRN(
            gene_num, 
            self.args.in_len, 
            self.args.out_len,
            self.args.seg_len,
            self.args.d_model, 
            self.args.d_ff,
            self.args.n_heads, 
            self.args.dropout,
            self.args.router, 
            self.args.next_step, 
            [int(x) for x in self.args.hidden_layers.split(',')], 
            self.args.batch_norm,
            self.device
        ).float()

        return model

    def _acquire_device(self):
        return torch.device('cuda:0')

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=5e-4)

    def _select_criterion(self):
        return {'rmse':myRMSELoss, 
                'nrmse':myNRMSELoss}[self.args.loss_func](self.args.mask_zero)
    
    def _select_cosine_scheduler(self, optim, num_steps):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps)
    
    def valid(self, vali_loader,edge_list,criterion):
        self.model.eval()
        valid_loss_list, recon_loss_list, graph_loss_list, sparse_loss_list = (list() for _ in range(4))
        with torch.no_grad():
            for batch_x,batch_y in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                graph_loss, predict = self.model(batch_x,edge_list,self.args.train_mode)
                recon_loss = criterion(predict, batch_y)
                sparse_loss = torch.mean(torch.abs(self.model.grn())) * 100
                valid_loss = (self.loss_gamma[0] * recon_loss + 
                              self.loss_gamma[1] * graph_loss + 
                              self.loss_gamma[2] * sparse_loss)

                valid_loss_list.append(valid_loss.detach().item())
                recon_loss_list.append(recon_loss.detach().item())
                graph_loss_list.append(graph_loss.detach().item())
                sparse_loss_list.append(sparse_loss.detach().item())
        valid_loss = np.nanmean(valid_loss_list)
        recon_loss = np.nanmean(recon_loss_list)
        graph_loss = np.nanmean(graph_loss_list)
        sparse_loss = np.nanmean(sparse_loss_list)
        return valid_loss, recon_loss, graph_loss, sparse_loss

    def train(self):
        shuffle_flag = True; drop_last = self.args.batch_norm; batch_size = self.args.batch_size
        data = self.data

        self.logger.info('加载训练数据集')
        train_data = Dataset_MTS(
            data[0],
            size=[self.args.in_len, self.args.out_len],
            flag = 'train',
            mode = self.args.train_mode
        )

        self.logger.info(f'训练集大小：{len(train_data)}')

        edge_list = data[1]

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        self.logger.info('加载验证数据集')
        val_data = Dataset_MTS(
            data[0],
            size=[self.args.in_len, self.args.out_len],
            flag = 'val',
            mode = self.args.train_mode
        )
        self.logger.info(f'验证集大小：{len(val_data)}')
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        self.logger.info('创建结果文件夹')
        path = os.path.join(self.args.result_path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)

        avg_train_loss_list, avg_graph_loss_list, avg_recon_loss_list, avg_sparse_loss_list = (list() for _ in range(4))
        valid_loss_list, valid_recon_loss_list, valid_graph_loss_list, valid_sparse_loss_list = (list() for _ in range(4))
        
        try:
            criterion = self._select_criterion()
            early_stopping = EarlyStopping(patience=self.args.patience, logger=self.logger)
            model_optim = self._select_optimizer()
            warmup_scheduler = warmup.LinearWarmup(model_optim, self.args.warm_up_steps)
            lr_scheduler = self._select_cosine_scheduler(model_optim, len(train_loader) * self.args.train_epochs)

            for epoch in range(self.args.train_epochs):
                print()
                self.logger.info(f'training epoch {epoch+1}')
                self.model.train()
                self.model.set_trainee(self.args.train_mode)

                train_loss_list = list()
                graph_loss_list = list()
                recon_loss_list = list()
                sparse_loss_list = list()

                for batch_x,batch_y in train_loader:
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    graph_loss, predict_y= self.model(batch_x,edge_list,self.args.train_mode)
                    recon_loss = criterion(predict_y, batch_y)
                    sparse_loss = torch.mean(torch.abs(self.model.grn())) * 100
                    whole_loss = (self.loss_gamma[0] * recon_loss + 
                                self.loss_gamma[1] * graph_loss + 
                                self.loss_gamma[2] * sparse_loss)

                    train_loss_list.append(whole_loss.item())
                    graph_loss_list.append(graph_loss.item())
                    recon_loss_list.append(recon_loss.item())
                    sparse_loss_list.append(sparse_loss.item())
                    
                    whole_loss.backward()
                    model_optim.step()
                    
                    with warmup_scheduler.dampening():  # warmup步数内lr线性增长到设定值；以外则warmup不起作用，lr_scheduler起作用，lr余弦下降
                        if warmup_scheduler.last_step >= self.args.warm_up_steps:
                            lr_scheduler.step()
                
                avg_train_loss = np.nanmean(train_loss_list)
                avg_graph_loss = np.nanmean(graph_loss_list)
                avg_recon_loss = np.nanmean(recon_loss_list)
                avg_sparse_loss = np.nanmean(sparse_loss_list)
                self.logger.info(f'train_loss:{avg_train_loss:.4g}   recon_loss:{avg_recon_loss:.4g}  graph_loss:{avg_graph_loss:.4g}   sparse_loss:{avg_sparse_loss:.4g}')

                valid_loss, valid_recon_loss, valid_graph_loss, valid_sparse_loss = self.valid(val_loader, edge_list, criterion)
                self.logger.info(f'val_loss:{valid_loss:.4g}   recon_loss:{valid_recon_loss:.4g}  graph_loss:{valid_graph_loss:.4g}   sparse_loss:{valid_sparse_loss:.4g}')
                early_stopping(valid_loss, self.model, path)

                avg_train_loss_list.append(avg_train_loss)
                avg_recon_loss_list.append(avg_recon_loss)
                avg_graph_loss_list.append(avg_graph_loss)
                avg_sparse_loss_list.append(avg_sparse_loss)
                valid_loss_list.append(valid_loss)
                valid_recon_loss_list.append(valid_recon_loss)
                valid_graph_loss_list.append(valid_graph_loss)
                valid_sparse_loss_list.append(valid_sparse_loss)

                if early_stopping.early_stop:
                    self.logger.info("Early stopping")
                    break
        except KeyboardInterrupt:
            self.logger.info("Stopped by user.")

        path = os.path.join(self.args.result_path, 'loss')
        if not os.path.exists(path):
            os.makedirs(path)

        x_coordinate = list(range(1, len(avg_train_loss_list)+1))
        y_coordinate = [avg_train_loss_list, 
                        avg_recon_loss_list, 
                        avg_graph_loss_list, 
                        avg_sparse_loss_list, 
                        valid_loss_list, 
                        valid_recon_loss_list, 
                        valid_graph_loss_list, 
                        valid_sparse_loss_list]
        y_title = ['train_loss', 'train_recon_loss', 'train_graph_loss', 'train_sparse_loss', 
                   'valid_loss', 'valid_recon_loss', 'valid_graph_loss', 'valid_sparse_loss']

        rows, cols = 2,4
        fig, axs = plt.subplots(rows,cols,figsize=(4*cols,3*rows))
        for i in range(rows):
            for j in range(cols):
                if (k:= i*cols+j) < len(y_coordinate):
                    axs[i,j].plot(x_coordinate, y_coordinate[k], marker='.')
                    axs[i,j].set_title(y_title[k])
                    min_value = min(y_coordinate[k])
                    min_index = y_coordinate[k].index(min_value) + 1
                    x_offset = 0 if min_index<=len(x_coordinate)/2 else -int(0.4*len(x_coordinate))
                    ymin, ymax = axs[i, j].get_ylim()
                    y_offset = (ymax-ymin)*0.75+ymin
                    axs[i, j].annotate(f'Min Loss: {min_value:.4g}', xy=(min_index, min_value),
                                       xytext=(min_index + x_offset, y_offset),
                                       arrowprops=dict(facecolor='black', arrowstyle='->'))
                else:
                    axs[i,j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path,'loss.jpg'))
        plt.clf()

        return

    def test(self,load_model=True):
        data = self.data
        if load_model:
            self.logger.info('加载已存模型...')
            if self.args.result_path == './tmp/':
                self.logger.warning('从默认路径./tmp/读取模型')
            if self.args.train_mode == 'seq':
                path = os.path.join(self.args.result_path, 'predict/model')
            else:
                path = os.path.join(self.args.result_path, 'model')
            ckpt = torch.load(path+'/'+'checkpoint.pth')
            self.model.load_state_dict(ckpt)

        self.model.eval()

        shuffle_flag = False; drop_last = self.args.batch_norm; batch_size = self.args.batch_size

        self.logger.info('加载测试数据集...')
        test_data = Dataset_MTS(
            data[0],
            size=[self.args.in_len, self.args.out_len],
            flag = 'test',
            mode = 'all'
        )
        self.logger.info(f'测试集大小：{len(test_data)}')

        edge_list = data[1]

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        metrics_all = []
        criterion =  self._select_criterion()

        self.logger.info('test...')

        with torch.no_grad():
            loss_list = list()
            for batch_x,batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                graph_loss, predict_y = self.model(batch_x,edge_list)

                batch_metric = np.array(metric(predict_y.detach().cpu().numpy(), batch_y.detach().cpu().numpy(), mask=self.args.mask_zero))
                recon_loss = criterion(predict_y, batch_y)

                sparse_loss = torch.mean(torch.abs(self.model.grn())) * 100
                whole_loss = (self.loss_gamma[0] * recon_loss + 
                              self.loss_gamma[1] * graph_loss + 
                              self.loss_gamma[2] * sparse_loss)
                
                loss_list.append(whole_loss.detach().item())
                metrics_all.append(batch_metric)

            avg_test_loss = np.nanmean(loss_list)
            self.logger.info(f'test_loss:{avg_test_loss:.4g}')
            metrics_all = np.stack(metrics_all, axis = 0)
            metrics_mean = np.nanmean(metrics_all, axis=0)
        
        # result save
        path = os.path.join(self.args.result_path, 'metrics')
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.args.velo_param != 'None':

            test_data = Dataset_MTS(
                data[0],
                size=[self.args.in_len, self.args.out_len],
                flag = 'test',
                mode = 'all', 
                padding = self.args.velo_param != 'None'
            )

            test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=self.args.num_workers,
                drop_last=drop_last)
            
            weights, nxts = [[float(x) for x in s.split(',')] for s in self.args.velo_param.split(';')]
            weights = np.array(weights).reshape(1,-1,1)
            nxts = [i-1 for i in nxts]
            velos = []
        
            with torch.no_grad():
                for batch_x,batch_y in test_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    in_x = batch_x.detach().cpu().numpy()
                    # out_y = batch_y.detach().cpu().numpy()
                    
                    graph_loss, predict_y = self.model(batch_x,edge_list)
                    out_y = predict_y.detach().cpu().numpy()
                    
                    velos.append(np.sum((np.take(out_y, nxts, axis=1) - in_x[:, [-1], :]) * weights, axis=1))

            self.logger.info("Saving velo.csv ...")
            velos = np.vstack(velos)
            np.savetxt(os.path.join(path, 'velo.csv'), velos, delimiter=',')

        mae, rmse, nrmse = metrics_mean
        self.logger.info("Saving recon_metric.txt ...")
        with open(os.path.join(path,'recon_metric.txt'),'w') as f:
            f.write(f'mae: {mae}\nrmse: {rmse}\nnrmse: {nrmse}\n')
        self.logger.info(f'mae:{mae:.4g}   rmse:{rmse:.4g}   nrmse:{nrmse:.4g}')

        self.logger.info('加载tf列表...')
        tf_list = pd.read_csv(self.args.tf_path)
        tf_genes = tf_list.iloc[:, 0].tolist()

        id_gene_map = data[2]
        gene_num = len(id_gene_map)
        filtered_idx = id_gene_map[id_gene_map.index.isin(tf_genes)]['idx'].tolist()
        TF_mask = np.zeros((gene_num,gene_num),dtype = int)
        TF_mask[:, filtered_idx] = 1
        TF_mask = TF_mask - np.eye(gene_num, dtype=int) * TF_mask

        #获取grn results
        self.logger.info('计算GRN...')
        grn = self.model.grn(method=self.args.grn, test_loader=test_loader, edge_list=edge_list, 
                             device=self.device, logger=self.logger).squeeze().detach().cpu().numpy()
        masked_grn = grn * TF_mask
        idx_rec, idx_send = np.where(masked_grn > 0)
        weight_list, weight_abs = list(), list()
        for i,j in zip(idx_rec,idx_send):
            weight_list.append(grn[i, j])
            weight_abs.append(abs(grn[i, j]))

        grn_df = pd.DataFrame({'Gene1': id_gene_map.iloc[idx_send].index.tolist(), 
                               'Gene2': id_gene_map.iloc[idx_rec].index.tolist(), 
                               'EdgeWeight': weight_list, 
                               'weight_abs': weight_abs})
        grn_df = grn_df.sort_values('weight_abs', ascending=False)

        ground_truth_network = gt = pd.read_csv(self.args.gt_path).drop_duplicates(keep='first')
        ground_truth_network = gt.loc[gt['Gene1'].isin(gn:= set(id_gene_map.index)) & gt['Gene2'].isin(gn), :].reset_index(drop=True)

        # # prior network eval
        # prior = pd.DataFrame({'Gene1': id_gene_map.iloc[edge_list.numpy()[0]].index.tolist(), 
        #                       'Gene2': id_gene_map.iloc[edge_list.numpy()[1]].index.tolist(), 
        #                       'EdgeWeight': [1]*edge_list.shape[1], 
        #                       'weight_abs': [1]*edge_list.shape[1]})
        # Eprec, Erec, p_tail, EPR = EarlyPrec(ground_truth_network, prior, weight_key='weight_abs', 
        #                                        all_nodes=id_gene_map.index, TFEdges=tf_genes, topk=None, logger=self.logger)
        # self.logger.info(f'for prior network, Eprec:{Eprec:.4g}   Erec:{Erec:.4g}   p_tail:{p_tail:.4g}   EPR:{EPR:.4g}')
        
        # my grn eval
        # test EdgeWeight or weight_abs：区别不大，一般比较大的都是正的
        # Eprec, Erec, p_tail, EPR = EarlyPrec(ground_truth_network, grn_df, weight_key='weight_abs', 
        #                                        all_nodes=id_gene_map.index, TFEdges=tf_genes, topk=35120, logger=self.logger)
        # self.logger.info(f'for topk=35120, Eprec:{Eprec:.4g}   Erec:{Erec:.4g}   p_tail:{p_tail:.4g}   EPR:{EPR:.4g}')

        # precs, recs, fprs, tprs, AUPRC, AUROC = computeScores(ground_truth_network,grn_df, weight_key='weight_abs',
        #                                                       all_nodes=id_gene_map.index, TFEdges=tf_genes, logger=self.logger)
        # self.logger.info(f'AUPRC:{AUPRC:.4g}   AUROC:{AUROC:.4g}')
        
        # self.logger.info('Saving grn_metrix.txt ...')
        # with open(os.path.join(path,'grn_metric.txt'),'w') as f:
        #     f.write(f'Eprec: {Eprec}\nErec: {Erec}\np_tail: {p_tail}\nEPR: {EPR}\nAUPRC: {AUPRC}\nAUROC: {AUROC}\n')
        
        # _, axs = plt.subplots(1, 2, figsize=(8, 3))
        # axs[0].plot(recs, precs)
        # axs[0].set_title('precision-recall curve')
        # axs[0].set_xlabel('recall')
        # axs[0].set_ylabel('precision')

        # axs[1].plot(fprs, tprs)
        # axs[1].set_title('tpr-fpr curve')
        # axs[1].set_xlabel('fpr')
        # axs[1].set_ylabel('tpr')

        # plt.tight_layout()
        # plt.savefig(os.path.join(path, 'curves.jpg'))
        # plt.clf()

        Eprec, Erec, EPR = EarlyPrec(ground_truth_network, grn_df, weight_key='weight_abs', TFEdges=True)
        self.logger.info(f'Eprec:{Eprec:.4g}   Erec:{Erec:.4g}   EPR:{EPR:.4g}')

        precs, recs, fprs, tprs, AUPRC, AUROC = computeScores(ground_truth_network, grn_df, weight_key='weight_abs', selfEdges=False)
        self.logger.info(f'AUPRC:{AUPRC:.4g}   AUROC:{AUROC:.4g}')
        
        self.logger.info('Saving grn_metrix.txt ...')
        with open(os.path.join(path,'grn_metric.txt'),'w') as f:
            f.write(f'Eprec: {Eprec}\nErec: {Erec}\nEPR: {EPR}\nAUPRC: {AUPRC}\nAUROC: {AUROC}\n')
        
        _, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(recs, precs)
        axs[0].set_title('precision-recall curve')
        axs[0].set_xlabel('recall')
        axs[0].set_ylabel('precision')

        axs[1].plot(fprs, tprs)
        axs[1].set_title('tpr-fpr curve')
        axs[1].set_xlabel('fpr')
        axs[1].set_ylabel('tpr')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'curves.jpg'))
        plt.clf()

        self.logger.info('Saving grn.csv ...')
        grn_df.to_csv(os.path.join(self.args.result_path, 'grn.csv'),index=False)
        return
