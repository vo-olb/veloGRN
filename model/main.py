import argparse
from utils import *
from preprocessing import preprocess
import pickle
from logging import Logger

def parse():
    parser = argparse.ArgumentParser(description='STGRN')
    
    #data info
    # 路径/home/pengrui/grn_workspace/ -> ../ （因为现在在grn_workspace/own_work）
    parser.add_argument('--dataset', type=str, default='mESC', help='dataset name')
    parser.add_argument('--expr_path', type=str, default='None', help='scRNA-seq data file')
    parser.add_argument('--network_path',type=str,default='None',help='prior network for GCN. if scATAC-seq file is provided, it will be processed into prior network according to Dictys')
    parser.add_argument('--tf_path', type=str, default='None', help='transcription factor list file')
    parser.add_argument('--time_info',type=str,default='None',help='pseudotime')
    parser.add_argument('--preprocess',type=str,default='None',help='preprocessing method')
    
    parser.add_argument('--mode', type=int, default=3, help='if_train + if_test*2')
    parser.add_argument('--gt_path',type=str,default='None',help='ground truth file for benchmark')
    parser.add_argument('--result_path', type=str, default='../test/tmp/', help='location to store model checkpoints and other results')
    parser.add_argument('--load_model', type=str, default='None', help='location to load model checkpoint. it should be used only when you have a pre-trained model and want to continue training it')
    
    # segment info
    parser.add_argument('--in_len', type=int, default=32, help='input MTS length (T)')
    parser.add_argument('--out_len', type=int, default=32, help='output MTS length (\tau). it should be less than or equal to in_len. it will not work in the "encode" mode')  #32
    parser.add_argument('--seg_len', type=int, default=8, help='segment length (L_seg)')
    parser.add_argument('--router', type=int, default=10,help='the num of router for attention encoder/decoder')
    
    # multihead info
    parser.add_argument('--d_model', type=int, default=64, help='dimension of hidden states of time layer transformer')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of MLP in transformer')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    
    # other model info
    parser.add_argument('--next_step', type=str, default='linear', help='method for predicting next state. "linear" or "ode"')
    parser.add_argument('--hidden_layers', type=str, default='64', help='decoder hidden layers')
    parser.add_argument('--batch_norm', action='store_true', help='decoder batch normalization layers')
    
    # train info
    parser.add_argument('--train_mode', type=str, default='all', help='all, encode, predict, or seq')
    parser.add_argument('--mask_zero', action='store_true', help='whether to mask zero entries when calculating losses')
    parser.add_argument('--loss_func', type=str, default='rmse', help='rmse or nrmse')
    parser.add_argument('--loss_gamma', type=str, default='0.5,0,0.5', help='weight for reconstruction loss, graph loss and sparsity loss')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')   #没有用
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warm_up_steps',type=int,default=100,help='warm up steps')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')    #1  #32太大了 会爆显存
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')   #200
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')    #1e-4
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    
    # test info
    parser.add_argument('--grn', type=str, default='embedding', help='weight, embedding or derivative')
    # parser.add_argument('--grn_alpha', type=float, default=0, help='if grn method is derivative, alpha is used to control the weight')
    parser.add_argument('--test', type=str, default='normal', help='normal, no_encoder, no_GCN, or no_Attention')
    parser.add_argument('--velo_param', type=str, default='0.333,0.333,0.333;2,8,32', help='parameters for pseudo-velocity calculating. \
                        format: "[weight1],...;[nxt1],...". please note that the quotation mark is essential because you have to include colon mark in the string.')
    
    args = parser.parse_args()
    
    return args

def warn(args, logger:Logger):
    if args.mode>>1 and (v:= args.velo_param) != 'None':
        _, nxts = [[float(x) for x in s.split(',')] for s in v.split(';')]
        if max(nxts) > args.out_len:
            logger.error("if you want to compute and save velocity, please ensure that the out_len is greater than the maximum of nxt parameters.")

def get_files(args, logger):
    files = {'mESC': ('../data/mESC/ExpressionData.csv', 
                      '../data/mouse-PriorNetwork.csv', 
                      '../data/mouse-TFs.csv', 
                      '../data/mESC/PseudoTime.csv', 
                      '../data/mESC/GroundTruth.csv'), 
             'mHSC-E': ('../data/mHSC-E/ExpressionData.csv', 
                        '../data/mouse-PriorNetwork.csv', 
                        '../data/mouse-TFs.csv', 
                        '../data/mHSC-E/PseudoTime.csv', 
                        '../data/mHSC-E/GroundTruth.csv')}
    
    args_attr = ('expr_path', 'network_path', 'tf_path', 'time_info', 'gt_path')
    for i, a in enumerate(args_attr):
        if getattr(args, a) == 'None':
            logger.info(f'for dataset {args.dataset}, {a} is set to {files[args.dataset][i]} by default')
            setattr(args, a, files[args.dataset][i])

def main():
    change_into_current_py_path()       #将工作目录改变到当前脚本位置
    # seed_everything(2024)   #设置随机种子
    args = parse()

    res = args.result_path if args.result_path[-1] not in '/\\' else args.result_path[:-1]
    logpath = os.path.join(os.path.dirname(res), "log.txt")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, 'a') as f:
        pass
    (logger:= set_logger(logpath)).info(args)    #设置logger, log.txt默认放在res的上一级目录

    warn(args, logger)
    get_files(args, logger)

    models = []
    for i in range(1, args.itr+1):
        if args.itr != 1:
            logger.info("第"+str(i)+"次实验...")
            args.result_path = res + "_" + str(i)
        
        os.makedirs(args.result_path, exist_ok=True)

        try:
            with open(os.path.join(args.result_path, 'data.pkl'), 'rb') as f:
                data = pickle.load(f)
            logger.info('成功读取已预处理数据集')
        except:
            try:
                path = args.result_path[:-len(str(i))] + '1'
                with open(os.path.join(path, 'data.pkl'), 'rb') as f:
                    data = pickle.load(f)
                logger.info('成功读取第1次实验中已预处理数据集')
            except:
                logger.info('预处理数据集...')
                data = preprocess(args)
            with open(os.path.join(args.result_path, 'data.pkl'), 'wb') as f:
                pickle.dump(data, f)

        if args.test == 'normal':
            from model import Entry_STGRN
        elif args.test == 'no_encoder':
            from model_no_encoder import Entry_STGRN
            if args.train_mode != 'all':
                args.train_mode = 'all'
                logger.warning('no_encoder模式中，train_mode强制设定为all')
        elif args.test == 'no_GCN':
            from model_no_GCN import Entry_STGRN
        elif args.test == 'no_Attention':
            from model_no_Attention import Entry_STGRN
        else:
            raise KeyError

        if args.mode&1:
            logger.info('train...')

            if args.train_mode != 'seq':
                model = Entry_STGRN(data, args, logger)
                model.train()

            else:
                args.train_mode = 'encode'
                args.result_path = os.path.join(args.result_path, 'encode')
                model = Entry_STGRN(data, args, logger)
                model.train()

                args.train_mode = 'predict'
                args.load_model = args.result_path
                args.result_path = os.path.join(args.result_path, '../predict')
                model = Entry_STGRN(data, args, logger)
                model.train()

                args.train_mode = 'all'
                args.load_model = args.result_path
                args.result_path = os.path.join(args.result_path, '../all')
                model = Entry_STGRN(data, args, logger)
                model.train()

                args.train_mode = 'seq'
                args.result_path = os.path.join(args.result_path, '..')

        if args.mode>>1:
            logger.info('test...')

            model = Entry_STGRN(data, args, logger)
            model.test()
        
        models.append(model)
        logger.info('-----END-----')
    
    return models

if __name__ == '__main__':
    _ = main()