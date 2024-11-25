import argparse
# from experiments.exp_long_term_forecasting2 import Exp_Long_Term_Forecast as Exp_Long_Term_Forecast2
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp_pretrain1
# from experiments.exp_transformer import Exp_Long_Term_Forecast as exp_transformer
# from experiments.exp_BiaTCGNet import Exp_Long_Term_Forecast as exp_BiaTCGNet
# from experiments.exp_brits import Exp_Long_Term_Forecast as Exp_brits
# from experiments.exp_grafiti import Exp_Long_Term_Forecast as Exp_grafiti
# from experiments.exp_cru import Exp_Long_Term_Forecast as Exp_cru
import random
import numpy as np
import wandb
import torch, gc


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int,  default=1, help='status')  # required=True,
    parser.add_argument('--seed', type=int,  default=2023)  # required=True,
    parser.add_argument('--plot', type=int,  default=0)  # required=True,

    parser.add_argument('--model_id', type=str,default='test', help='model id') # required=True, 
    parser.add_argument('--model', type=str,  default='grafiti',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer,S_Mamba ]')#required=True,

    # data loader
    parser.add_argument('--data', type=str, default='classify', help='dataset type')  #required=True,
    parser.add_argument('--test', type=str, default='test', help='test function type')  #required=True,

    parser.add_argument('--root_path', type=str, default='/home/project/S4M/data/ETT-small/chunk_missing/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh15_3.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/home/project/S4M/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--mask',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--check',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--reverse_mask',action='store_true', help='whether use reverse mask')  # required=True
    parser.add_argument('--grud',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--s4_pred',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--s4_pred_inner',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--bidirectional',action='store_true', help='whether use bidirectional')  # required=True
    parser.add_argument('--pos_emb',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--pad', type=str,  default=0.0, help='how to fill nan')  # required=True
    parser.add_argument('--drop', action='store_true', help='whether to drop nan')  # 
    parser.add_argument('--analysis', action='store_true', help='whether to drop nan') 
    parser.add_argument('--mean_type', type=str, default='emperical', help='dataset type')

    parser.add_argument('--fillna', type=str, default='mean', help='dataset type')
    parser.add_argument('--encoder_type', type=int, default=1, help='dataset type')
 
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=862, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=862, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=862, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--classification', type=int, default=0, help='whether to classify')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--embed_type', type=int,default=2)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_weight1', type=float, default=1.0, help='loss weight1')
    parser.add_argument('--loss_weight2', type=float, default=1.0, help='loss weight2')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')
    parser.add_argument('--ssm', type=str, default='s4', help='ssm type')
    parser.add_argument('--num_dG', type=float, default=1)
    parser.add_argument('--num_pattn', type=int, default=20)    

    
    # Mermory Module
    parser.add_argument('--short_len', type=int, default=50)    
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--W', type=int, default=6)
    parser.add_argument('--en_conv_hidden_size', type=int, default=256)    
    parser.add_argument('--en_rnn_hidden_sizes', type=int, nargs='+', default=[20,32])    
    parser.add_argument('--output_keep_prob', type=float, default=0.9)    
    parser.add_argument('--input_keep_prob', type=float, default=0.9)  
    parser.add_argument('--weight', type=float, default=1.0)      
    parser.add_argument('--individual', type=int, default=1)
    parser.add_argument('--memnet', type=int, default=13)
    parser.add_argument('--mem_type', type=int, default=2)
    parser.add_argument('--shuffle', type=int, default=1)


    #diffusion
    parser.add_argument('--layers', type=int, default=2)    
    parser.add_argument('--channels', type=int, default=64)    
    parser.add_argument('--nheads', type=int, default=8)    
    parser.add_argument('--diffusion_embedding_dim', type=int, default=128)    
    parser.add_argument('--beta_start', type=float, default=0.0001)    
    parser.add_argument('--beta_end', type=float, default=0.5)    
    parser.add_argument('--num_steps', type=int, default=50)    
    parser.add_argument('--schedule', type=str, default='quad')    
    parser.add_argument('--is_linear', type=bool, default=True)    

    parser.add_argument('--is_unconditional', type=int, default=0)    
    parser.add_argument('--timeemb', type=int, default=128)    
    parser.add_argument('--featureemb', type=int, default=16)    
    parser.add_argument('--target_strategy', type=str, default="test")    
    parser.add_argument('--num_sample_features', type=int, default=64)    
    parser.add_argument("--nsample", type=int, default=10)

    #Brits
    parser.add_argument("--impute_weight", type=float, default=0.3)
    parser.add_argument("--label_weight", type=float, default=1)
    parser.add_argument("--impute_path", type=str, default='./')


    #BiaTCGNet
    parser.add_argument("--gcn_true", type=bool, default=True)
    parser.add_argument("--buildA_true", type=bool, default=True)
    parser.add_argument("--gcn_depth", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=207)
    parser.add_argument('--kernel_set', type=list, default=[2,3,6,7], help='kernel set')
    parser.add_argument("--subgraph_size", type=int, default=5)
    parser.add_argument("--node_dim", type=int, default=3)
    # parser.add_argument("--node_number", type=int, default=3)
    parser.add_argument("--conv_channels", type=int, default=8)
    parser.add_argument("--skip_channels", type=int, default=8)
    parser.add_argument("--seq_channels", type=int, default=8)
    parser.add_argument("--residual_channels", type=int, default=8)
    parser.add_argument("--end_channels", type=int, default=8)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--in_dim", type=int, default=1)
    parser.add_argument("--tanhalpha", type=float, default=3)
    parser.add_argument("--propalpha", type=float, default=0.05)
    parser.add_argument("--layer_norm_affline", type=bool, default=True)


    # memory bank
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--topK", type=int, default=10)
    parser.add_argument("--topM", type=int, default=100)
    parser.add_argument("--max_k", type=int, default=1000)
    parser.add_argument("--x_in", type=int, default=32)
    parser.add_argument("--thres1", type=float, default=0.6)
    parser.add_argument("--thres2", type=float, default=0.3)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--memory_size", type=int, default=256)
    parser.add_argument("--per_mem_size", type=int, default=50)

    parser.add_argument('--pretrain', action='store_true', help='whether to pretrain') 
    parser.add_argument('--no_renew', action='store_true', help='whether to pretrain') 
    parser.add_argument("--tau", type=int, default=32)
    parser.add_argument('--remask', type=float,default=0.0, help='whether to remask') 
    parser.add_argument("--hidden_v", type=int, default=32)
    parser.add_argument("--hidden_t", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--mem_repeat", type=int, default=10)
    parser.add_argument('--kernel', type=int, nargs='+', default=[5,5])    
    parser.add_argument('--moving_avgs', type=int, nargs='+', default=[25,20,15,10,5,5])   
    parser.add_argument('--tucker_kernels', type=int, nargs='+', default=[10,5,5])   


    #CRU
    parser.add_argument('--task', type=str, default='extrapolation')   
    parser.add_argument('--ts', type=float, default=1, help="Scaling factor of timestamps for numerical stability.")
    parser.add_argument('--hidden-units', type=int, default=24, help="Hidden units of encoder and decoder.")
    parser.add_argument('--num-basis', type=int, default=15, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
    parser.add_argument('--bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
    parser.add_argument('--enc-var-activation', type=str, default='elup1', help="Variance activation function in encoder. Possible values elup1, exp, relu, square")
    parser.add_argument('--dec-var-activation', type=str, default='elup1', help="Variance activation function in decoder. Possible values elup1, exp, relu, square")
    parser.add_argument('--trans_net_hidden_activation', type=str, default='tanh', help="Activation function for transition net.")
    parser.add_argument('--trans_net_hidden_units', type=list, default=[], help="Hidden units of transition net.")
    parser.add_argument('--trans_var_activation', type=str, default='elup1', help="Activation function for transition net.")
    parser.add_argument('--learn_trans_covar', type=bool, default=True, help="If to learn transition covariance.")
    parser.add_argument('--learn_initial_state_covar', action='store_true', help="If to learn the initial state covariance.")
    parser.add_argument('--initial_state_covar', type=int, default=1, help="Value of initial state covariance.")
    parser.add_argument('--trans_covar', type=float, default=0.1, help="Value of initial transition covariance.")
    parser.add_argument('--t_sensitive_trans_net',  action='store_true', help="If to provide the time gap as additional input to the transition net. Used for RKN-Delta_t model in paper")
    parser.add_argument('--f-cru',  action='store_true', help="If to use fast transitions based on eigendecomposition of the state transitions (f-CRU variant).")
    parser.add_argument('--rkn',  action='store_true', help="If to use discrete state transitions (RKN baseline).")
    parser.add_argument('--orthogonal', type=bool, default=True, help="If to use orthogonal basis matrices in the f-CRU variant.")


    parser.add_argument('--merge_size',type=int,default = 2)
    parser.add_argument('--dp_rank', type=int,default = 8)
    parser.add_argument('--use_statistic',action='store_true', default=False)
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--task_name', type=str,default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    args = parser.parse_args()
    
    
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.device = torch.device('cuda:{}'.format(args.gpu))
    # args.device = "cuda: 3"
    
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if "electricity" in args.data_path:
        args.d_var = 321
    elif "traffic" in args.data_path:
        args.d_var= 862
    elif "exchange_rate" in args.data_path:
        args.d_var = 8
    elif "weather" in args.data_path:
        args.d_var = 21
    elif "climate" in args.data_path:
        args.d_var = 9
    elif "ETT2" in args.data_path:
        args.d_var = 7
    elif "ETTh1" in args.data_path:
        args.d_var = 7        
    elif "ETTm1" in args.data_path:
        args.d_var = 7        
    elif "Solar" in args.data_path:
        args.d_var = 137    
    elif "FaceDetection" in args.data_path:
        args.d_var = 144
    elif "LSST" in args.data_path:
        args.d_var = 6
    elif 'FingerMovements' in args.data_path:
        args.d_var = 28
    elif 'MotorImagery' in args.data_path:
        args.d_var = 64
    elif 'Heartbeat' in args.data_path:
        args.d_var = 61    
    elif 'ArticularyWordRecognition' in args.data_path:
        args.d_var = 9  
    elif "simulation" in args.data_path:
        args.d_var = 10

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')    
    print(args)

    if args.model in ["GRUD","S4_Fill0",'S4_GRUD']:
        Exp = Exp_Long_Term_Forecast2
    elif args.model in ['Transformer','Transformer_M','Autoformer','CARD','iTransformer']:
        Exp = exp_transformer
        args.label_len = int(args.seq_len/2)
    elif args.model in ['grafiti']:
        Exp = Exp_grafiti
        args.data = 'grafiti'
    elif args.model in ['BiaTCGNet']:
        Exp = exp_BiaTCGNet
        args.data = 'custom2'
    elif args.model in ['S4M']:
        Exp = Exp_pretrain1
        args.data = 'custom4'
    elif args.model in ['Brits']:
        Exp = Exp_brits
        args.data = 'Brits'
    elif args.model in ['CRU']:
        Exp = Exp_cru
        args.data = 'cru'
    
    
    if args.model=="S4_Fill0" and args.data=="transformer_impute":
        Exp = exp_transformer
        args.label_len = int(args.seq_len/2)
        
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    wandb.init(config=args, project='S4',name = args.model)
    
    if args.is_training==1:
            
        for ii in range(args.itr):
            mask =1 if args.mask else 0
            # setting record of experiments
            setting = '{}_{}_ft{}_ll{}_pl{}_dm{}_el{}_eb{}_dt{}_{}_data_{}tag1_fill_{}_drop_{}_dG_{}_pattn_{}_n_{}_slen_{}_W_{}_ts1_{}_ts2_{}_K_{}_n_{}_M_{}_pre_{}_en_{}_s_{}_top{}_P_{}_m_{}_t_{}'.format(
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.embed,
                args.des,
                args.class_strategy, ii,args.data_path[:-4],args.fillna,str(args.drop),str(args.num_dG),str(args.num_pattn),str(args.n),str(args.short_len),str(args.W),
                str(args.thres1),
                str(args.thres2),
                str(args.K),
                str(args.n),
                str(args.M),
                str(args.pretrain),
                str(args.en_conv_hidden_size),
                str(args.moving_avg),
                str(args.topK),
                str(args.per_mem_size),
                str(args.memnet),
                str(args.test))

            args.check_path = setting
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.test=='test':
                exp.test(setting)
            else:
                exp.test1(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

    elif args.is_training == 2:
        for ii in range(args.itr):
            mask =1 if args.mask else 0
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_mask_{}_data_{}tag1_encoder_{}_fill_{}_drop_{}_pos_emb_{}_dG_{}_pattn_{}_n_{}_short_len_{}_W_{}_M_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii,str(mask),args.data_path[:-4],args.encoder_type,args.fillna,str(args.drop),str(args.pos_emb),str(args.num_dG),str(args.num_pattn),str(args.n),str(args.short_len),str(args.W),str(args.M))

            exp = Exp(args)
            exp.get_input(setting)
        
        torch.cuda.empty_cache()