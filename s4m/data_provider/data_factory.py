from data_provider.data_loader import Dataset_Grafiti,Dataset_CRU,Dataset_Brits,Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred,Dataset_Custom2,Dataset_Solar1,Dataset_Custom4,Dataset_Transformer,Dataset_Transformer_Impute
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar1,
    'Brits':Dataset_Brits,
    'PEMS': Dataset_PEMS,
    'custom2':Dataset_Custom2,
    'custom4':Dataset_Custom4,
    'cru': Dataset_CRU,
    'grafiti': Dataset_Grafiti,
    'transformer':Dataset_Transformer,
    'transformer_impute':Dataset_Transformer_Impute
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size =  args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = True
        batch_size =  args.batch_size 
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        args=args,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        mask=args.mask 
    )
    
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=True)
    return data_set, data_loader