import pandas as pd
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_transformer import DataTransformer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
    data = pd.read_csv(csv_filename, header='infer' if header else None)
    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name']
            for column in metadata['columns']
            if column['type'] != 'continuous'
        ]

    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = []

    return data, discrete_columns

def load_data(
    train=True,
    args=None
):
    DATA_DIR = './data'
    if train:
        data_path = f'{DATA_DIR}/{args.dataset}.data.csv'
    else:
        data_path = f'{DATA_DIR}/{args.dataset}.test.csv'
        
    metadata_path = f'{DATA_DIR}/{args.dataset}.json'
    
    if args.dataset == 'adult':
        target = 'income'
    elif args.dataset == 'credit':
        target = 'label'
    elif args.dataset == 'compas':
        target = 'Two_yr_Recidivism'
    elif args.dataset == 'lsac':
        target = 'pass_bar'
    elif args.dataset == 'communities':
        # sensitive: race
        target = 'ViolentCrimesPerPop'
    elif args.dataset == 'student_performance':
        # sensitive: sex
        target = 'G1'
    else:
        assert(0)
        
    data_org, discrete_columns = read_csv(
        csv_filename=data_path,
        meta_filename=metadata_path,
        header=args.header,
        discrete=args.discrete
    )
    
    return data_org, discrete_columns, target

def get_train_data(
    df,
    transformer_path,
    converter=True,
    args=None
):
    train_data = copy.deepcopy(df)
    train_data = train_data[train_data.columns.difference([args.sensitive, args.target])]
    
    if converter:
        transformer = DataTransformer()
        transformer.fit(train_data, args.discrete_columns)
        torch.save(transformer, transformer_path)
    else:
        transformer = torch.load(transformer_path)
        
    train_data = transformer.transform(train_data)
    
    df[args.sensitive] = df[args.sensitive].astype('category')
    df[args.sensitive] = df[args.sensitive].cat.codes

    if df[args.sensitive].nunique() <= 2:
        # Binary Encoding
        sensitive_data = df[args.sensitive].to_numpy().reshape(-1, 1)
        sensitive_size = 1
    else:
        # One-hot Encoding
        sensitive_data = pd.get_dummies(df[args.sensitive]).to_numpy().reshape(-1, df[args.sensitive].nunique())
        sensitive_size = sensitive_data.shape[1]
    
    train_data = np.concatenate((train_data, sensitive_data), axis =1)
    
    return train_data, transformer, sensitive_size

def converter_get_dataloader(
    data,
    shuffle=True,
    args=None,
):
    dataset = TensorDataset(
        torch.from_numpy(
            data.astype('float32')))
    
    loader = DataLoader(
        dataset,
        batch_size=args.converter_batch_size,
        shuffle=shuffle,
        drop_last=False
    )
    
    return loader

def dualfair_get_dataset(
    data_org,
    encoder,
    decoder,
    args=None
):
    trainds_list = []
    for s in data_org[args.sensitive].unique():
        trainds = DualFairDataset(
            data_org,
            encoder,
            decoder,
            condition=s,
            args=args,
        )
        trainds_list.append(trainds)
        
    random_trainds = DualFairDataset(
        data_org,
        encoder,
        decoder,
        condition=None,
        args=args
    )
    
    return trainds_list, random_trainds

def convert_sensitives(
    data,
    encoder,
    decoder,
    args=None
):
    s = args.sensitive
    transformer = args.transformer
    
    # Exclude the sensitive attribute
    ns_data = data[data.columns.difference([s])]
    
    # Transform the data
    ns_data = transformer.transform(ns_data)
    ns_data = torch.tensor(ns_data).to(args.gpu)
    
    # Make sensitive attribute to np array. Conduct One-hot encoding if needed.
    sensitive_size = args.sensitive_size
    s_data = torch.tensor(data[s].values).to(args.gpu)

    if sensitive_size > 1:
        si_data = s_data
        s_data = F.one_hot(s_data.long(), num_classes=sensitive_size)
    else:    
        s_data = s_data.reshape(-1, 1)
    
    # Combine non-sensitive data and sensitive data
    data = torch.cat((ns_data, s_data), dim=1).float()
    
    with torch.no_grad():
        mu, std, logvar = encoder(data[:, :-sensitive_size])
        eps = torch.randn_like(std)
        emb = eps * std + mu

        if sensitive_size == 1:
            converted_sensitives = 1 - data[:, -1:]
            changed_sensitives = converted_sensitives.long()
        else:
            random_idx = torch.randint(1, sensitive_size, (data.size(0),)).to(args.gpu)
            changed_sensitives = torch.remainder((si_data + random_idx), sensitive_size)
            converted_sensitives = F.one_hot(changed_sensitives, num_classes=sensitive_size)
            
        changed_sensitives = changed_sensitives.squeeze().detach().cpu().numpy()
                
        emb_cat = torch.cat((emb, converted_sensitives), dim=1)
        fake, sigmas = decoder(emb_cat)
        fake = torch.tanh(fake)
        
    counterfactual_df = transformer.inverse_transform(fake.detach().cpu(), sigmas.detach().cpu().numpy())
    counterfactual_df[s] = changed_sensitives
    
    return counterfactual_df

class DualFairDataset(torch.utils.data.Dataset):
    def tabmix(self, data):
#         column_num = len(data.columns)
        n_row, n_col = data.shape
        random_idx = torch.randint(0, len(self.all_data), (len(data),))
        random_row = torch.index_select(self.all_data, dim=0, index=random_idx)

        column_num = len(self.cat_columns) + len(self.con_columns)
        mask = []
        for r in range(n_row):
            mask_idx = np.random.choice(column_num, int(column_num * self.args.k), replace=False)
            mask_row = torch.zeros(column_num)
            mask_row[mask_idx] = 1.0
            mask.append(mask_row)
        mask = torch.stack(mask, dim=0)
        
        r_mask = []
        for c in range(column_num):
            n = self.portions[c]
            mask_col = mask[:, c]
            r_mask_col = mask_col.unsqueeze(dim=1).repeat(1, n)
            r_mask.append(r_mask_col)
        r_mask = torch.cat(r_mask, dim=1)
        
        ret = random_row * r_mask + data * (1 - r_mask)
                
        return ret
    
    def update_perturb(self):
        # Create counter-factual samples
        self.counter_df = convert_sensitives(self.c_df, self.encoder, self.decoder, self.args)     
        counter_cat_data = self.counter_df[self.cat_columns]
        counter_cat_data = torch.tensor(np.array(self.onehot_enc.transform(counter_cat_data)))
        counter_con_data = self.counter_df[self.con_columns]
        counter_con_data = torch.tensor(np.array(self.scaler.transform(counter_con_data)))
        self.counter_data = torch.cat((counter_cat_data, counter_con_data), dim=1)
        
        self.perturb_data = self.tabmix(self.refined_data)
    
    def __init__(
        self,
        df,
        encoder,
        decoder,
        condition,
        args=None
    ):
#         self.df_org = df
        self.df = df[df.columns.difference([args.target])]
        self.df_size = len(self.df)
        self.condition = condition
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

        # Prepare categorical data
        cat_data = self.df.select_dtypes(include=['object'])
        self.cat_columns = cat_data.columns.difference([args.sensitive])
        cat_data = self.df[self.cat_columns]

        # Prepare continuous data
        self.con_columns = self.df.columns.difference(self.cat_columns)
        self.con_columns = self.con_columns.difference([args.sensitive])
        con_data = self.df[self.con_columns]
        
        # Prepare and scaler
        self.onehot_enc = OneHotEncoder(sparse=False)
        self.onehot_enc.fit(cat_data)

        self.scaler = StandardScaler()
        self.scaler.fit(con_data)
        
        if self.condition != None:
            self.c_df = self.df.groupby(args.sensitive).get_group(condition)
        else:
            self.c_df = self.df
        
        # Conduct one-hot encoding and scaling for all condition
        cat_data = self.df[self.cat_columns]
        cat_uniques = cat_data.nunique().values
        cat_data = torch.tensor(np.array(self.onehot_enc.transform(cat_data)))
        con_data = self.df[self.con_columns]
        con_data = torch.tensor(np.array(self.scaler.transform(con_data)))
        self.all_data = torch.cat((cat_data, con_data), dim=1)
        
        self.portions = list(cat_uniques) + ([1] * len(self.con_columns))
        
        # Conduct one-hot encoding and scaling for current condition
        cat_data = self.c_df[self.cat_columns]
        cat_data = torch.tensor(np.array(self.onehot_enc.transform(cat_data)))
        con_data = self.c_df[self.con_columns]
        con_data = torch.tensor(np.array(self.scaler.transform(con_data)))
        self.refined_data = torch.cat((cat_data, con_data), dim=1)
        
        self.update_perturb()
                                  
    def __len__(
        self,
    ):
        return len(self.c_df)
    
    def __getitem__(
        self,
        idx,
    ):
        origin = self.refined_data[idx]
        perturb = self.perturb_data[idx]
        counter = self.counter_data[idx]
        
        return origin, perturb, counter
    
class EvaluateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        enc=None,
        scaler=None,
        args=None
    ):
        self.df = df[df.columns.difference([args.target])]
        
        cat_data = self.df.select_dtypes(include=['object'])
        self.cat_columns = cat_data.columns.difference([args.sensitive]).to_list()            
        cat_data = self.df[self.cat_columns]
        
        if enc != None:
            self.enc = enc
        else:
            self.enc = OneHotEncoder(sparse=False)
            self.enc.fit(cat_data)
        
        self.con_columns = self.df.columns.difference(self.cat_columns)
        self.con_columns = self.con_columns.difference([args.sensitive])
        con_data = self.df[self.con_columns]
        
        if scaler != None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(con_data)
            
        self.columns = self.df.columns           
        self.row_cat_df = np.array(self.enc.transform(self.df[self.cat_columns]))
        self.row_con_df = np.array(self.scaler.transform(self.df[self.con_columns]))
        self.row_df = np.concatenate((self.row_cat_df, self.row_con_df), axis=1)
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
#         row_item = self.df.iloc[idx]
#         row_item_final = self.row_item_final_df[idx, :]
        row_item = self.row_df[idx, :]
        return row_item
    
def get_evaluation_dataloader(
    trainX,
    trainY,
    testX,
    testY,
    trainconvertX,
    trainconvertY,
    testconvertX,
    testconvertY,
    args,
):
    train_finetune_ds = TensorDataset(trainX, trainY)
    test_finetune_ds = TensorDataset(testX, testY)
    train_finetune_loader = DataLoader(
        train_finetune_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_finetune_loader = DataLoader(
        test_finetune_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    train_convert_finetune_ds = TensorDataset(trainconvertX, trainconvertY)
    test_convert_finetune_ds = TensorDataset(testconvertX, testconvertY)
    train_convert_finetune_loader = DataLoader(
        train_convert_finetune_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_convert_finetune_loader = DataLoader(
        test_convert_finetune_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    return train_finetune_loader, test_finetune_loader, train_convert_finetune_loader, test_convert_finetune_loader