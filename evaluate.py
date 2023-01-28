import os
import numpy as np
import torch
from utils import parse_arguments, reset_seed
from model import get_encoder, get_decoder, get_model
from dataset import load_data, get_train_data, EvaluateDataset, convert_sensitives, get_evaluation_dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import itertools

def avgPairs(lst): 
    diffs = [abs(e[1] - e[0]) for e in itertools.combinations(lst, 2)]
    return np.mean(diffs)

def get_embed(
    loader,
    args=None
):
    f_list = []
    g1_list = []
    g2_list = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(args.gpu), y.to(args.gpu)
            f_embed, g1_embed, g2_embed, _ = model(X.float())
            f_list.append(f_embed.detach().cpu().numpy())
            g1_list.append(g1_embed.detach().cpu().numpy())
            g2_list.append(g2_embed.detach().cpu().numpy())
            
    f_embed = np.concatenate(f_list, axis=0)
    g1_embed = np.concatenate(g1_list, axis=0)
    g2_embed = np.concatenate(g2_list, axis=0)
    
    return f_embed, g1_embed, g2_embed

def do_regression(
    train_embed,
    trainY,
    test_embed,
    test_convert_embed,
    args=None,
):
    if args.eval_type == 'classification':
        regression_model = LogisticRegression(solver='liblinear')
        regression_model.fit(train_embed, trainY.numpy())
        y_pred = regression_model.predict(test_embed)
        y_pred_proba = regression_model.predict_proba(test_embed)
        y_pred_proba2 = regression_model.predict_proba(test_convert_embed)
        
        return regression_model, y_pred, y_pred_proba, y_pred_proba2
    
    elif args.eval_type == 'regression':
        regression_model = RandomForestRegressor(max_depth=10, random_state=0)
        regression_model.fit(train_embed, trainY)
        y_pred = regression_model.predict(test_embed)
        y_pred2 = regression_model.predict(test_convert_embed)
    
        return regression_model, y_pred, y_pred2
    
    else:
        assert(0)

def eval_classification(
    train_embed,
    trainY,
    test_embed,
    test_convert_embed,
    testY,
    args=None,
):
    regression_model, y_pred, y_pred_proba, y_pred_proba2 = do_regression(
        train_embed,
        trainY,
        test_embed,
        test_convert_embed,
        args=args,
    )

    accuracy = accuracy_score(y_pred, testY.numpy())

    AUROC = roc_auc_score(testY.numpy(), y_pred_proba[:, 1])

    testds.df['predict_linear'] = y_pred_proba[:, 1]
    predict_mean_vals = testds.df.groupby(args.sensitive).mean()['predict_linear'].values
    DP = avgPairs(predict_mean_vals)

    testds.df['predict_convert_linear'] = y_pred_proba2[:, 1]
    CP = (testds.df['predict_linear'] - testds.df['predict_convert_linear']).abs().mean()

    equalize_odd_list = []
    for i, grouped in testds.df.groupby(args.sensitive):
        numy = len(grouped[grouped['predict_linear'] >= 0.5])
        totaly = len(grouped)
        equalize_odd_list.append(numy / totaly)
    Eodd = avgPairs(equalize_odd_list)
    
    return accuracy, AUROC, DP, CP, Eodd

def eval_regression(
    train_embed,
    trainY,
    test_embed,
    test_convert_embed,
    testY,
    args=None,
):
    min_max_scaler = preprocessing.MinMaxScaler()
    trainY = min_max_scaler.fit_transform(trainY.reshape(-1, 1)).squeeze()
    testY = min_max_scaler.transform(testY.reshape(-1, 1)).squeeze()
    
    regression_model, y_pred, y_pred2 = do_regression(
        train_embed,
        trainY,
        test_embed,
        test_convert_embed,
        args=args,
    )

    RMSE = mean_squared_error(testY, y_pred)
    
    testds.df['predict_linear'] = y_pred
    predict_mean_vals = testds.df.groupby(args.sensitive).mean()['predict_linear'].values
    DP = avgPairs(predict_mean_vals)

    testds.df['predict_convert_linear'] = y_pred2
    CP = (testds.df['predict_linear'] - testds.df['predict_convert_linear']).abs().mean()
    
    return RMSE, DP, CP

if __name__ == "__main__":
    args = parse_arguments()
    args.gpu = torch.device(f'cuda:{args.gpu}')
    reset_seed(args.seed)
    
    args.compress_dim = [int(x) for x in args.generator_dim.split(',')]
    args.decompress_dim = [int(x) for x in args.discriminator_dim.split(',')]
    
    if args.dataset in ['adult', 'compas', 'lsac', 'credit']:
        args.eval_type = 'classification'
    elif args.dataset in ['communities', 'student_performance']:
        args.eval_type = 'regression'
    else:
        assert(0)
    
    if args.save_pre == None:
        print('Automatically find save_pre...')
        save_pre_content = f'dualfair_{args.dataset}_{args.sensitive}_seed_{args.seed}'
        candidates = [f for f in os.listdir(f'{args.output}/dualfair') if f.startswith(save_pre_content)]
        candidates = sorted(candidates)
        args.save_pre = candidates[-1]
        print(f'Evaluate {args.save_pre}')
    
    output_dir = f'{args.output}/dualfair/{args.save_pre}'
    
    data_org, discrete_columns, target = load_data(args=args)
    args.discrete_columns = discrete_columns
    args.target = target
    print(f'Dataset ready')
    
    data_org[args.target] = data_org[args.target].astype('category')
    data_org[args.target] = data_org[args.target].cat.codes
    data_org[args.sensitive] = data_org[args.sensitive].astype('category')
    data_org[args.sensitive] = data_org[args.sensitive].cat.codes
    
    data_org_test, _, _ = load_data(train=False, args=args)
    data_org_test[args.target] = data_org_test[args.target].astype('category')
    data_org_test[args.target] = data_org_test[args.target].cat.codes
    data_org_test[args.sensitive] = data_org_test[args.sensitive].astype('category')
    data_org_test[args.sensitive] = data_org_test[args.sensitive].cat.codes
    
    converter_path = f'./output/converter/converter_combined_{args.dataset}_{args.sensitive}'
    _, transformer, sensitive_size = get_train_data(
        data_org,
        transformer_path=f'{converter_path}/transformer.ckpt',
        converter=False,
        args=args
    )
    
    args.data_dim = transformer.output_dimensions
    args.transformer = transformer
    
    args.sensitive_size = sensitive_size
    if sensitive_size == 1:
        args.class_num = 2
    else:
        args.class_num = sensitive_size
    print(f'Data transformation ready')
        
    encoder = get_encoder(args)
    decoder = get_decoder(args)
    print("Encoder, decoder created")

    encoder.load_state_dict(torch.load(f"{converter_path}/encoder.ckpt", map_location=args.gpu))
    decoder.load_state_dict(torch.load(f"{converter_path}/decoder.ckpt", map_location=args.gpu))
    encoder.to(args.gpu)
    decoder.to(args.gpu)
    print("encoder, decoder load finished")
    
    trainds = EvaluateDataset(data_org, args=args)
    testds = EvaluateDataset(data_org_test, enc=trainds.enc, scaler=trainds.scaler, args=args)
    
    train_convert_df = convert_sensitives(trainds.df, encoder, decoder, args=args)
    trainds.convert_cat_df = np.array(trainds.enc.transform(train_convert_df[trainds.cat_columns]))
    trainds.convert_con_df = np.array(trainds.scaler.transform(train_convert_df[trainds.con_columns]))
    trainds.convert_df = np.concatenate((trainds.convert_cat_df, trainds.convert_con_df), axis=1)
    
    test_convert_df = convert_sensitives(testds.df, encoder, decoder, args=args)
    testds.convert_cat_df = np.array(testds.enc.transform(test_convert_df[testds.cat_columns]))
    testds.convert_con_df = np.array(testds.scaler.transform(test_convert_df[testds.con_columns]))
    testds.convert_df = np.concatenate((testds.convert_cat_df, testds.convert_con_df), axis=1)
    
    trainX = torch.tensor(trainds.row_df)
    trainY = torch.tensor(data_org[args.target].to_numpy())

    testX = torch.tensor(testds.row_df)
    testY = torch.tensor(data_org_test[args.target].to_numpy())

    trainconvertX = torch.tensor(trainds.convert_df)
    trainconvertY = torch.tensor(data_org[args.target].to_numpy())

    testconvertX = torch.tensor(testds.convert_df)
    testconvertY = torch.tensor(data_org_test[args.target].to_numpy())
    
    torch.save(train_convert_df, f"{converter_path}/train_convert_df")
    torch.save(test_convert_df, f"{converter_path}/test_convert_df")
    
    x = trainX[0]
    args.input_dim = x.shape[0]
    print(f'Input shape: {x.shape}')
    
    train_finetune_loader, test_finetune_loader, \
    train_convert_finetune_loader, test_convert_finetune_loader = get_evaluation_dataloader(
        trainX,
        trainY,
        testX,
        testY,
        trainconvertX,
        trainconvertY,
        testconvertX,
        testconvertY,
        args,
    )
    
    if args.eval_type == 'classification':
        accuracy_list = []
        AUROC_list = []
        DP_list = []
        CP_list = []
        Eodd_list = []
        
    elif args.eval_type == 'regression':
        RMSE_list = []
        DP_list = []
        CP_list = []
    else:
        assert(0)
        
    model_list = [f"model_pretrain_epoch{200 - i}.ckpt" for i in range(10)]
    for model_path in model_list:
        model = get_model(args)
        model.load_state_dict(torch.load(f"{output_dir}/{model_path}", map_location=args.gpu))
        model = model.to(args.gpu)
        print(f'{model_path} load finished')
        
        model.eval()
        
        train_f_embed, train_g1_embed, train_g2_embed = get_embed(train_finetune_loader, args)
#         train_f_convert_embed, train_g1_convert_embed, train_g2_convert_embed = get_embed(train_convert_finetune_loader, args)
        test_f_embed, test_g1_embed, test_g2_embed = get_embed(test_finetune_loader, args)
        test_f_convert_embed, test_g1_convert_embed, test_g2_convert_embed = get_embed(test_convert_finetune_loader, args)
            
        train_embed = train_f_embed
        test_embed = test_f_embed
        test_convert_embed = test_f_convert_embed
        
        if args.eval_type == 'classification':
            accuracy, AUROC, DP, CP, Eodd = eval_classification(
                train_embed,
                trainY,
                test_embed,
                test_convert_embed,
                testY,
                args
            )
            
            accuracy_list.append(accuracy)
            AUROC_list.append(AUROC)
            DP_list.append(DP)
            CP_list.append(CP)
            Eodd_list.append(Eodd)
            
            print(f'Accuracy: {accuracy}')
            print(f'AUROC: {AUROC}')
            print(f'DP: {DP}')
            print(f'CP: {CP}')
            print(f'Eodd: {Eodd}')
            
        elif args.eval_type == 'regression':
            RMSE, DP, CP = eval_regression(
                train_embed,
                trainY,
                test_embed,
                test_convert_embed,
                testY,
                args
            )
            
            RMSE_list.append(RMSE)
            DP_list.append(DP)
            CP_list.append(CP)
            
            print(f'RMSE: {RMSE}')
            print(f'DP: {DP}')
            print(f'CP: {CP}')
        else:
            assert(0)

    print('-------------- Final Result -----------------')
    log = ''
    log += f'Arguments:\n'
    log += f'{args}\n\n'
    
    result = ''
    if args.eval_type == 'classification':
        result += f'Accuracy: {np.mean(accuracy_list)}\n'
        result += f'AUROC: {np.mean(AUROC_list)}\n'
        result += f'DP: {np.mean(DP_list)}\n'
        result += f'CP: {np.mean(CP_list)}\n'
        result += f'Eodds: {np.mean(Eodd_list)}\n'
        
    elif args.eval_type == 'regression':
        result += f'RMSE: {np.mean(RMSE_list)}\n'
        result += f'DP: {np.mean(DP_list)}\n'
        result += f'CP: {np.mean(CP_list)}\n'

    else:
        assert(0)
        
    print(result)
    log += f'{result}\n'
    
    with open(f'{output_dir}/eval_result', 'w') as f:
        print(log, file=f, flush=True)