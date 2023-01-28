from utils import parse_arguments, reset_seed
from dataset import load_data, get_train_data, dualfair_get_dataset
from model import get_encoder, get_decoder, get_model
from train_eval import train_dualfair
import torch
import os
import time
from time import gmtime, strftime

if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    args.gpu = torch.device(f'cuda:{args.gpu}')
    reset_seed(args.seed)
    
    args.compress_dim = [int(x) for x in args.generator_dim.split(',')]
    args.decompress_dim = [int(x) for x in args.discriminator_dim.split(',')]
    print(f'Arguments:\n{args}\n')
    
    current_time = strftime("%m%d:%H:%M", gmtime())
    save_pre = f'dualfair_{args.dataset}_{args.sensitive}_seed_{args.seed}_{current_time}'

    if os.path.isdir(args.output) == False:
        os.mkdir(args.output)
        
    if os.path.isdir(f'{args.output}/dualfair') == False:
        os.mkdir(f'{args.output}/dualfair')
        
    output_dir = f'{args.output}/dualfair/{save_pre}'
    
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    
    print(f'Output directory ready')
    
    data_org, discrete_columns, target = load_data(args=args)
    args.discrete_columns = discrete_columns
    args.target = target
    print(f'Dataset ready')
    
    converter_path = f'./output/converter/converter_combined_{args.dataset}_{args.sensitive}'
    train_data, transformer, sensitive_size = get_train_data(
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
    
    trainds_list, random_trainds = dualfair_get_dataset(
        data_org,
        encoder,
        decoder,
        args=args
    )
    
    print("dataset load finished")
    
    origin, perturb, counter = trainds_list[0][0]
    input_dim, = origin.shape
    args.input_dim = input_dim

    model = get_model(args)
    model = model.to(args.gpu)
    
    print(args)
    
    train_dualfair(
        model,
        trainds_list,
        random_trainds,
        output_dir,
        args=args
    )
    
    end = time.time()
    print("elapsed_time: ", end - start)

    torch.save(model.state_dict(), "{}/model_pretrain_final.ckpt".format(output_dir))
