from utils import parse_arguments, reset_seed
from dataset import load_data, get_train_data, converter_get_dataloader
from model import get_encoder, get_decoder, get_discriminator
from train_eval import train_converter, eval_converter
import torch
import os

if __name__ == "__main__":
    args = parse_arguments()
    args.gpu = torch.device(f'cuda:{args.gpu}')
    reset_seed(7)
    
    args.compress_dim = [int(x) for x in args.generator_dim.split(',')]
    args.decompress_dim = [int(x) for x in args.discriminator_dim.split(',')]
    print(args)
    
    save_pre = f'converter_combined_{args.dataset}_{args.sensitive}'
    if os.path.isdir(args.output) == False:
        os.mkdir(args.output)
        
    if os.path.isdir(f'{args.output}/converter') == False:
        os.mkdir(f'{args.output}/converter')
        
    output_dir = f'{args.output}/converter/{save_pre}'
    
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    
    data_org, discrete_columns, target = load_data(args=args)
    args.discrete_columns = discrete_columns
    args.target = target
    
    train_data, transformer, sensitive_size = get_train_data(
        data_org,
        transformer_path=f'{output_dir}/transformer.ckpt',
        converter=True,
        args=args
    )

    args.data_dim = transformer.output_dimensions
    args.transformer = transformer
    
    args.sensitive_size = sensitive_size
    if sensitive_size == 1:
        args.class_num = 2
    else:
        args.class_num = sensitive_size
        
    loader = converter_get_dataloader(train_data, args=args)
    encoder = get_encoder(args).to(args.gpu)
    decoder = get_decoder(args).to(args.gpu)
    discriminator = get_discriminator(args).to(args.gpu)
    
    encoder, decoder, discriminator = train_converter(
        loader,
        encoder,
        decoder,
        discriminator,
        args=args
    )
    
    torch.save(encoder.state_dict(), f"{output_dir}/encoder.ckpt")
    torch.save(decoder.state_dict(), f"{output_dir}/decoder.ckpt")
    
    memoryloader = converter_get_dataloader(train_data, shuffle=False, args=args)
    counterfactual_df = eval_converter(
        memoryloader,
        encoder,
        decoder,
        args=args
    )
    
    counterfactual_df.to_csv(f'{output_dir}/counterfactual.csv', index=False)
    
    print(data_org.head(2)[data_org.columns.difference([args.target])])
    print(counterfactual_df[data_org.columns.difference([args.target])].head(2))
