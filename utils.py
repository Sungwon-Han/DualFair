import argparse
import random
import numpy as np
import torch

def reset_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use')

    parser.add_argument('--no-header', dest='header', action='store_false',
                        help='The CSV file has no header. Discrete columns will be indices.')
    parser.add_argument('-d', '--discrete',
                    help='Comma separated list of discrete columns without whitespaces.')
    
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for the main model.')
    parser.add_argument('--converter_epochs', type=int, default=300,
                        help='Number of epochs for the counterfactual converter')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch Size for the main model. Must be an even number.')
    parser.add_argument('--converter_batch_size', type=int, default=500,
                        help='Batch Size for the counterfactual converter. Must be an even number.')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the main model and autoencoder in the counterfactual converter.')
    parser.add_argument('--discriminator_learning_rate', type=float, default=1e-2,
                        help='Learning rate for discriminator in the counterfactual converter.')
    
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for the main model.')
    parser.add_argument('--discriminator_weight_decay', type=float, default=0.0,
                        help='Weight decay for the discriminator in the counterfactual converter')
    parser.add_argument('--converter_weight_decay', type=float, default=1e-5,
                        help='Weight decay for the autoencoder in the counterfactual converter')
    
    parser.add_argument('--loss_factor', type=int, default=2,
                        help='loss_factor')
    
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')
    
    parser.add_argument('--k', type=float, default=0.5,
                        help='Feature ratio for TabMIX')
    parser.add_argument('--sensitive', default='sex', type=str,
                        help='Name of sensitive attribute')
    parser.add_argument('--dataset', default='adult', type=str,
                        help='Name of sensitive attribute')
    
    parser.add_argument('--output', default='./output', type=str,
                        help='Output path')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for preprocessing')
    parser.add_argument('--save_pre', default=None, type=str,
                        help='Target to evaluate')
    
    if notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    return args
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
