import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
import torch.nn.functional as F

def get_model(args=None):
    return Classifierv6(input_dim=args.input_dim, output_dim=args.class_num)

def get_encoder(args=None):
    return Encoder(args.data_dim, args.compress_dim, args.embedding_dim)

def get_decoder(args=None):
    return Decoder(args.embedding_dim + args.sensitive_size, args.compress_dim, args.data_dim)

def get_discriminator(args=None):
    return Discriminator(args.embedding_dim, args.class_num)

class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma

    
class Discriminator(Module):
    def __init__(self, embedding_dim, sensitive_dim):
        super(Discriminator, self).__init__()
        self.linear = Linear(embedding_dim, sensitive_dim)

    def forward(self, input):
        return self.linear(input)
    
class Classifierv6(nn.Module):
    def __init__(self, input_dim, output_dim=2, embed_dim=256, hidden_dim=256):
        super(Classifierv6, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        self.g1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.g2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )        
        

    def forward(self, x):
        feature = self.f(x)
        g_out = self.g(feature)
        g1_out = self.g1(g_out)
        g2_out = self.g2(g_out)
        return feature, g1_out, g2_out, g_out