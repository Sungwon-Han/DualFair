from utils import AverageMeter
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

import random
import torch.nn.functional as F

def loss_function(
    recon_x,
    x,
    sigmas,
    mu,
    logvar,
    args=None
):
    st = 0
    loss = []
    
    for column_info in args.transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn != "softmax":
                ed = st + span_info.dim
                std = sigmas[st]
                loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return sum(loss) * args.loss_factor / x.size()[0], KLD / x.size()[0]

def dist_match_loss(hidden, alpha=1.0):
    device = hidden.device
    hidden_dim = hidden.shape[-1]
    rand_w = torch.Tensor(np.eye(hidden_dim, dtype=np.float64)).to(device)
    loss_dist_match = get_swd_loss(hidden, rand_w, alpha)
    return loss_dist_match


def get_swd_loss(states, rand_w, alpha=1.0):
    device = states.device
    states_shape = states.shape
    states = torch.matmul(states, rand_w)
    states_t, _ = torch.sort(states.t(), dim=1)

    states_prior = torch.Tensor(np.random.normal(size = states_shape)).to(device) # (bsz, dim)
    states_prior = torch.matmul(states_prior, rand_w) # (dim, dim)
    states_prior_t, _ = torch.sort(states_prior.t(), dim=1) # (dim, bsz)
    return torch.mean(torch.sum((states_prior_t - states_t) ** 2, axis=0))


def js_loss(x1, x2, xa, t=0.1, t2=0.1):
    pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2  = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0

def train_epoch_dualfair(
    net,
    epoch,
    data_loader_list,
    random_loader,
    train_optimizer,
    args=None
):
    net.train()
    
    total_meter = AverageMeter()
    counter_meter = AverageMeter()
    dist_meter = AverageMeter()
    pos_meter = AverageMeter()
    
    random_iter = iter(random_loader)
    for data_loader in data_loader_list:
        for original, perturb, counter in data_loader:
            try:
                random_original, _, _ = next(random_iter)
            except:
                random_iter = iter(random_loader)
                random_original, _, _ = next(random_iter)
    
            original = original.to(args.gpu)
            perturb = perturb.to(args.gpu)
            counter = counter.to(args.gpu)
            random_original = random_original.to(args.gpu)

            bsz = original.size(0)

            total_input = torch.cat((original, perturb, counter, random_original)).float()
            _, g1_out_all, g2_out_all, g_out_all = net(total_input)

            g_out_original, _, g_out_perturb, _ = torch.split(g_out_all, len(g_out_all) // 4)
            g1_out_original, g1_out_perturb, _, g1_out_random = torch.split(g1_out_all, len(g1_out_all) // 4)
            g2_out_original, _, g2_out_counter, _ = torch.split(g2_out_all, len(g2_out_all) // 4)

            dist_loss = dist_match_loss(g2_out_original)
            counter_loss = F.mse_loss(g2_out_counter, g2_out_original)
            pos_loss = -torch.sum(F.normalize(g1_out_perturb, dim=-1) * F.normalize(g_out_original.detach()), dim=-1).mean()

            loss = counter_loss + dist_loss + pos_loss
                
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_meter.update(loss.item(), bsz)
            counter_meter.update(counter_loss.item(), bsz)
            dist_meter.update(dist_loss.item(), bsz)
            pos_meter.update(pos_loss.item(), bsz)

    log = ''
    log += f"Epoch: {epoch}"
    log += f", total: {total_meter.avg:.5f}"
    log += f", counter: {counter_meter.avg:.5f}"
    log += f", pos: {pos_meter.avg:.5f}"
    log += f", dist: {dist_meter.avg:.5f}"
    print(log, flush=True)
    
    return net
    
def train_dualfair(
    net,
    trainds_list,
    random_trainds,
    output_dir,
    args=None
):
    optimizer = Adam(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    random_loader = DataLoader(
        random_trainds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    for epoch in range(1, args.epochs + 1):
        train_loader_list = []
        for i in range(len(trainds_list)):
            train_loader = DataLoader(
                trainds_list[i],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True
            )
            train_loader_list.append(train_loader)
            
        net = train_epoch_dualfair(
            net,
            epoch,
            train_loader_list,
            random_loader,
            optimizer,
            args=args
        )
        
        for i in range(len(trainds_list)):
            trainds_list[i].update_perturb()
            
        if epoch % 100 == 0 or epoch > args.epochs-10:
            torch.save(net.state_dict(), f"{output_dir}/model_pretrain_epoch{epoch}.ckpt")
        

def train_converter(
    loader,
    encoder,
    decoder,
    discriminator,
    args=None
):
    optimizerAE = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        weight_decay=args.converter_weight_decay,
        lr=args.learning_rate,
    )
    
    optimizerDis = Adam(
        list(discriminator.parameters()),
        weight_decay=args.discriminator_weight_decay,
        lr=args.discriminator_learning_rate
    )
    
    pred_criterion = CrossEntropyLoss().to(args.gpu)
    losses_rec = AverageMeter()
    losses_kl = AverageMeter() 
    losses_cyc = AverageMeter() 
    losses_pred = AverageMeter()
    
    encoder.train()
    decoder.train()
    discriminator.train()
    
    tbar = tqdm(range(args.converter_epochs))
    for epoch in tbar:
        for id_, data in enumerate(loader):
            real = data[0].to(args.gpu)
            bsz = real.size(0)

            mu, std, logvar = encoder(real[:, :-args.sensitive_size])
            eps = torch.randn_like(std)
            emb = eps * std + mu

            # Discriminator step
            sensitive_logit = discriminator(emb.detach())
            loss_pred = pred_criterion(sensitive_logit, torch.argmax(real[:, -args.sensitive_size:], dim=1))
            optimizerDis.zero_grad()
            loss_pred.backward(retain_graph=True)
            optimizerDis.step()

            # Generator step
            sensitive_logit = discriminator(emb)
            loss_pred = pred_criterion(sensitive_logit, torch.argmax(real[:, -args.sensitive_size:], dim=1))
            emb_cat = torch.cat((emb, real[:, -args.sensitive_size:]), dim=1)
            rec, sigmas = decoder(emb_cat)
            loss_rec, loss_kl = loss_function(
                rec,
                real,
                sigmas,
                mu,
                logvar,
                args
            )

            # Cyclic step
            u_0 = emb

            if args.sensitive_size == 1:
                converted_sensitives = 1 - real[:, -1:]
                #changed_sensitives = converted_sensitives.long()
            else:
                si_data = torch.argmax(real[:, -args.sensitive_size:], dim=1)
                random_idx = torch.randint(1, args.sensitive_size, (real.size(0),)).to(args.gpu)
                changed_sensitives = torch.remainder((si_data + random_idx), args.sensitive_size)
                converted_sensitives = F.one_hot(changed_sensitives, num_classes=args.sensitive_size)

            u_0_s_ = torch.cat((u_0, converted_sensitives), dim=1)
            x_1_s_, _ = decoder(u_0_s_)
            x_1_s_ = torch.tanh(x_1_s_)
            mu_, std_, logvar_ = encoder(x_1_s_)
            eps_ = torch.randn_like(std_)
            u_1 = eps_ * std_ + mu_
            u_1_s = torch.cat((u_1, real[:, -args.sensitive_size:]), dim=1)
            x_2_s, sigmas_ = decoder(u_1_s)
               
            loss_cyc, _ = loss_function(
                x_2_s,
                real,
                sigmas_,
                mu_,
                logvar_,
                args
            )

            loss = loss_rec + loss_kl + loss_cyc - loss_pred
            optimizerAE.zero_grad()
            loss.backward()
            optimizerAE.step()

            losses_rec.update(loss_rec.item(), bsz)
            losses_kl.update(loss_kl.item(), bsz)
            losses_cyc.update(loss_cyc.item(), bsz)
            losses_pred.update(loss_pred.item(), bsz)
            
            tbar.set_description(
                f"Epoch-{epoch} / [Loss Info] rec:{losses_rec.avg:.4f}, kl:{losses_kl.avg:.4f}, cyc:{losses_cyc.avg:.4f}, pred:{losses_pred.avg:.4f}",
                refresh=True
            )
            
            tbar.refresh()
            decoder.sigma.data.clamp_(0.01, 1.0)
            
    return encoder, decoder, discriminator

def eval_converter(
    loader,
    encoder,
    decoder,
    args=None,
):
    encoder.eval()
    decoder.eval()
    
    data_all = []
    emb_all = []
    changed_sensitives = []
    
    with torch.no_grad():
        for id_, data in enumerate(loader):
            real = data[0].to(args.gpu)
            mu, std, logvar = encoder(real[:, :-args.sensitive_size])
            eps = torch.randn_like(std)
            emb = eps * std + mu
            emb_all.append(emb.detach().cpu().numpy())

            if args.sensitive_size == 1:
                converted_sensitives = 1 - real[:, -1:]
                changed_sensitives.append(converted_sensitives.long().cpu().numpy())
            else:
                _, si = torch.max(real[:, -args.sensitive_size:], dim=1)

                random_idx = []
                for i in range(real.size(0)):
                    r = si[i].item()
                    while (r==si[i].item()):
                        r = random.randint(0, args.sensitive_size-1)
                    random_idx.append(r)
                    
                changed_sensitives.append(np.array(random_idx))
                random_idx = torch.Tensor(random_idx).to(args.gpu)

                converted_sensitives = F.one_hot(random_idx.long(), args.sensitive_size)
                

            emb_cat = torch.cat((emb, converted_sensitives), dim=1)

            fake, sigmas = decoder(emb_cat)
            fake = torch.tanh(fake)
            data_all.append(fake.cpu().numpy())
        
    emb_all = np.concatenate(emb_all, axis=0)
    data_all = np.concatenate(data_all, axis=0)
    changed_sensitives = np.concatenate(changed_sensitives, axis=0).squeeze()
    counterfactual_df = args.transformer.inverse_transform(data_all, sigmas.detach().cpu().numpy())

    counterfactual_df[args.sensitive] = changed_sensitives
    
    return counterfactual_df
