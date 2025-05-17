from model.PPGN import IFA_MatchingNet
from util.utils import count_params, set_seed, mIOU
import argparse
import einops
from einops import repeat
from copy import deepcopy
import os
import time
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from tqdm import tqdm
from data.dataset import FSSDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root', type=str, default='../dataset', help='root path of training dataset')
    parser.add_argument('--dataset',  type=str,  default="lung", choices=['isic', 'deepglobe', 'isic', 'lung'], help='training dataset')
    parser.add_argument('--batch-size', type=int,  default=12, help='batch size of training')
    parser.add_argument('--lr',  type=float,  default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay',  type=float,  default=0.001, help='weight_decay')
    parser.add_argument('--crop-size',  type=int, default=473, help='cropping size of training samples')
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50', help='backbone of semantic segmentation model')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument('--episode', type=int, default=6000, help='total episodes of training')
    parser.add_argument('--snapshot', type=int, default=1200, help='save the model after each snapshot episodes')
    parser.add_argument('--seed', type=int, default=0, help='random seed to generate tesing samples')
    args = parser.parse_args()
    return args


def kl_divergence(mu, logvar):
    kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=[1, 2, 3])
    return kl_loss.mean()


def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_s_list = img_s_list.permute(1, 0, 2, 3, 4)
        mask_s_list = mask_s_list.permute(1, 0, 2, 3)
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()

        img_q, mask_q = img_q.cuda(), mask_q.cuda()

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()


        cls = cls + 1 # cls.shape: (b)
        cls = repeat(cls, 'b -> b h w', h=mask_q.shape[1], w=mask_q.shape[2]).cuda() # cls: (b, h, w)
        with torch.no_grad():
            pred = model(img_s_list, mask_s_list, img_q, mask_q)[0]
            pred = torch.argmax(pred, dim=1)

        pred[pred == 1] = cls[pred == 1] # pred: (b, h, w)
        mask_q[mask_q == 1] = cls[mask_q == 1].to(dtype = mask_q.dtype) # mask_q: (b, h, w)

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    save_path = './outdir/PPGN/%s/train' % (args.dataset)
    os.makedirs(save_path, exist_ok=True)

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    trainloader = FSSDataset.build_dataloader('pascal', args.batch_size, 4, 4, 'trn', args.shot)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    model = IFA_MatchingNet(args.backbone)

    print('\nParams: %.1fM' % count_params(model))

    for param in model.dinov2_vit_base.parameters():
        param.requires_grad = False
    for param in model.dinov2_vit_base.encoder.layer[11].parameters():
        param.requires_grad = True
    for param in model.dinov2_vit_base.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    total_params = 0
    total_trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            total_trainable_params += param.numel()

    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {total_trainable_params}")

    loss_ce = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    model = DataParallel(model).cuda()
    best_model = None
    previous_best = 0

    # each snapshot is considered as an epoch
    for epoch in range(args.episode // args.snapshot):
        print("\n==> Epoch %i\t\t learning rate = %.6f\t\t weight_decay= %.6f\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(epoch + args.seed)

        for i, (img_s_list, mask_s_list, img_q, mask_q, _, _, _) in enumerate(tbar):

            img_s_list = img_s_list.permute(1, 0, 2, 3, 4)
            mask_s_list = mask_s_list.permute(1, 0, 2, 3)
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()

            img_q, mask_q = img_q.cuda(), mask_q.cuda()
            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
                img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

            mask_s = torch.cat(mask_s_list, dim=0)
            mask_s = mask_s.long()

            (query_out, pred_res_query_3, pred_res_query_2, pred_res_query_1, pred_dinov2_query_3, pred_dinov2_query_2,
             pred_dinov2_query_1, mean_query_fp, log_var_query_fp, mean_query_bp, log_var_query_bp) = (
                model(img_s_list=img_s_list, mask_s_list=mask_s_list, img_q=img_q, mask_q=mask_q))

            loss_query_out = loss_ce(query_out, mask_q)

            loss_pred_res_query_3 = loss_ce(pred_res_query_3, mask_q)
            loss_pred_res_query_2 = loss_ce(pred_res_query_2, mask_q)
            loss_pred_res_query_1 = loss_ce(pred_res_query_1, mask_q)

            loss_pred_dinov2_query_3 = loss_ce(pred_dinov2_query_3, mask_q)
            loss_pred_dinov2_query_2 = loss_ce(pred_dinov2_query_2, mask_q)
            loss_pred_dinov2_query_1 = loss_ce(pred_dinov2_query_1, mask_q)

            kl_loss_mean_query_fp = kl_divergence(mean_query_fp, log_var_query_fp)
            kl_loss_mean_query_bp = kl_divergence(mean_query_bp, log_var_query_bp)

            loss = (loss_query_out + loss_pred_res_query_3 + loss_pred_res_query_2 + loss_pred_res_query_1 + loss_pred_dinov2_query_3
                    + loss_pred_dinov2_query_2 + loss_pred_dinov2_query_1 + kl_loss_mean_query_fp + kl_loss_mean_query_bp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        if epoch > 0 and epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] /= 2.0
            optimizer.param_groups[0]['weight_decay'] *= 2.0

        model.eval()
        set_seed(args.seed + epoch)
        miou = evaluate(model, testloader, args)

        # if epoch >= 2:
        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, miou)))

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_avg_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))

if __name__ == '__main__':
    main()