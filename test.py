from model.PPGN import IFA_MatchingNet
from util.utils import count_params, set_seed, mIOU
import argparse
import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import glob
import einops
from einops import repeat
from data.dataset import FSSDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root', type=str,  default='../dataset', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='isic', choices=['fss', 'deepglobe', 'isic', 'lung'], help='training dataset')
    parser.add_argument('--backbone',  type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='backbone of semantic segmentation model')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument('--seed', type=int, default=0, help='random seed to generate tesing samples')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of testing')
    args = parser.parse_args()
    return args


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

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
            
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

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    model = IFA_MatchingNet(args.backbone)
    model = DataParallel(model)

    ### Please modify the following paths with your model path if needed.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PPGN/deepglobe/fine_tuning/resnet50_1shot_avg_48.93.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PPGN/deepglobe/fine_tuning/resnet50_5shot_avg_58.76.pth'
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PPGN/isic/fine_tuning/resnet50_1shot_avg_72.94.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PPGN/isic/fine_tuning/resnet50_5shot_avg_69.77.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PPGN/lung/fine_tuning/resnet50_1shot_avg_74.59.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PPGN/lung/fine_tuning/resnet50_5shot_avg_74.59.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PPGN/fss/fine_tuning/resnet50_1shot_avg_85.40.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PPGN/fss/fine_tuning/resnet50_5shot_avg_82.36.pth'


    print('Evaluating model:', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    print('\nParams: %.1fM' % count_params(model))

    best_model = DataParallel(model).cuda()

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

if __name__ == '__main__':
    main()

