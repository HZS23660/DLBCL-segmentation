import argparse
import logging
import os
import random
import shutil
import sys
import time
import pandas as pd


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import SimpleITK as sitk

from dataloader.Lymphoma_3D_dataset_P import (Lymphoma_3D_prior, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor, TwoStreamBatchSampler)

from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case, test_all_case_prior, test_all_case_prior_N


basicdir = '/home/tju/Documents/HZS/PET_lymphoma/PET_lymphoma_PW_new/'
parser = argparse.ArgumentParser()
parser.add_argument('--root_trainpath', type=str,
                    default=basicdir + 'PET_lymphoma_TrainData_prior_120_128 64 64', help='Name of Experiment')
parser.add_argument('--root_valpath', type=str,
                    default=basicdir + 'PET_lymphoma_ValData_prior', help='Name of Experiment')
parser.add_argument('--root_valweakpath', type=str,
                    default=basicdir + 'PET_lymphoma_ValData_prior_60', help='Name of Experiment')
parser.add_argument('--root_modelpath', type=str,
                    default='/home/tju/Documents/HZS/Code_huang/model/PET_lymphoma_3D/WSS_l0.5_a0.2_60_labeled_128 64 64/vnet_att/vnet_att_best_model.pth', \
                                        help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='PET_lymphoma_3D_PW_new/WSN_l0.5_a0.2', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='VNet_att', help='model_name')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 64, 64],  # D H W
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')


# label and unlabel
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=60,
                    help='labeled data')
parser.add_argument('--total_num', type=int, default=120,
                    help='total data')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output_num_classes')
parser.add_argument('--val_num', type=int, default=20,
                    help='validation data')

# costs
parser.add_argument('--consistency', type=float,
                    default=0.5, help='consistency')
parser.add_argument('--alpha', type=float,
                    default=0.2, help='alpha')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, epochs):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, epochs)


# def updata_model(net, ema=False):
#     # Network definition
#     model = net
#     if ema:
#         for param1, param2 in zip(model.parameters(), net.parameters()):
#             param1 = param2.detach()
#     return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    model = net_factory_3d(net_type=args.model, in_chns=2, class_num=args.num_classes)
    # model.load_state_dict(torch.load(args.root_modelpath))
    prior_dir = os.path.join(args.root_trainpath, 'prior_save')

    dice = []
    ce = []
    SS = []
    WS = []
    DS = []
    Sup = []
    all_cosloss = []

    # val_weak_dice = []
    # val_weak_iou = []
    # val_weak_tpr = []


    db_train = Lymphoma_3D_prior(base_dir=args.root_trainpath,
                             prior_dir=prior_dir,
                             split='train',
                             # num=70,
                             transform=transforms.Compose([
                                 RandomRotFlip(),
                                 RandomCrop(args.patch_size),
                                 ToTensor(),
                             ]))

    labeled_idxs = list(range(0, args.labeled_num * 27))
    unlabeled_idxs = list(range(args.labeled_num * 27, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # featuresavepath = (snapshot_path + '/feature/')
    # if os.path.exists(featuresavepath):
    #     shutil.rmtree(featuresavepath)
    # os.makedirs(featuresavepath)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    cos_loss = losses.Cosine_Similarity_Loss(args.num_classes, args.alpha)
    # Re_SS_Cos_Loss = losses.Re_SS_Cos_Loss(args.num_classes, args.alpha)
    Re_WS_Cos_Loss = losses.Re_WS_Cos_Loss(args.num_classes)
    # ce_loss = losses.FocalLoss()

    total_iter_num = max_epochs * len(trainloader)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch, total iterations number{} ".format(len(trainloader), total_iter_num))

    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(0, max_epochs), ncols=50)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_CT, volume_PET, label_batch = sampled_batch['image_CT'], sampled_batch['image_PET'], sampled_batch['label']
            prior_CT, prior_PET = sampled_batch['prior_CT'], sampled_batch['prior_PET']
            prior_tumor_batch = sampled_batch['prior_label']
            # print('prior_tumor_batch:', torch.sum(prior_tumor_batch))
            prior_data_batch = torch.cat((prior_CT, prior_PET), dim=1)
            # prior_data_batch = F.interpolate(prior_data_batch, size=(args.patch_size[0], args.patch_size[1], \
            #                          args.patch_size[2]), mode='trilinear', align_corners=True)
            prior_data_batch = prior_data_batch.cuda()
            prior_tumor_batch = prior_tumor_batch.cuda()
            # prior_tumor_batch = F.interpolate(prior_tumor_batch.unsqueeze(1).type(torch.float32), size=(args.patch_size[0], args.patch_size[1], \
            #                                                args.patch_size[2]), mode='trilinear', align_corners=True)
            # prior_tumor_batch = prior_tumor_batch.squeeze(1)
            #----------------------带#为重采样代码-----------------------

            volume_batch = torch.cat((volume_CT, volume_PET), dim=1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(label_batch.min().item(), label_batch.max().item())

            outputs, model_feature, prior_feature = model(volume_batch, prior_data_batch)
            # outputs, model_feature = model(volume_batch)

            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss1 = 0.5 * (loss_dice + loss_ce)

            lamba = get_current_consistency_weight(iter_num, total_iter_num)
            supervised_loss2 = cos_loss(model_feature[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            # consistency_loss = Re_SS_Cos_Loss(model_feature[args.labeled_bs:], outputs_soft[args.labeled_bs:])
            weak_supervised_loss = Re_WS_Cos_Loss(prior_feature[args.labeled_bs:], prior_tumor_batch.unsqueeze(1)[args.labeled_bs:], \
                                                model_feature[args.labeled_bs:], outputs_soft[args.labeled_bs:])
            all_consistency_loss = supervised_loss2 + weak_supervised_loss# + consistency_loss

            # lamba = 0.5
            loss = supervised_loss1 + lamba * all_consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #学习率更新
            lr_ = base_lr * (1.0 - epoch_num / max_epochs) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/supervised_loss1', supervised_loss1, iter_num)
            writer.add_scalar('info/weak_supervised_loss', weak_supervised_loss, iter_num)
            # writer.add_scalar('info/semi_supervised_loss', consistency_loss, iter_num)
            writer.add_scalar('info/supervised_loss2', supervised_loss2, iter_num)
            writer.add_scalar('info/all_consistency_loss', all_consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, all_consistency_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), all_consistency_loss.item()))

            # SS.append(consistency_loss.item())
            WS.append(weak_supervised_loss.item())
            DS.append(supervised_loss2.item())
            all_cosloss.append(all_consistency_loss.item())
            dice.append(loss_dice.item())
            ce.append(loss_ce.item())
            Sup.append(supervised_loss1.item())

            # if (epoch_num+1) % 50 == 0 and i_batch == 60:  # 保存feature时的训练batchsize必须设为1
            #     volume_PET, label_batch = volume_PET.squeeze().detach().cpu().numpy(), label_batch.squeeze().detach().cpu().numpy()
            #     generation_mask = generation_mask.squeeze().detach().cpu().numpy()
            #     img_itk = sitk.GetImageFromArray(volume_PET)
            #     img_itk.SetSpacing((1.0, 1.0, 1.0))
            #     img_itk.SetOrigin((0, 0, 0))
            #     sitk.WriteImage(img_itk, featuresavepath + "{}_pet.nii.gz".format(epoch_num))
            #
            #     img_itk = sitk.GetImageFromArray(label_batch.astype(np.uint8))
            #     img_itk.SetSpacing((1.0, 1.0, 1.0))
            #     img_itk.SetOrigin((0, 0, 0))
            #     sitk.WriteImage(img_itk, featuresavepath + "{}_label.nii.gz".format(epoch_num))
            #
            #     img_itk = sitk.GetImageFromArray(generation_mask)
            #     img_itk.SetSpacing((1.0, 1.0, 1.0))
            #     img_itk.SetOrigin((0, 0, 0))
            #     sitk.WriteImage(img_itk, featuresavepath + "{}_feature.nii.gz".format(epoch_num))


        model.eval()

        avg_metric = test_all_case_prior_N(model, args.root_valpath, val_num=args.val_num, num_classes=2,
                                   patch_size=args.patch_size, stride_xy=32, stride_z=64)
        if avg_metric[:, 0].mean() > best_performance:
            best_performance = avg_metric[:, 0].mean()
            save_mode_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}.pth'.format(
                                              epoch_num, round(best_performance, 4)))
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)

        # writer.add_scalar('info/val_dice_score',
        #                   avg_metric[0, 0], iter_num)
        # writer.add_scalar('info/val_hd95',
        #                   avg_metric[0, 1], iter_num)
        logging.info('epoch %d : iteration %d : dice_score : %f hd95 : %f' % \
                     (epoch_num, iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))


        #----------------------------------
        # logging.info('val_weakdata_dice')
        #
        # avg_metric1 = test_all_case_prior_N(model, args.root_valweakpath, val_num=60, num_classes=2,
        #                                    patch_size=args.patch_size, stride_xy=32, stride_z=64)
        # dice_mean = avg_metric1[:, 0]
        # iou_mean = avg_metric1[:, 1]
        # tpr_mean = avg_metric1[:, 2]
        # val_weak_dice.append(dice_mean)
        # val_weak_iou.append(iou_mean)
        # val_weak_tpr.append(tpr_mean)

        #----------------------------------
        model.train()


        if epoch_num >= max_epochs:
            iterator.close()
            break

    writer.close()
    frame = pd.DataFrame({'SS': SS, 'WS': WS, 'DS': DS, 'all_cosloss': all_cosloss,
                          'dice': dice, 'ce': ce, 'Sup': Sup})
    frame.to_csv(snapshot_path + "/loss_all.csv")

    # frame1 = pd.DataFrame({'val_weak_dice': val_weak_dice, 'val_weak_iou': val_weak_iou
    #                           ,'val_weak_tpr': val_weak_tpr})
    # frame1.to_csv(snapshot_path + "/val_weaklabel.csv")

    return "Training Finished!"


if __name__ == "__main__":

    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
    # torch.backends.cudnn.benchmark = True
    #
    # 如果想要避免这种结果波动，设置：
    # torch.backends.cudnn.deterministic = True

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/{}_{}_labeled_{} {} {}/{}".format(args.exp, args.labeled_num,\
                            args.patch_size[0], args.patch_size[1], args.patch_size[2], args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)