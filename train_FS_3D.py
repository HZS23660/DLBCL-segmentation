import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import SimpleITK as sitk

from dataloader import utils
# from dataloader.BraTs2021_3D_dataset import (BraTs2021_3D, CenterCrop, RandomCrop,
#                                    RandomRotFlip, ToTensor,
#                                    TwoStreamBatchSampler)
from dataloader.Lymphoma_3D_dataset import (CT_PET_lymphoma_3D, Lymphoma_3D_prior, CenterCrop, RandomCrop,
                                            RandomRotFlip, ToTensor)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case, test_all_case_prior, test_all_case_prior_N




basicdir = '/home/tju/Documents/HZS/PET_lymphoma/PET_lymphoma_PW/'

parser = argparse.ArgumentParser()
parser.add_argument('--root_trainpath', type=str,
                    default=basicdir + 'PET_lymphoma_TrainData_prior_120_128 64 64', help='Name of Experiment')
parser.add_argument('--root_valpath', type=str,
                    default=basicdir + 'PET_lymphoma_ValData_prior', help='Name of Experiment')
parser.add_argument('--root_modelpath', type=str,
                    default='/home/tju/Documents/HZS/Code_huang/model/PET_lymphoma_3D/FS_120_labeled_256 128 128/vnet/vnet_best_model.pth', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='PET_lymphoma_3D/FS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet_att_new', help='model_name')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 64, 64],  # D H W
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=120,
                    help='labeled data')
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



def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    num_classes = 2 #2
    model = net_factory_3d(net_type=args.model, in_chns=2, class_num=num_classes)
    # model.load_state_dict(torch.load(args.root_modelpath))
    # db_train = CT_PET_lymphoma_3D(base_dir=args.root_trainpath,
    #                          split='train',
    #                          num=args.labeled_num,
    #                          transform=transforms.Compose([
    #                              RandomRotFlip(),
    #                              RandomCrop(args.patch_size),
    #                              ToTensor(),
    #                          ]))

    db_train = Lymphoma_3D_prior(base_dir=args.root_trainpath,
                             split='train',
                             # num=args.labeled_num,
                             transform=transforms.Compose([
                                 RandomRotFlip(),
                                 RandomCrop(args.patch_size),
                                 ToTensor(),
                             ]))


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # featuresavepath = (snapshot_path + '/feature/')
    # if os.path.exists(featuresavepath):
    #     shutil.rmtree(featuresavepath)
    # os.makedirs(featuresavepath)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    # cos_loss = losses.Cosine_Similarity_Loss(num_classes, args.alpha)
    cos_loss_ce = losses.CE_Cosine_Similarity_Loss()
    # ce_loss = losses.FocalLoss()

    total_iter_num = max_epochs * len(trainloader)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch, total iterations number{} ".format(len(trainloader), total_iter_num))

    iter_num = 0
    best_performance = 0.0
    iterator = tqdm(range(max_epochs), ncols=50)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_CT, volume_PET, label_batch = sampled_batch['image_CT'], sampled_batch['image_PET'], sampled_batch['label']
            volume_batch = torch.cat((volume_CT, volume_PET), dim=1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # outputs, model_feature = model(volume_batch)
            outputs, outputs_fc = model(volume_batch)
            # outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            # supervised_loss2 = cos_loss(model_feature, label_batch.unsqueeze(1))
            supervised_loss2 = cos_loss_ce(outputs_fc, label_batch.unsqueeze(1))

            lamba = get_current_consistency_weight(iter_num, total_iter_num)
            # lamba = 0.5
            loss = 0.5 * (loss_dice + loss_ce) + lamba * supervised_loss2
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
            writer.add_scalar('info/loss_ds', supervised_loss2, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_ds: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), supervised_loss2.item()))


            # if iter_num % 20 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label',
            #                      grid_image, iter_num)
            #
            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(
            #         0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label',
            #                      grid_image, iter_num)


            # if epoch_num % 50 == 0 and i_batch == 1:  # 保存feature时的训练batchsize必须设为1
            #     volume_PET, label_batch = volume_PET.squeeze().detach().cpu().numpy(), label_batch.squeeze().detach().cpu().numpy()
            #     img_itk = sitk.GetImageFromArray(volume_PET)
            #     img_itk.SetSpacing((1.0, 1.0, 1.0))
            #     img_itk.SetOrigin((0, 0, 0))
            #     sitk.WriteImage(img_itk, featuresavepath + "{}_pet.nii.gz".format(epoch_num))
            #
            #     img_itk = sitk.GetImageFromArray(label_batch.astype(np.uint8))
            #     img_itk.SetSpacing((1.0, 1.0, 1.0))
            #     img_itk.SetOrigin((0, 0, 0))
            #     sitk.WriteImage(img_itk, featuresavepath + "{}_label.nii.gz".format(epoch_num))



        model.eval()

        avg_metric = test_all_case(model, args.root_valpath, val_num=args.val_num, num_classes=2,
                                   patch_size=args.patch_size, stride_xy=32, stride_z=64)

        if avg_metric[:, 0].mean() > best_performance:
            best_performance = avg_metric[:, 0].mean()
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_dice_{}.pth'.format(
                                              epoch_num, round(best_performance, 4)))
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)

        writer.add_scalar('info/val_dice_score',
                          avg_metric[0, 0], iter_num)
        writer.add_scalar('info/val_hd95',
                          avg_metric[0, 1], iter_num)
        logging.info(
            'epoch %d : iteration %d : dice_score : %f hd95 : %f' % (epoch_num, iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
        model.train()

        # if iter_num % 3000 == 0:
        #     save_mode_path = os.path.join(
        #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epochs:
            iterator.close()
            break
    writer.close()
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
