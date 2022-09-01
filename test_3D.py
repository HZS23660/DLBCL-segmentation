import argparse
import os
import shutil
from glob import glob

import torch

from networks.net_factory_3d import net_factory_3d
from test_3D_util import test_all_case, test_all_case_prior, test_all_case_prior_N

# PET_lymphoma_TestData_prior
basicdir = '/home/tju/Documents/HZS/PET_lymphoma/PET_lymphoma_PW_new/'

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default=basicdir + 'PET_lymphoma_TestData_prior',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='PET_lymphoma_3D_PW_new/WSN_l0.5_a0.2',
                    help='experiment_name')
parser.add_argument('--model', type=str,
                    default='VNet_att',
                    help='model_name')
parser.add_argument('--labeled_num', type=int,
                    default=60,
                    help='labeled data')
parser.add_argument('--test_num', type=int,
                    default=27,
                    help='test data')
parser.add_argument('--patch_size', type=list,
                    default=[128, 64, 64],  # D H W
                    help='patch size of network input')




def Inference(FLAGS):
    # net1 = 'epoch_130_dice_0.8277.pth'
    # net1 = 'epoch_140_dice_0.8305.pth'
    # net1 = 'epoch_144_dice_0.8364.pth'
    net1 = FLAGS.model + '_best_model.pth'
    # net_dir = net1.split(".pth")[0]


    snapshot_path = "./model/{}_{}_labeled_{} {} {}/{}".format(FLAGS.exp, FLAGS.labeled_num,\
                                                               FLAGS.patch_size[0], FLAGS.patch_size[1],
                                                               FLAGS.patch_size[2], FLAGS.model)
    num_classes = 2
    # test_save_path = "../test_lymphoma/{}_{}_{} {} {}/{}_predictions/".format(FLAGS.exp, FLAGS.labeled_num, \
    #                         FLAGS.patch_size[0], FLAGS.patch_size[1], FLAGS.patch_size[2], net_dir)
    test_save_path = "../test_lymphoma/{}_{}_{} {} {}3/{}_predictions/".format(FLAGS.exp, FLAGS.labeled_num, \
                            FLAGS.patch_size[0], FLAGS.patch_size[1], FLAGS.patch_size[2], FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=2, class_num=num_classes)
    save_mode_path = os.path.join(snapshot_path, net1)
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case_prior_N(net, FLAGS.root_path, num=FLAGS.test_num, method=FLAGS.model, num_classes=num_classes,
                               patch_size=FLAGS.patch_size, stride_xy=32, stride_z=64, test_save_path=test_save_path)
    # avg_metric = test_all_case(net, FLAGS.root_path, num=FLAGS.test_num, method=FLAGS.model, num_classes=num_classes,
    #                                    patch_size=FLAGS.patch_size, stride_xy=32, stride_z=64, test_save_path=test_save_path)
    return avg_metric



if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print('dice, sensitivity, specificity, iou:')
    print(metric)
