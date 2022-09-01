import argparse
import os
import shutil
from glob import glob
import numpy as np
import math
import SimpleITK as sitk
import h5py
from tqdm import tqdm


import torch
import torch.nn.functional as F
from networks.net_factory_3d import net_factory_3d


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='C:\Dataset\PET_lymphome\PET_lymphome_TestData_t_prior', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='PET_lymphome_3D/FS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet_res', help='model_name')
parser.add_argument('--labeled_num', type=int, default=120,
                    help='labeled data')
parser.add_argument('--test_num', type=int, default=14,
                    help='test data')
parser.add_argument('--patch_size', type=list,  default=[256, 128, 128],  # D H W
                    help='patch size of network input')




def Inference(FLAGS):
    snapshot_path = "./model/{}_{}_labeled_{} {} {}/{}".format(FLAGS.exp, FLAGS.labeled_num, \
                                                               FLAGS.patch_size[0], FLAGS.patch_size[1],
                                                               FLAGS.patch_size[2], FLAGS.model)
    num_classes = 2
    test_save_path = "./test/{}_{}_labeled_{} {} {}/{}_{}_predictions/".format(FLAGS.exp, \
                                                FLAGS.labeled_num, FLAGS.patch_size[0], FLAGS.patch_size[1],
                                                FLAGS.patch_size[2],FLAGS.model, FLAGS.test_num)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=2, class_num=num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case_feature(net, FLAGS.root_path, num = FLAGS.test_num, method=FLAGS.model,test_save_path=test_save_path)
    return avg_metric



def test_all_case_feature(net, path, method="unet_3D", num=None, test_save_path=None):
    image_list = []
    for name in os.listdir(path):
        image_list.append(os.path.join(path, name))
    if num is not None:
        image_list = image_list[:num]
        image_list1 = image_list[1:num]
        image_list1.append(image_list[0])
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path, image_path1 in tqdm(zip(image_list, image_list1)):
            ids = image_path.split(os.sep)[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image_PET = h5f['image_PET'][:]
            image_CT = h5f['image_CT'][:]
            label = h5f['label'][:]
            prior_label = h5f['prior_tumor_label'][:]
            image_CT = torch.from_numpy(np.expand_dims(np.expand_dims(image_CT, axis=0), axis=0).astype(np.float32)).cuda()
            image_PET = torch.from_numpy(np.expand_dims(np.expand_dims(image_PET, axis=0), axis=0).astype(np.float32)).cuda()
            prior_label_new = torch.from_numpy(np.expand_dims(prior_label, axis=0).astype(np.uint8)).cuda()
            label_new = torch.from_numpy(np.expand_dims(label, axis=0).astype(np.uint8)).cuda()
            test_patch = torch.cat((image_CT, image_PET), dim=1)

            h5f = h5py.File(image_path1, 'r')
            image_PET1 = h5f['image_PET'][:]
            image_CT1 = h5f['image_CT'][:]
            label1 = h5f['label'][:]
            prior_label1 = h5f['prior_tumor_label'][:]
            image_CT1 = torch.from_numpy(np.expand_dims(np.expand_dims(image_CT1, axis=0), axis=0).astype(np.float32)).cuda()
            image_PET1 = torch.from_numpy(np.expand_dims(np.expand_dims(image_PET1, axis=0), axis=0).astype(np.float32)).cuda()
            prior_label_new1 = torch.from_numpy(np.expand_dims(prior_label1, axis=0).astype(np.uint8)).cuda()
            label_new1 = torch.from_numpy(np.expand_dims(label1, axis=0).astype(np.uint8)).cuda()
            test_patch1 = torch.cat((image_CT1, image_PET1), dim=1)

            feature = feature_extracter(net, test_patch, test_patch1, label_new1)

            img_itk = sitk.GetImageFromArray(prior_label)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            img_itk.SetOrigin((0, 0, 0))
            sitk.WriteImage(img_itk, test_save_path + "{}_prior_label.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(label)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            img_itk.SetOrigin((0, 0, 0))
            sitk.WriteImage(img_itk, test_save_path + "{}_label.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(feature)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            img_itk.SetOrigin((0, 0, 0))
            sitk.WriteImage(img_itk, test_save_path + "{}_feature.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image_PET.squeeze().cpu().numpy())
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            img_itk.SetOrigin((0, 0, 0))
            sitk.WriteImage(img_itk, test_save_path + "{}_PET.nii.gz".format(ids))

    f.close()
    print("Testing end")



def feature_extracter(net, input_data1, input_data2, prior_label):

    with torch.no_grad():
        supp_feat_0 = net.block_one(input_data1)
        supp_feat_1 = net.block_one_dw(supp_feat_0)
        supp_feat_2 = net.block_two(supp_feat_1)
        supp_feat_2 = net.block_two_dw(supp_feat_2)
        supp_feat_3 = net.block_three(supp_feat_2)
        supp_feat_3 = net.block_three_dw(supp_feat_3)
        supp_feat_4 = net.block_four(supp_feat_3)
        supp_feat_4 = net.block_four_dw(supp_feat_4)
        x1 = net.block_five(supp_feat_4)

    with torch.no_grad():
        supp_feat_0 = net.block_one(input_data2)
        supp_feat_1 = net.block_one_dw(supp_feat_0)
        supp_feat_2 = net.block_two(supp_feat_1)
        supp_feat_2 = net.block_two_dw(supp_feat_2)
        supp_feat_3 = net.block_three(supp_feat_2)
        supp_feat_3 = net.block_three_dw(supp_feat_3)
        supp_feat_4 = net.block_four(supp_feat_3)
        supp_feat_4 = net.block_four_dw(supp_feat_4)
        x2 = net.block_five(supp_feat_4)

        input_mask = train_free_prior_mask(x1, x2, prior_label.unsqueeze(1))
        input_mask = F.interpolate(input_mask.type(torch.float32), size=(input_data1.size(2), input_data1.size(3),\
                                            input_data1.size(4)), mode='trilinear', align_corners=True)
        input_mask = input_mask.squeeze().cpu().numpy()

    return input_mask



def train_free_prior_mask(query_feat, supp_feat, mask_supp):


    cosine_eps = 1e-7

    tmp_supp_feat = supp_feat
    tmp_mask_supp = mask_supp

    resize_size1, resize_size2, resize_size3 = tmp_supp_feat.size(2), tmp_supp_feat.size(3), tmp_supp_feat.size(4)
    tmp_mask = F.interpolate(tmp_mask_supp.type(torch.float32), size=(resize_size1, resize_size2, resize_size3), \
                             mode='trilinear', align_corners=True)

    tmp_supp_feat = tmp_supp_feat * tmp_mask
    q = query_feat
    s = tmp_supp_feat
    bsize, ch_sz, sp_sd, sp_sh, sp_sw = q.size()[:]

    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sd * sp_sh * sp_sw)
    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
    corr_query_mask = similarity.view(bsize, 1, sp_sd, sp_sh, sp_sw)


    return corr_query_mask


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print('dice, ravd, hd, asd:')
    print(metric)
