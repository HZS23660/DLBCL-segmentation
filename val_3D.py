import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import os


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)

    dd, hh, ww = image.shape
    sd = math.ceil((dd - patch_size[0]) / stride_z) + 1  # d
    sh = math.ceil((hh - patch_size[1]) / stride_xy) + 1  # h
    sw = math.ceil((ww - patch_size[2]) / stride_xy) + 1  # w
    # print("{}, {}, {}".format(sx, sy, sz))

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sd):
        xs = min(stride_z * x, dd - patch_size[0])
        for y in range(0, sh):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sw):
                zs = min(stride_xy * z, ww - patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map



def test_single_case_2chs(net, image_PET, image_CT, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image_PET.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image_PET = np.pad(image_PET, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image_CT = np.pad(image_PET, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)

    dd, hh, ww = image_PET.shape
    sd = math.ceil((dd - patch_size[0]) / stride_z) + 1  # d
    sh = math.ceil((hh - patch_size[1]) / stride_xy) + 1  # h
    sw = math.ceil((ww - patch_size[2]) / stride_xy) + 1  # w
    # print("{}, {}, {}".format(sx, sy, sz))

    score_map = np.zeros((num_classes,) + image_PET.shape).astype(np.float32)
    cnt = np.zeros(image_PET.shape).astype(np.float32)

    for x in range(0, sd):
        xs = min(stride_z * x, dd - patch_size[0])
        for y in range(0, sh):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sw):
                zs = min(stride_xy * z, ww - patch_size[2])
                test_patch1 = image_PET[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch2 = image_CT[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch1 = np.expand_dims(np.expand_dims(
                    test_patch1, axis=0), axis=0).astype(np.float32)
                test_patch2 = np.expand_dims(np.expand_dims(
                    test_patch2, axis=0), axis=0).astype(np.float32)
                test_patch1 = torch.from_numpy(test_patch1).cuda()
                test_patch2 = torch.from_numpy(test_patch2).cuda()
                test_patch = torch.cat((test_patch2, test_patch1), dim = 1)

                with torch.no_grad():
                    # y1, _ = net(test_patch)
                    y1 = net(test_patch)
                    # _, y1 = net(test_patch)
                    # _, y1 = net(test_patch, test_patch1)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map



def test_single_case_2chs_prior(net, image_PET, image_CT, prior_mask, stride_xy, stride_z, patch_size,
                                num_classes=1):
    w, h, d = image_PET.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image_PET = np.pad(image_PET, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image_CT = np.pad(image_CT, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)

    dd, hh, ww = image_PET.shape
    sd = math.ceil((dd - patch_size[0]) / stride_z) + 1  # d
    sh = math.ceil((hh - patch_size[1]) / stride_xy) + 1  # h
    sw = math.ceil((ww - patch_size[2]) / stride_xy) + 1  # w
    # print("{}, {}, {}".format(sx, sy, sz))

    score_map = np.zeros((num_classes,) + image_PET.shape).astype(np.float32)
    cnt = np.zeros(image_PET.shape).astype(np.float32)

    for x in range(0, sd):
        xs = min(stride_z * x, dd - patch_size[0])
        for y in range(0, sh):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sw):
                zs = min(stride_xy * z, ww - patch_size[2])
                test_patch1 = image_PET[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch2 = image_CT[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch1 = np.expand_dims(np.expand_dims(
                    test_patch1, axis=0), axis=0).astype(np.float32)
                test_patch2 = np.expand_dims(np.expand_dims(
                    test_patch2, axis=0), axis=0).astype(np.float32)
                test_patch1 = torch.from_numpy(test_patch1).cuda()
                test_patch2 = torch.from_numpy(test_patch2).cuda()
                test_patch = torch.cat((test_patch2, test_patch1), dim = 1)
                prior_mask1 = prior_mask[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                prior_mask1 = np.expand_dims(prior_mask1, axis=0).astype(np.uint8)
                new_mask = torch.from_numpy(prior_mask1).cuda()

                with torch.no_grad():
                    y1, _, _ = net(test_patch, test_patch, new_mask)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map



def test_single_case_2chs_prior_N(net, image_PET, image_CT, prior_CT, prior_PET, prior_mask, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image_PET.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image_PET = np.pad(image_PET, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image_CT = np.pad(image_CT, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)

    dd, hh, ww = image_PET.shape
    sd = math.ceil((dd - patch_size[0]) / stride_z) + 1  # d
    sh = math.ceil((hh - patch_size[1]) / stride_xy) + 1  # h
    sw = math.ceil((ww - patch_size[2]) / stride_xy) + 1  # w
    # print("{}, {}, {}".format(sx, sy, sz))

    score_map = np.zeros((num_classes,) + image_PET.shape).astype(np.float32)
    cnt = np.zeros(image_PET.shape).astype(np.float32)

    prior_CT = torch.from_numpy(np.expand_dims(np.expand_dims(prior_CT, axis=0), axis=0).astype(np.float32))
    prior_PET = torch.from_numpy(np.expand_dims(np.expand_dims(prior_PET, axis=0), axis=0).astype(np.float32))
    prior_data_batch = torch.cat((prior_CT, prior_PET), dim = 1)
    prior_data_batch = prior_data_batch.cuda()
    # prior_data_batch = F.interpolate(prior_data_batch, size=(patch_size[0], patch_size[1], \
    #                                  patch_size[2]), mode='trilinear', align_corners=True)


    prior_mask1 = torch.from_numpy(np.expand_dims(prior_mask, axis=0).astype(np.uint8))
    # prior_mask1 = F.interpolate(prior_mask1.unsqueeze(1).float(), size=(patch_size[0], patch_size[1], \
    #                                       patch_size[2]), mode='trilinear', align_corners=True)
    new_mask = prior_mask1.squeeze(1).cuda()

    for x in range(0, sd):
        xs = min(stride_z * x, dd - patch_size[0])
        for y in range(0, sh):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sw):
                zs = min(stride_xy * z, ww - patch_size[2])
                test_patch1 = image_PET[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch2 = image_CT[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch1 = np.expand_dims(np.expand_dims(
                    test_patch1, axis=0), axis=0).astype(np.float32)
                test_patch2 = np.expand_dims(np.expand_dims(
                    test_patch2, axis=0), axis=0).astype(np.float32)
                test_patch1 = torch.from_numpy(test_patch1).cuda()
                test_patch2 = torch.from_numpy(test_patch2).cuda()
                test_patch = torch.cat((test_patch2, test_patch1), dim = 1)

                with torch.no_grad():
                    # y1, _, _ = net(test_patch, prior_data_batch, new_mask)
                    # _, y1, _, _ = net(test_patch, prior_data_batch)
                    y1, _, _ = net(test_patch, prior_data_batch)
                    # y1, _, _, _, _ = net(test_patch, prior_data_batch)
                    # y1, _, _, _ = net(test_patch, prior_data_batch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        iou = metric.binary.jc(pred, gt)
        sensitivity = metric.binary.sensitivity(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, iou, sensitivity, hd95])
    else:
        return np.zeros(4)



def test_all_case(net, path, val_num = None, num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    image_list = []
    for name in os.listdir(path):
        image_list.append(os.path.join(path, name))
    if val_num is not None:
        image_list = image_list[:val_num]
    total_metric = np.zeros((num_classes-1, 4))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image_PET = h5f['image_PET'][:]
        image_CT = h5f['image_CT'][:]
        label = h5f['label'][:]
        prediction = test_single_case_2chs(net, image_PET, image_CT, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)



def test_all_case_prior(net, path, val_num = None, num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    image_list = []
    for name in os.listdir(path):
        image_list.append(os.path.join(path, name))
    if val_num is not None:
        image_list = image_list[:val_num]
    total_metric = np.zeros((num_classes-1, 4))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image_PET = h5f['image_PET'][:]
        image_CT = h5f['image_CT'][:]
        label = h5f['label'][:]
        prior_mask = h5f['prior_tumor_label'][:]
        prediction = test_single_case_2chs_prior(net, image_PET, image_CT, prior_mask, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)



def test_all_case_prior_N(net, path, val_num = None, num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    image_list = []
    for name in os.listdir(path):
        image_list.append(os.path.join(path, name))
    if val_num is not None:
        image_list = image_list[:val_num]
    total_metric = np.zeros((num_classes-1, 4))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image_PET = h5f['image_PET'][:]
        image_CT = h5f['image_CT'][:]
        label = h5f['label'][:]
        prior_label = h5f['weak_label'][:]
        prior_CT, prior_PET, prior_label = body_bbox(image_CT, image_PET, prior_label, output_size=patch_size)
        prediction = test_single_case_2chs_prior_N(net, image_PET, image_CT, prior_CT, prior_PET, prior_label, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)



def body_bbox(data_CT, data_PET, label, output_size = [96, 96, 96]):

    mask = (label != 0)
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    # data_CT_bboxed = data_CT[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    # data_PET_bboxed = data_PET[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    # data_label_bboxed = label[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    assert output_size[0] >= (maxZidx-minZidx)+2 and output_size[1] >= (maxXidx-minXidx)+2 \
           and output_size[2] >= (maxYidx-minYidx)+2, print("输出的patch size小于先验大小")

    wmin = max(maxZidx-output_size[0]+2, 0)
    hmin = max(maxXidx-output_size[1]+2, 0)
    dmin = max(maxYidx-output_size[2]+2, 0)

    w1 = np.random.randint(wmin, minZidx-2)
    h1 = np.random.randint(hmin, minXidx-2)
    d1 = np.random.randint(dmin, minYidx-2)

    CT_bboxed = data_CT[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    PET_bboxed = data_PET[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    label_bboxed = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

    assert sum(sum(sum(label_bboxed))) == sum(sum(sum(label))),\
        print("先验裁剪的patch size有误,label_bboxed={},label={}".format(sum(sum(sum(label_bboxed))), sum(sum(sum(label)))))

    pw = max((output_size[0] - label_bboxed.shape[0]) // 2 + 3, 0)
    ph = max((output_size[1] - label_bboxed.shape[1]) // 2 + 3, 0)
    pd = max((output_size[2] - label_bboxed.shape[2]) // 2 + 3, 0)
    data_CT_bboxed = np.pad(CT_bboxed, [(pw, pw), (ph, ph), (pd, pd)],
                    mode='constant', constant_values=0)
    data_PET_bboxed = np.pad(PET_bboxed, [(pw, pw), (ph, ph), (pd, pd)],
                    mode='constant', constant_values=0)
    data_label_bboxed = np.pad(label_bboxed, [(pw, pw), (ph, ph), (pd, pd)],
                    mode='constant', constant_values=0)

    (w, h, d) = data_CT_bboxed.shape
    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
    d1 = int(round((d - output_size[2]) / 2.))

    CT_bboxed = data_CT_bboxed[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    PET_bboxed = data_PET_bboxed[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    label_bboxed = data_label_bboxed[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]


    return CT_bboxed, PET_bboxed, label_bboxed