import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler



# class Lymphoma_3D_prior(Dataset):
#     """ PET_lymphoma Dataset """
#     # 共180个patients
#
#     def __init__(self, base_dir=None, split='train', num = None, transform=None, patch_size = [64, 64, 64]):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.image_list = []
#         self.patch_size = patch_size
#
#         all_files = sorted(os.listdir(self._base_dir))
#
#         if split == 'train':
#             for name in all_files:
#                 if os.path.isdir(os.path.join(self._base_dir, name)):
#                     continue
#                 self.image_list.append(name.split(".")[-2])
#         elif split == 'test':
#             for name in all_files:
#                 self.image_list.append(name.split(".")[-2])
#
#         if num is not None:
#             self.num = num
#             self.image_list = self.image_list[:self.num]
#
#         print("total {} samples".format(len(self.image_list)))
#
#
#     def __len__(self):
#         return len(self.image_list)
#
#
#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         h5f = h5py.File(glob(self._base_dir + "/{}.h5".format(image_name))[0], 'r')
#         image_PET = h5f['image_PET'][:].astype(np.float32)
#         image_CT = h5f['image_CT'][:].astype(np.float32)
#         label = h5f['label'][:].astype(np.uint8)
#         prior_label = h5f['prior_tumor_label'][:].astype(np.uint8)
#         prior_CT, prior_PET, prior_label = body_bbox(image_CT, image_PET, prior_label, output_size=self.patch_size)
#         sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label,\
#                    'prior_CT': prior_CT, 'prior_PET': prior_PET, 'prior_label': prior_label}
#         # sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label, \
#         #           'prior_label': prior_label}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample



class Lymphoma_3D_prior(Dataset):
    """ PET_lymphoma Dataset """
    # 共180个patients

    def __init__(self, base_dir=None, prior_dir=None, split='train', num=None, transform=None, patch_size=[64, 64, 64]):
        self._base_dir = base_dir
        self.prior_dir = prior_dir
        self.transform = transform
        self.image_list = []
        self.patch_size = patch_size

        all_files = sorted(os.listdir(self._base_dir))

        if split == 'train':
            for name in all_files:
                if os.path.isdir(os.path.join(self._base_dir, name)):
                    continue
                self.image_list.append(name.split(".")[-2])

        if num is not None:
            self.num = num
            self.image_list = self.image_list[:self.num]

        # print(self.image_list)
        print("total {} samples".format(len(self.image_list)))


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(glob(self._base_dir + "/{}.h5".format(image_name))[0], 'r')
        image_PET = h5f['image_PET'][:].astype(np.float32)
        image_CT = h5f['image_CT'][:].astype(np.float32)
        label = h5f['label'][:].astype(np.uint8)

        prior_name = image_name.split('_')[0]
        h5f1 = h5py.File(glob(self.prior_dir + "/{}_prior.h5".format(prior_name))[0], 'r')
        prior_PET = h5f1['prior_PET'][:].astype(np.float32)
        prior_CT = h5f1['prior_CT'][:].astype(np.float32)
        prior_label = h5f1['prior_label'][:].astype(np.uint8)

        sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label, \
                   'prior_CT': prior_CT, 'prior_PET': prior_PET, 'prior_label': prior_label}
        # sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label, \
        #           'prior_label': prior_label}
        if self.transform:
            sample = self.transform(sample)
        return sample



class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # image1, image2, image3, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], sample['label']
        image1, image2, image3, image4, image5, label = sample['image_PET'], sample['image_CT'], sample['prior_label'],\
                                                        sample['prior_CT'], sample['prior_PET'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image1 = np.pad(image1, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            image2 = np.pad(image2, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image1.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image1 = image1[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image2 = image2[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        # return {'image_PET': image1, 'image_CT': image2, 'label': label,\
        #           'prior_label': image3}
        return {'image_PET': image1, 'image_CT': image2, 'label': label,\
                  'prior_label': image3, 'prior_CT': image4, 'prior_PET': image5}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        # image1, image2, image3, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], sample['label']
        image1, image2, image3, image4, image5, label = sample['image_PET'], sample['image_CT'], sample['prior_label'],\
                                                        sample['prior_CT'], sample['prior_PET'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image1 = np.pad(image1, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            image2 = np.pad(image2, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image1.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        # stride_z = 64
        # stride_xy = 32
        # w1 = np.random.randint(0, (w - self.output_size[0]) // stride_z + 1) * stride_z
        # h1 = np.random.randint(0, (h - self.output_size[1]) // stride_xy + 1) * stride_xy
        # d1 = np.random.randint(0, (d - self.output_size[2]) // stride_xy + 1) * stride_xy
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image1 = image1[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image2 = image2[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image_PET': image1, 'image_CT': image2, 'label': label,\
                  'prior_label': image3, 'sdf': sdf}
        else:
            # return {'image_PET': image1, 'image_CT': image2, 'label': label,\
            #       'prior_label': image3}
            return {'image_PET': image1, 'image_CT': image2, 'label': label, \
                    'prior_label': image3, 'prior_CT': image4, 'prior_PET': image5}



class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        # image1, image2, image3, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], sample['label']
        image1, image2, image3, image4, image5, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], \
                                                        sample['prior_CT'], sample['prior_PET'], sample['label']

        k = np.random.randint(0, 4)
        image1 = np.rot90(image1, k)
        image2 = np.rot90(image2, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image1 = np.flip(image1, axis=axis).copy()
        image2 = np.flip(image2, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        # return {'image_PET': image1, 'image_CT': image2, 'label': label,\
        #           'prior_label': image3}
        return {'image_PET': image1, 'image_CT': image2, 'label': label,\
                  'prior_label': image3, 'prior_CT': image4, 'prior_PET': image5}



class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image1, image2, image3, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image1.shape[0], image1.shape[1], image1.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image1 = image1 + noise
        image2 =image2 + noise
        return {'image_PET': image1, 'image_CT': image2, 'label': label,\
                  'prior_label': image3}



class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image1, image2, label = sample['image_PET'], sample['image_CT'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image_PET': image1, 'image_CT': image2, 'label': label, 'onehot_label': onehot_label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image1, image2, image3, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], sample['label']
        image1, image2, image3, image4, image5, label = sample['image_PET'], sample['image_CT'], sample['prior_label'], \
                                                        sample['prior_CT'], sample['prior_PET'], sample['label']
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2]).astype(np.float32)
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1], image2.shape[2]).astype(np.float32)
        image4 = image4.reshape(1, image4.shape[0], image4.shape[1], image4.shape[2]).astype(np.float32)
        image5 = image5.reshape(1, image5.shape[0], image5.shape[1], image5.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image_PET': torch.from_numpy(image1), 'image_CT': torch.from_numpy(image2),
                    'label': torch.from_numpy(label).long(),
                    'prior_label': torch.from_numpy(image3).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            # return {'image_PET': torch.from_numpy(image1), 'image_CT': torch.from_numpy(image2),
            #         'label': torch.from_numpy(label).long(),
            #         'prior_label': torch.from_numpy(image3).long()}
            return {'image_PET': torch.from_numpy(image1), 'image_CT': torch.from_numpy(image2), 'label': torch.from_numpy(label).long(),\
                        'prior_label': torch.from_numpy(image3).long(), 'prior_CT': torch.from_numpy(image4), 'prior_PET': torch.from_numpy(image5)}



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



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