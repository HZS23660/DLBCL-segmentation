import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler


# class CT_PET_lymphoma_3D(Dataset):
#     """ PET_lymphoma Dataset """
#     # 共180个patients
#
#     def __init__(self, base_dir=None, split='train', num = None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.image_list = []
#         if num is not None:
#             self.num = num
#             all_files = sorted(os.listdir(self._base_dir))[:self.num]
#         else:
#             all_files = sorted(os.listdir(self._base_dir))
#
#
#         if split == 'train':
#             for name in all_files:
#                 self.image_list.append(name.split(".")[-2])
#         elif split == 'test':
#             for name in all_files:
#                 self.image_list.append(name.split(".")[-2])
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
#         sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label.astype(np.uint8)}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample


class CT_PET_lymphoma_3D(Dataset):
    """ PET_lymphoma Dataset """
    # 共180个patients

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = []

        all_files = sorted(os.listdir(self._base_dir))

        if split == 'train':
            for name in all_files:
                if os.path.isdir(os.path.join(self._base_dir, name)):
                    continue
                self.image_list.append(name.split(".")[-2])

        if num is not None:
            self.num = num
            self.image_list = self.image_list[:self.num]

        print("total {} samples".format(len(self.image_list)))


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(glob(self._base_dir + "/{}.h5".format(image_name))[0], 'r')
        image_PET = h5f['image_PET'][:].astype(np.float32)
        image_CT = h5f['image_CT'][:].astype(np.float32)
        label = h5f['label'][:].astype(np.uint8)
        sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample




class Lymphoma_3D_prior(Dataset):
    """ PET_lymphoma Dataset """
    # 共180个patients

    def __init__(self, base_dir=None, split='train', num=None, transform=None, patch_size = [64, 64, 64]):
        self._base_dir = base_dir
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

        sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)

        return sample




class Lymphoma_3D_weak(Dataset):
    """ PET_lymphoma Dataset """
    # 共180个patients

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = []

        all_files = sorted(os.listdir(self._base_dir))

        if split == 'train':
            for name in all_files:
                if os.path.isdir(os.path.join(self._base_dir, name)):
                    continue
                self.image_list.append(name.split(".")[-2])

        if num is not None:
            self.num = num
            self.image_list = self.image_list[:self.num]

        print("total {} samples".format(len(self.image_list)))


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(glob(self._base_dir + "/{}.h5".format(image_name))[0], 'r')
        image_PET = h5f['image_PET'][:].astype(np.float32)
        image_CT = h5f['image_CT'][:].astype(np.float32)
        weak_label = h5f['weak_label'][:].astype(np.uint8)
        label = h5f['label'][:].astype(np.uint8)
        sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label, \
            'weak_label': weak_label}
        if self.transform:
            sample = self.transform(sample)
        return sample



class Lymphoma_3D_SS(Dataset):
    """ PET_lymphoma Dataset """
    # 共180个patients

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = []

        all_files = sorted(os.listdir(self._base_dir))

        if split == 'train':
            for name in all_files:
                if os.path.isdir(os.path.join(self._base_dir, name)):
                    continue
                self.image_list.append(name.split(".")[-2])

        if num is not None:
            self.num = num
            self.image_list = self.image_list[:self.num]

        print("total {} samples".format(len(self.image_list)))


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(glob(self._base_dir + "/{}.h5".format(image_name))[0], 'r')
        image_PET = h5f['image_PET'][:].astype(np.float32)
        image_CT = h5f['image_CT'][:].astype(np.float32)
        label = h5f['label'][:].astype(np.uint8)
        sample = {'image_PET': image_PET, 'image_CT': image_CT, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample



class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image1, image2, label = sample['image_PET'], sample['image_CT'], sample['label']
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

        return {'image_PET': image1, 'image_CT': image2, 'label': label}



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
        image1, image2, label = sample['image_PET'], sample['image_CT'],  \
                                            sample['label']
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
        # stride_z = 32
        # stride_xy = 16
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        # print(w1, h1, d1)
        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image1 = image1[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image2 = image2[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image_PET': image1, 'image_CT': image2, 'label': label, 'sdf': sdf}
        else:
            return {'image_PET': image1, 'image_CT': image2, 'label': label}



class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image1, image2, label = sample['image_PET'], sample['image_CT'], \
                                            sample['label']
        k = np.random.randint(0, 4)
        image1 = np.rot90(image1, k)
        image2 = np.rot90(image2, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image1 = np.flip(image1, axis=axis).copy()
        image2 = np.flip(image2, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image_PET': image1, 'image_CT': image2, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image1, image2, label = sample['image_PET'], sample['image_CT'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image1.shape[0], image1.shape[1], image1.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image1 = image1 + noise
        image2 =image2 + noise
        return {'image_PET': image1, 'image_CT': image2, 'label': label}



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
        image1, image2 = sample['image_PET'], sample['image_CT']
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2]).astype(np.float32)
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1], image2.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image_PET': torch.from_numpy(image1), 'image_CT': torch.from_numpy(image2),
                    'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image_PET': torch.from_numpy(image1), 'image_CT': torch.from_numpy(image2),
                    'label': torch.from_numpy(sample['label']).long()}


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



if __name__ == "__main__":
    for i in range(0, 100):
        a = np.random.randint(0, 64//16 + 1) * 16
        print('a =', a)