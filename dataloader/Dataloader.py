import numpy as np
import torch



def load_data_kfold(args, dataset, k, n, worker_init_fn, batch_size):
    # This function using functions in preprocessing.py to build dataset,
    # and then randomly split dataset with a fixed random seed.
    print("Splitting DataSet ...")

    l = len(dataset)
    #print(l)
    shuffle_dataset = False
    random_seed = args.seed  # fixed random seed
    indices = list(range(l))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)  # shuffle
    # Collect indexes of samples for validation set.
    val_indices = indices[int(l / k) * n:int(l / k) * (n + 1)]
    # Collect indexes of samples for train set. Here the logic is that a sample
    # cannot in train set if already in validation set
    train_indices = list(set(indices).difference(set(val_indices)))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)  # build Sampler
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, \
                                               num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=valid_sampler,\
                                                    num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    print("Complete")
    return train_loader, validation_loader



def load_data(args, dataset, batch_size, worker_init_fn):

    print("Splitting DataSet ...")
    validation_split = 0.1
    l = len(dataset)
    #print(l)
    shuffle_dataset = False
    random_seed = args.seed
    indices = list(range(l))
    split = int(np.floor(validation_split * l))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8,
                            pin_memory=True, worker_init_fn=worker_init_fn)  # build dataloader for train set
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=valid_sampler,
                        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)  # build dataloader for validate set
    print("Complete")

    return train_loader, validation_loader