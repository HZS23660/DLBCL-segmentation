from networks.vnet import VNet, VNet_cat, VNet_att, VNet_att_new


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "VNet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_cat":
        net = VNet_cat(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_att":
        net = VNet_att(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_att_new":
        net = VNet_att_new(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    else:
        net = None
    return net
