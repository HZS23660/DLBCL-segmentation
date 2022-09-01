import torch
from torch import nn
import torch.nn.functional as F


#---------------------------------------------------

class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''

    def __init__(self, inChans=256, outChans=256, dense_features=(4, 2, 2), stride=2, kernel_size=3, padding=1, #(10, 12, 8)
                 activation='LeakyReLU', normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()

        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=inChans)
        if activation == "LeakyRelu":
            self.actv1 = nn.LeakyReLU(negative_slope=0.15, inplace=True)
            self.actv2 = nn.LeakyReLU(negative_slope=0.15, inplace=True)
        else:
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)

        zip_channels = 32

        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=zip_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.dense1 = nn.Linear(in_features=zip_channels * dense_features[0] * dense_features[1] * dense_features[2],
                                out_features=inChans)
        self.dense2 = nn.Linear(in_features=midChans,
                                out_features=midChans * dense_features[0] * dense_features[1] * dense_features[2])
        self.up0 = LinearUpSampling(midChans, outChans)

    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = out.view(-1, self.num_flat_features(out))
        out_vd = self.dense1(out)
        # print(out_vd.size())
        distr = out_vd
        out_vd = VDraw(out_vd)
        out = self.dense2(out_vd)
        out = self.actv2(out)
        out = out.view((-1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)

        return out, distr

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


def VDraw(x):
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.Normal(x[:, :128], x[:, 128:]).sample()




class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''

    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)

    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode,
                                        align_corners=self.align_corners)

        if skipx is not None:
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)

        return out

#-------------------------------------------------------------------------------------------



class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', relu='LeakyReLU'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if relu == 'ReLU':
                ops.append(nn.ReLU(inplace=True))
            elif relu == 'LeakyReLU':
                ops.append(nn.LeakyReLU(negative_slope=0.15, inplace=True))
            # ops.append(nn.PReLU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', relu='LeakyReLU'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                # ops.append(nn.ReLU(inplace=True))
                ops.append(nn.LeakyReLU(negative_slope=0.15, inplace=True))
                # ops.append(nn.PReLU())

        self.conv = nn.Sequential(*ops)
        if relu == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif relu == 'LeakyReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.15, inplace=True)
        # self.relu = nn.PReLU()

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', relu='LeakyReLU'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        if relu == 'ReLU':
            ops.append(nn.ReLU(inplace=True))
        elif relu == 'LeakyReLU':
            ops.append(nn.LeakyReLU(negative_slope=0.15, inplace=True))
        # ops.append(nn.PReLU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', relu='LeakyReLU'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        if relu == 'ReLU':
            ops.append(nn.ReLU(inplace=True))
        elif relu == 'LeakyReLU':
            ops.append(nn.LeakyReLU(negative_slope=0.15, inplace=True))
        # ops.append(nn.PReLU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', relu='LeakyReLU'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False

        if relu == 'ReLU':
            ops.append(nn.ReLU(inplace=True))
        elif relu == 'LeakyReLU':
            ops.append(nn.LeakyReLU(negative_slope=0.15, inplace=True))
        # ops.append(nn.PReLU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


#-----------------------------------------------------------------


class AttentionBlock3D(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock3D, self).__init__()
        self.Wx = nn.Sequential(nn.Conv3d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm3d(int_channels))
        self.Wg = nn.Sequential(nn.Conv3d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm3d(int_channels))
        self.psi = nn.Sequential(nn.Conv3d(int_channels, int_channels, kernel_size=1),
                                 nn.Conv3d(int_channels, 1, kernel_size=1),
                                 nn.BatchNorm3d(1),
                                 nn.Sigmoid())
        # self.pooling = nn.MaxPool3d(kernel_size=2**(n_stage-1), stride=2**(n_stage-1))

    def forward(self, x, g):
        # apply the Wx to the skip connection
        # g1 = self.Wg(self.pooling(g))
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = self.Wg(F.interpolate(g, size=(x1.shape[2:]), mode='trilinear', align_corners=True))
        out = self.psi(nn.ReLU()(x1 + g1))
        # out = self.psi(x1 + g1)
        return out * x


# class AttentionBlock3D(nn.Module):
#     def __init__(self, int_channels_x, in_channels_g, int_channels):
#         super(AttentionBlock3D, self).__init__()
#         self.psi = nn.Sequential(nn.Conv3d(in_channels_g, int_channels_x, kernel_size=1),
#                                  nn.Conv3d(int_channels, int_channels_x, kernel_size=1))
#         # self.pooling = nn.MaxPool3d(kernel_size=2**(n_stage-1), stride=2**(n_stage-1))
#
#     def forward(self, x, g):
#         # after applying Wg to the input, upsample to the size of the skip connection
#         out = self.psi(F.interpolate(g, size=(x.shape[2:]), mode='trilinear', align_corners=True))
#         return out + x


class basic_ASPP(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(basic_ASPP, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.15, inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),  # (1,1,1)means ouput_dim
                                             nn.Conv3d(in_channel, depth, kernel_size=1, stride=1, bias=False),
                                             # nn.BatchNorm3d(depth),
                                             nn.LeakyReLU(negative_slope=0.15, inplace=True))
        self.atrous_block1 = basic_ASPP(in_channel, depth)
        self.atrous_block6 = basic_ASPP(in_channel, depth, rate=6)
        self.atrous_block12 = basic_ASPP(in_channel, depth, rate=12)
        self.atrous_block18 = basic_ASPP(in_channel, depth, rate=18)
        self.conv_1x1_output = nn.Conv3d(depth * 5, depth, kernel_size=1, stride=1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.global_avg_pool(x)
        image_features = F.interpolate(image_features, size=size, mode='trilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))

        return net


#----------------此V-net没有残差结构------------------
# class VNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
#         super(VNet, self).__init__()
#         self.has_dropout = has_dropout
#
#         self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
#         self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
#
#         self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
#
#         self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
#
#         self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
#
#         self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
#         self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
#
#         self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
#
#         self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
#
#         self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
#
#         self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
#         self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
#
#         self.dropout = nn.Dropout3d(p=0.5, inplace=False)
#         # self.__init_weight()
#
#     def encoder(self, input):
#         x1 = self.block_one(input)
#         x1_dw = self.block_one_dw(x1)
#
#         x2 = self.block_two(x1_dw)
#         x2_dw = self.block_two_dw(x2)
#
#         x3 = self.block_three(x2_dw)
#         x3_dw = self.block_three_dw(x3)
#
#         x4 = self.block_four(x3_dw)
#         x4_dw = self.block_four_dw(x4)
#
#         x5 = self.block_five(x4_dw)
#         # x5 = F.dropout3d(x5, p=0.5, training=True)
#         if self.has_dropout:
#             x5 = self.dropout(x5)
#
#         res = [x1, x2, x3, x4, x5]
#
#         return res
#
#     def decoder(self, features):
#         x1 = features[0]
#         x2 = features[1]
#         x3 = features[2]
#         x4 = features[3]
#         x5 = features[4]
#
#         x5_up = self.block_five_up(x5)
#         x5_up = x5_up + x4
#
#         x6 = self.block_six(x5_up)
#         x6_up = self.block_six_up(x6)
#         x6_up = x6_up + x3
#
#         x7 = self.block_seven(x6_up)
#         x7_up = self.block_seven_up(x7)
#         x7_up = x7_up + x2
#
#         x8 = self.block_eight(x7_up)
#         x8_up = self.block_eight_up(x8)
#         x8_up = x8_up + x1
#         x9 = self.block_nine(x8_up)
#         # x9 = F.dropout3d(x9, p=0.5, training=True)
#         if self.has_dropout:
#             x9 = self.dropout(x9)
#         out = self.out_conv(x9)
#         return out
#
#
#     def forward(self, input, turnoff_drop=False):
#         if turnoff_drop:
#             has_dropout = self.has_dropout
#             self.has_dropout = False
#         features = self.encoder(input)
#         out = self.decoder(features)
#         if turnoff_drop:
#             self.has_dropout = has_dropout
#         return out
#----------------此V-net没有残差结构------------------


#----------------此V-net的input有增加ResidualConvBlock卷积层------------------
# class VNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
#         super(VNet_res, self).__init__()
#         self.has_dropout = has_dropout
#
#         self.input_conv = ConvBlock(1, n_channels, n_filters, normalization=normalization)
#         self.block_one = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization)
#         self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
#
#         self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
#
#         self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
#
#         self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
#
#         self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
#         self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
#
#         self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
#
#         self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
#
#         self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
#
#         self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization)
#         self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
#
#         self.dropout = nn.Dropout3d(p=0.5, inplace=False)
#         # self.__init_weight()
#
#     def encoder(self, input):
#         fusion = self.input_conv(input)
#         x1 = self.block_one(fusion)
#         x1_dw = self.block_one_dw(x1)
#
#         x2 = self.block_two(x1_dw)
#         x2_dw = self.block_two_dw(x2)
#
#         x3 = self.block_three(x2_dw)
#         x3_dw = self.block_three_dw(x3)
#
#         x4 = self.block_four(x3_dw)
#         x4_dw = self.block_four_dw(x4)
#
#         x5 = self.block_five(x4_dw)
#         # x5 = F.dropout3d(x5, p=0.5, training=True)
#         if self.has_dropout:
#             x5 = self.dropout(x5)
#
#         res = [x1, x2, x3, x4, x5]
#
#         return res
#
#     def decoder(self, features):
#         x1 = features[0]
#         x2 = features[1]
#         x3 = features[2]
#         x4 = features[3]
#         x5 = features[4]
#
#         x5_up = self.block_five_up(x5)
#         x5_up = x5_up + x4
#
#         x6 = self.block_six(x5_up)
#         x6_up = self.block_six_up(x6)
#         x6_up = x6_up + x3
#
#         x7 = self.block_seven(x6_up)
#         x7_up = self.block_seven_up(x7)
#         x7_up = x7_up + x2
#
#         x8 = self.block_eight(x7_up)
#         x8_up = self.block_eight_up(x8)
#         x8_up = x8_up + x1
#         x9 = self.block_nine(x8_up)
#         # x9 = F.dropout3d(x9, p=0.5, training=True)
#         if self.has_dropout:
#             x9 = self.dropout(x9)
#         out = self.out_conv(x9)
#         return out
#
#     def forward(self, input, turnoff_drop=False):
#         if turnoff_drop:
#             has_dropout = self.has_dropout
#             self.has_dropout = False
#         features = self.encoder(input)
#         out = self.decoder(features)
#         if turnoff_drop:
#             self.has_dropout = has_dropout
#         return out
#----------------此V-net的input有增加ResidualConvBlock卷积层------------------

class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=True):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization, relu=relu)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, relu=relu)

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, relu=relu)

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, relu=relu)

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, relu=relu)

        self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, relu=relu)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()


    def forward(self, input):

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]



class Encoder_new(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=True):
        super(Encoder_new, self).__init__()
        self.has_dropout = has_dropout

        self.Encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization, relu=relu,
                               has_dropout=has_dropout)

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, relu=relu)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, relu=relu)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)

        # self.attentionblock_one = AttentionBlock3D(n_filters, 1, n_filters)
        # self.attentionblock_two = AttentionBlock3D(n_filters*2, 1, n_filters*2)
        # self.attentionblock_three = AttentionBlock3D(n_filters*4, 1, n_filters*4)
        # self.attentionblock_four = AttentionBlock3D(n_filters*8, 1, n_filters*8)

        # self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        # self.pool2 = nn.AvgPool3d(kernel_size=4, stride=4)
        # self.pool3 = nn.AvgPool3d(kernel_size=8, stride=8)
        # self.pool4 = nn.AvgPool3d(kernel_size=16, stride=16)

        # self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.pool2 = nn.MaxPool3d(kernel_size=4, stride=4)
        # self.pool3 = nn.MaxPool3d(kernel_size=8, stride=8)
        # self.pool4 = nn.MaxPool3d(kernel_size=16, stride=16)

        self.aspp = ASPP(in_channel=n_filters * (16 + 8 + 4), depth=256)


    def forward(self, input):

        # fusion = self.input_conv(input)
        [x1, x2, x3, x4, x5] = self.Encoder(input)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)

        _, _, d, h, w = x7.size()
        x5_resample = F.interpolate(x5.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        x6_resample = F.interpolate(x6.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        # x7_resample = F.interpolate(x7.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        # x8_resample = F.interpolate(x8.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        # x9_resample = F.interpolate(x9.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        model_feature = torch.cat([x5_resample, x6_resample, x7], dim=1)  #x5_resample
        model_feature_new = self.aspp(model_feature)
        # res = [x1, x2, x3, x4, x5, model_feature]

        res = [x1, x2, x7, model_feature_new]

        return res


class Decoder(nn.Module):
    def __init__(self, n_classes=1, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=True):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, relu=relu)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, relu=relu)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, relu=relu)

        self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization, relu=relu)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization, relu=relu)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, feature):

        x5_up = self.block_five_up(feature[-1])

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        feature_decoder = [feature[-1], x6, x7, x8, x9]

        return out, feature_decoder



class Decoder_new(nn.Module):
    def __init__(self, n_classes=1, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=True):
        super(Decoder_new, self).__init__()
        self.has_dropout = has_dropout

        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, relu=relu)

        self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization, relu=relu)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization, relu=relu)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x7 = features[2]
        # x5 = features[4]
        # model_feature = features[3]

        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        return out



class Decoder_skip(nn.Module):
    def __init__(self, n_classes=1, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=True):
        super(Decoder_skip, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, relu=relu)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, relu=relu)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, relu=relu)

        self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization, relu=relu)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization, relu=relu)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        x5 = feature[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        return out



class Decoder_cat(nn.Module):
    def __init__(self, n_classes=1, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=True):
        super(Decoder_cat, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, relu=relu)

        self.block_six = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, relu=relu)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 4, normalization=normalization, relu=relu)

        self.block_seven = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 2, normalization=normalization, relu=relu)

        self.block_eight = ResidualConvBlock(2, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 4, n_filters, normalization=normalization, relu=relu)

        self.block_nine = ResidualConvBlock(1, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.out_conv = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        x5 = feature[4]

        x5_up = self.block_five_up(x5)
        x5_up = torch.cat((x4, x5_up), dim=1)

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = torch.cat((x3, x6_up), dim=1)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = torch.cat((x2, x7_up), dim=1)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = torch.cat((x1, x8_up), dim=1)
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out




#-----------------------------以上为basic block----------------------------------------#
#--------------------------------以下为models------------------------------------------#



#--------------------------------全监督模型-------------------------------------#

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.Encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)
        self.Decoder_skip = Decoder_skip(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # self.__init_weight()

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.Encoder(input)
        out = self.Decoder_skip(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out



class SegNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(SegNet, self).__init__()
        self.has_dropout = has_dropout

        self.Encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.Decoder = Decoder(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features_encoder = self.Encoder(input)
        out, feature_decoder = self.Decoder(features_encoder)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, features_encoder, feature_decoder




class VNet_cat(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_cat, self).__init__()
        self.has_dropout = has_dropout

        self.Encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.Decoder_cat = Decoder_cat(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                                         has_dropout=has_dropout)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.Encoder(input)
        out = self.Decoder_cat(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out


#--------------------------------全/无监督模型-------------------------------------#



#---------------只利用high-level feature生成prior mask-------------------


# class VNet_att(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
#         super(VNet_att, self).__init__()
#         self.has_dropout = has_dropout
#
#         self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
#         self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
#
#         self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
#
#         self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
#
#         self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
#
#         self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
#         self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
#
#         self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
#         self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
#
#         self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
#         self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
#
#         self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
#         self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
#
#         self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization)
#         self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
#
#         self.dropout = nn.Dropout3d(p=0.5, inplace=False)
#
#         self.attentionblock_one = AttentionBlock3D(n_filters, 1, n_filters)
#         self.attentionblock_two = AttentionBlock3D(n_filters * 2, 1, n_filters * 2)
#         self.attentionblock_three = AttentionBlock3D(n_filters * 4, 1, n_filters * 4)
#         self.attentionblock_four = AttentionBlock3D(n_filters * 8, 1, n_filters * 8)
#         self.attentionblock_five = AttentionBlock3D(n_filters * 16, 1, n_filters * 16)
#         # self.__init_weight()
#
#     def encoder(self, input):
#         # fusion = self.input_conv(input)
#         x1 = self.block_one(input)
#         x1_dw = self.block_one_dw(x1)
#
#         x2 = self.block_two(x1_dw)
#         x2_dw = self.block_two_dw(x2)
#
#         x3 = self.block_three(x2_dw)
#         x3_dw = self.block_three_dw(x3)
#
#         x4 = self.block_four(x3_dw)
#         x4_dw = self.block_four_dw(x4)
#
#         x5 = self.block_five(x4_dw)
#         # x5 = F.dropout3d(x5, p=0.5, training=True)
#         if self.has_dropout:
#             x5 = self.dropout(x5)
#
#         res = [x1, x2, x3, x4, x5]
#
#         return res
#
#
#     def encoder_new(self, prior_input, prior_mask = None):
#
#         # fusion = self.input_conv(prior_input)
#         supp_feat_0 = self.block_one(prior_input)
#         supp_feat_1 = self.block_one_dw(supp_feat_0)
#
#         supp_feat_2 = self.block_two(supp_feat_1)
#         supp_feat_2 = self.block_two_dw(supp_feat_2)
#
#         supp_feat_3 = self.block_three(supp_feat_2)
#         supp_feat_3 = self.block_three_dw(supp_feat_3)
#
#         supp_feat_4 = self.block_four(supp_feat_3)
#         supp_feat_4 = self.block_four_dw(supp_feat_4)
#
#         if prior_mask is not None:
#             mask = F.interpolate(prior_mask.type(torch.float32), size=(supp_feat_4.size(2), supp_feat_4.size(3), \
#                                                 supp_feat_4.size(4)), mode='trilinear', align_corners=True)
#             x = self.block_five(supp_feat_4 * mask)
#         else:
#             x = self.block_five(supp_feat_4)
#
#         return x
#
#
#     def decoder(self, features):
#         x1 = features[0]
#         x2 = features[1]
#         x3 = features[2]
#         x4 = features[3]
#         x5 = features[4]
#
#         x5_up = self.block_five_up(x5)
#         x5_up = x5_up + x4
#
#         x6 = self.block_six(x5_up)
#         x6_up = self.block_six_up(x6)
#         x6_up = x6_up + x3
#
#         x7 = self.block_seven(x6_up)
#         x7_up = self.block_seven_up(x7)
#         x7_up = x7_up + x2
#
#         x8 = self.block_eight(x7_up)
#         x8_up = self.block_eight_up(x8)
#         x8_up = x8_up + x1
#         x9 = self.block_nine(x8_up)
#         # x9 = F.dropout3d(x9, p=0.5, training=True)
#         if self.has_dropout:
#             x9 = self.dropout(x9)
#         out = self.out_conv(x9)
#         return out
#
#
#     def forward(self, input, prior_input, prior_mask, turnoff_drop=False):
#         if turnoff_drop:
#             has_dropout = self.has_dropout
#             self.has_dropout = False
#         features = self.encoder(input)
#         model_feature = features[-1]
#         with torch.no_grad():
#             prior_feature = self.encoder_new(prior_input)
#
#         input_mask = train_free_prior_mask(model_feature, prior_feature, prior_mask.unsqueeze(1))
#
#         features[4] = self.attentionblock_five(features[4], input_mask)
#         features[3] = self.attentionblock_four(features[3], input_mask)
#         features[2] = self.attentionblock_three(features[2], input_mask)
#         features[1] = self.attentionblock_two(features[1], input_mask)
#         features[0] = self.attentionblock_one(features[0], input_mask)
#
#         out = self.decoder(features)
#         if turnoff_drop:
#             self.has_dropout = has_dropout
#         return out, model_feature, prior_feature
#
#
#     # def __init_weight(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv3d):
#     #             torch.nn.init.kaiming_normal_(m.weight)
#     #         elif isinstance(m, nn.BatchNorm3d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#---------------只利用high-level feature生成prior mask-------------------



class VNet_att(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_att, self).__init__()
        self.has_dropout = has_dropout

        self.Encoder_new = Encoder_new(n_channels=n_channels, n_filters=n_filters, normalization=normalization,
                               has_dropout=has_dropout)

        self.Decoder_new = Decoder_new(n_classes=n_classes, n_filters=n_filters, normalization=normalization,
                                         has_dropout=has_dropout)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)


    def encoder_decoder_new(self, input, prior_input):

        features = self.Encoder_new(input)
        model_feature = features[-1]
        with torch.no_grad():
            features1 = self.Encoder_new(prior_input)
            prior_feature = features1[-1]
        out = self.Decoder_new(features)

        return out, model_feature, prior_feature


    # def encoder_decoder_new(self, input, prior_input, prior_mask):
    #
    #     features = self.encoder_new(input)
    #     model_feature = features[-1]
    #
    #     with torch.no_grad():
    #         features1 = self.encoder_new(prior_input)
    #         prior_feature = features1[-1]
    #     prior_mask_N = self.pool2(prior_mask.unsqueeze(1).float())  #pool的选取与特征上采样的大小有关
    #     generation_mask = train_free_prior_mask(model_feature, prior_feature, prior_mask_N)
    #     features[0] = self.attentionblock_one(features[0], generation_mask)
    #     features[1] = self.attentionblock_two(features[1], generation_mask)
    #     out = self.decoder_new(features)
    #     # out = self.attentionblock_one(out, generation_mask)
    #     out = self.out_conv(out)
    #
    #     return out, model_feature, generation_mask


    def forward(self, input, prior_input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        out, model_feature, prior_feature = self.encoder_decoder_new(input, prior_input)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out, model_feature, prior_feature


    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()



class VNet_att_new(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='batchnorm', relu='LeakyReLU', has_dropout=False):
        super(VNet_att_new, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization, relu=relu)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, relu=relu)

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, relu=relu)

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, relu=relu)

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization, relu=relu)

        self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization, relu=relu)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization, relu=relu)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, relu=relu)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, relu=relu)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, relu=relu)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, relu=relu)

        self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, relu=relu)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization, relu=relu)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization, relu=relu)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.aspp = ASPP(in_channel=n_filters * (16+8+4), depth=256)
        self.fc_conv = nn.Conv3d(256, n_classes, 1, padding=0)

        # self.__init_weight()


    def encoder(self, input):
        # fusion = self.input_conv(input)

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res


    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)

        b, c, d, h, w = x7.size()
        x5_resample = F.interpolate(x5.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        x6_resample = F.interpolate(x6.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        # x7_resample = F.interpolate(x7.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        # x8_resample = F.interpolate(x8.float(), size=(d, h, w), mode='trilinear', align_corners=True)
        feature = torch.cat([x5_resample, x6_resample, x7], dim=1)  # x5_resample
        feature = self.aspp(feature)
        out_ds = self.fc_conv(feature)

        return out, out_ds



    def encoder_decoder(self, input, prior_input):

        res = self.encoder(input)
        out, out_ds = self.decoder(res)
        with torch.no_grad():
            res1 = self.encoder(prior_input)
            _, out_ds_prior = self.decoder(res1)

        return out, out_ds, out_ds_prior


    def forward(self, input, prior_input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        out, out_ds, out_ds_prior = self.encoder_decoder(input, prior_input)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out, out_ds, out_ds_prior


    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


#-------------------------------------------------------------------------------------------




if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    # from thop import profile
    # from thop import clever_format
    model = VNet_att(n_channels=2, n_classes=2)
    input = torch.randn(1, 2, 224, 96, 96)
    a = torch.randn(1, 2, 224, 96, 96)
    b = torch.randn(1, 224, 96, 96)
    out = model(input, a, b)
    print(out[0].size())
    # flops, params = profile(model, inputs=(input,))
    # print(flops, params)
    # macs, params = clever_format([flops, params], "%.3f")
    # print(macs, params)
    # print("VNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
