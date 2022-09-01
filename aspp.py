import torch
import torch.nn as nn



class DeeplabV3(ResNet):

    def __init__(self, n_class, block, layers, pyramids, grids, output_stride=16):
        self.inplanes = 64
        super(DeeplabV3, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        else:
            raise NotImplementedError

        # Backbone Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # h/4, w/4

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0]) # h/4, w/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1]) # h/8, w/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2]) # h/16,w/16
        self.layer4 = self._make_MG_unit(block, 512, blocks=grids, stride=strides[3], rate=rates[3])# h/16,w/16

        # Deeplab Modules
        self.aspp1 = ASPP_module(2048, 256, rate=pyramids[0])
        self.aspp2 = ASPP_module(2048, 256, rate=pyramids[1])
        self.aspp3 = ASPP_module(2048, 256, rate=pyramids[2])
        self.aspp4 = ASPP_module(2048, 256, rate=pyramids[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        # get result features from the concat
        self._conv1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        # generate the final logits
        self._conv2 = nn.Conv2d(256, n_class, kernel_size=1, bias=False)

        self.init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)


    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        # image-level features
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self._conv1(x)
        x = self._conv2(x)

        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x




class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()