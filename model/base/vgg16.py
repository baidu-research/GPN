from torch import nn
import math
import torch.utils.model_zoo as model_zoo


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


class VGG(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.pretrained = pretrained
        self.features = make_layers(cfg['D'])
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        if self.pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        "Loading pretrained weights of conv layers from vgg16"
        state_dict = model_zoo.load_url(model_urls['vgg16'])

        self.features[0].weight.data.copy_(state_dict['features.0.weight'])
        self.features[2].weight.data.copy_(state_dict['features.2.weight'])
        self.features[5].weight.data.copy_(state_dict['features.5.weight'])
        self.features[7].weight.data.copy_(state_dict['features.7.weight'])
        self.features[10].weight.data.copy_(state_dict['features.10.weight'])
        self.features[12].weight.data.copy_(state_dict['features.12.weight'])
        self.features[14].weight.data.copy_(state_dict['features.14.weight'])
        self.features[17].weight.data.copy_(state_dict['features.17.weight'])
        self.features[19].weight.data.copy_(state_dict['features.19.weight'])
        self.features[21].weight.data.copy_(state_dict['features.21.weight'])

        # removing the 2nd last pooling layer
        self.features[23].weight.data.copy_(state_dict['features.24.weight'])
        self.features[25].weight.data.copy_(state_dict['features.26.weight'])
        self.features[27].weight.data.copy_(state_dict['features.28.weight'])


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512]  # noqa
}
