from torch import nn

from model.base.vgg16 import VGG
from model.ellipse_target_layer import EllipseTargetLayer
from model.ellipse_proposal_layer import EllipseProposalLayer
from model.loss_layer import LossCls, LossEllipseSL1, LossEllipseKLD


class GPN(nn.Module):

    def __init__(self, cfg):
        super(GPN, self).__init__()
        self.num_anchors = len(cfg['ANCHOR_SCALES'])
        self.ellipse_target = EllipseTargetLayer(cfg)
        self.ellipse_proposal = EllipseProposalLayer(cfg)

        if cfg['base_model'] == 'vgg16':
            self.base_model = VGG(cfg['pretrained'])
        else:
            raise Exception(
                'base model : {} not supported...'.format(cfg['base_model']))

        self.loss_cls = LossCls()

        if cfg['ELLIPSE_LOSS'] == 'KLD':
            self.loss_ellipse = LossEllipseKLD(cfg)
        elif cfg['ELLIPSE_LOSS'] == 'SL1':
            self.loss_ellipse = LossEllipseSL1()
        else:
            raise Exception(
                'ELLIPSE_LOSS : {} not supported...'.format(
                    cfg['ELLIPSE_LOSS']))

        self.conv_gpn = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.relu_gpn = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(512, self.num_anchors * 2, kernel_size=1,
                                  stride=1, padding=0, bias=True)
        self.conv_ellipse = nn.Conv2d(512, self.num_anchors * 5, kernel_size=1,
                                      stride=1, padding=0, bias=True)

        # GPN related initialization
        self.conv_gpn.weight.data.normal_(0, 0.01)
        self.conv_gpn.bias.data.zero_()
        self.conv_cls.weight.data.normal_(0, 0.01)
        self.conv_cls.bias.data.zero_()
        self.conv_ellipse.weight.data.normal_(0, 0.01)
        self.conv_ellipse.bias.data.zero_()

    def cuda(self, device=None):
        self.ellipse_target = self.ellipse_target.cuda()
        self.ellipse_proposal = self.ellipse_proposal.cuda()
        self.loss_ellipse = self.loss_ellipse.cuda()
        return self._apply(lambda t: t.cuda(device))

    def forward(self, img):
        base_feat = self.base_model(img)

        x = self.conv_gpn(base_feat)
        x = self.relu_gpn(x)

        batch_size, _, feat_height, feat_width = x.shape

        out_cls = self.conv_cls(x)
        out_ellipse = self.conv_ellipse(x)
        out_cls = out_cls.permute(0, 2, 3, 1).contiguous().view(
            batch_size, feat_height, feat_width, self.num_anchors, 2)
        out_ellipse = out_ellipse.permute(0, 2, 3, 1).contiguous().view(
            batch_size, feat_height, feat_width, self.num_anchors, 5)

        return (out_cls, out_ellipse)
