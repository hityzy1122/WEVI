from models.baseModule import BaseNet, conv3x3, deconv


class FrameUnet(BaseNet):
    def __init__(self, cfg):
        super(FrameUnet, self).__init__(cfg.netInitType)
        self.netScale = 16
        self.conv1 = conv3x3(6, 32, cfg, ks=1)  # 32, 1x
        self.conv2 = conv3x3(32, 64, cfg)  # 64, 2x
        self.conv3 = conv3x3(64, 128, cfg)  # 128, 4x
        self.conv4 = conv3x3(128, 256, cfg)  # 256, 8x
        self.conv5 = conv3x3(256, 256, cfg)  # 256, 16x

        self.deconv2 = deconv(256, 256, 3, cfg)  # 256, 9x
        self.deconv3 = deconv(512, 128, 3, cfg)  # 128, 4x
        self.deconv4 = deconv(256, 64, 3, cfg)  # 64, 2x
        self.deconv5 = deconv(128, 32, 3, cfg)  # 32, 1x
        self.randomInitNet()

    def forward(self, Xt):
        feat1 = self.conv1(Xt)  # 32
        feat2 = self.conv2(feat1)  # 64
        feat3 = self.conv3(feat2)  # 128
        feat4 = self.conv4(feat3)  # 256
        z16 = self.conv5(feat4)  # 256

        z8 = self.deconv2(z16, feat4)  # 512
        z4 = self.deconv3(z8, feat3)  # 256
        z2 = self.deconv4(z4, feat2)  # 128
        z1 = self.deconv5(z2, feat1)  # 64

        return z16, z8, z4, z2, z1