import torch
import torch.nn as nn
from models.baseModule import FuseBlock, BaseNet, Interp


class FuseNet(BaseNet):
    def __init__(self):
        super(FuseNet, self).__init__()
        self.ConvIn = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False)
        self.AADBlk1 = FuseBlock(cin=512, cout=256, c_ef=512)
        self.AADBlk2 = FuseBlock(cin=256, cout=128, c_ef=256)
        self.AADBlk3 = FuseBlock(cin=128, cout=64, c_ef=128)
        self.AADBlk4 = FuseBlock(cin=64, cout=32, c_ef=64)
        self.Up2x = Interp(scale=2)
        self.ItStage1 = nn.Sequential(nn.ReplicationPad2d([1, 1, 1, 1]),
                                      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
                                      )

        self.randomInitNet()

    def forward(self, z_e, z_f):
        ST16x = self.ConvIn(torch.cat([z_e[0], z_f[0]], dim=1))  # 64

        ST8x = self.AADBlk1(self.Up2x(ST16x), z_e[1], z_f[1])  # 32

        ST4x = self.AADBlk2(self.Up2x(ST8x), z_e[2], z_f[2])  # 16

        ST2x = self.AADBlk3(self.Up2x(ST4x), z_e[3], z_f[3])  # 8

        ST1x = self.AADBlk4(self.Up2x(ST2x), z_e[4], z_f[4])  # 4

        ItStage1 = self.ItStage1(ST1x)

        return ItStage1