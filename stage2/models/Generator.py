import torch
import torch.nn.functional as F
from models.baseModule import BaseNet
from models.EventUnet import EventUnet
from models.FrameUnet import FrameUnet
from models.FuseNet import FuseNet
import math


class Generator(BaseNet):
    def __init__(self, cfg):
        super(Generator, self).__init__(cfg.netInitType, cfg.netInitGain)
        self.cfg = cfg
        self.netScale = 16

        self.eventUnet = EventUnet(cfg)
        self.frameUnet = FrameUnet(cfg)
        self.fuseNet = FuseNet()

        if cfg.step in [1, 2, 3]:
            self.initPreweight(cfg.pathWeight)

    def getWeight(self, pathPreWeight: str = None):
        checkpoints = torch.load(pathPreWeight, map_location=torch.device('cpu'))
        try:
            weightDict = checkpoints['Generator']
        except Exception as e:
            weightDict = checkpoints['model_state_dict']
        return weightDict

    def adap2Net(self, tensor: torch.Tensor):
        Height, Width = tensor.size(2), tensor.size(3)

        Height_ = int(math.floor(math.ceil(Height / self.netScale) * self.netScale))
        Width_ = int(math.floor(math.ceil(Width / self.netScale) * self.netScale))

        if any([Height_ != Height, Width_ != Width]):
            tensor = F.pad(tensor, [0, Width_ - Width, 0, Height_ - Height])

        return tensor

    def forward(self, I0t, I1t, Et):
        N, C, H, W = I0t.shape

        I0t = self.adap2Net(I0t)
        I1t = self.adap2Net(I1t)

        Et = self.adap2Net(Et)

        z_e = self.eventUnet(Et)
        z_f = self.frameUnet(torch.cat([I0t, I1t], dim=1))

        fusedOut, ST4x, ST2x, ST1x = self.fuseNet(z_e, z_f)

        ST1x = ST1x[:, :, 0:H, 0:W]
        ST2x = ST2x[:, :, 0:H // 2, 0:W // 2]
        ST4x = ST4x[:, :, 0:H // 4, 0:W // 4]

        output = fusedOut[:, :, 0:H, 0:W]

        return output, ST4x, ST2x, ST1x
