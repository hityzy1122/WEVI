import os
import logging
import torch
from torch.backends import cudnn
from datetime import datetime
from lib import fileTool as FT
from pathlib import Path
from tensorboardX import SummaryWriter


class Config(object):
    def __init__(self, gpuList=None, envDistributed=False, expName='gopro_gn_Test'):
        super(Config, self).__init__()

        # always change---------------------------------------------------------------------------------------
        self.a_name = expName

        self.step = 2
        self.snapShot = 20
        self.pathOut = './output/'
        self.pathExp = 'New_gn__202101161228'
        self.pathWeight = './output/New_gn__202101161228/state/bestModel_epoch400.pth'

        self.trainSize = (432, 768)
        # self.trainSize = (255, 255)
        self.testScale = 0.5
        self.valNumInter = 9

        # self.lrInit = 5e-4
        self.lrInit = 2e-5
        self.trainEpoch = 5000
        self.trainBatchPerGPU = 1
        self.lrGamma = 0.999  # decay scale for exp scheduler

        self.trainMean = 0
        self.trainStd = 1
        # network param-------------------------------------------------------------------------------------------
        self.netActivate = 'leakyrelu'  # prelu, relu, leakyrelu, swish
        if '_bn_' in self.a_name:
            self.netNorm = 'bn'  # instance, lrn, bn, identity, group
        if '_gn_' in self.a_name:
            self.netNorm = 'group'  # instance, lrn, bn, identity, group

        # self.pathTestEvent = str(Path(__file__).parents[1] / Path('dataset/event/testNew'))
        self.pathTestEvent = '/home/sensetime/data/VideoInterpolation/highfps/goPro/240fps/GoPro_public/event/testNew/'
        self.pathInference = '/home/sensetime/data/VideoInterpolation/highfps/goPro/240fps/GoPro_public/event/Inference/'
        self.envUseApex = False
        self.envApexLevel = 'O0'

        # hardware environment-----------------------------------------------------------------------------------
        self.envDistributed = envDistributed
        self.envnodeName = self.getEnviron('SLURMD_NODENAME', 'SingleNode')
        self.envWorldSize = self.getEnviron('SLURM_NTASKS', 1)
        self.envNodeID = self.getEnviron('SLURM_NODEID', 0)
        self.envRank = self.getEnviron('SLURM_PROCID', 0)
        self.envLocalRank = self.getEnviron('SLURM_LOCALID', 0)

        if gpuList is not None and self.envDistributed:
            self.gpulist = gpuList[self.envnodeName]
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpulist

        self.envNumGPUs = torch.cuda.device_count()
        if self.envNumGPUs > 0:
            assert (torch.cuda.is_available()) and cudnn.enabled

        self.envParallel = True if (self.envNumGPUs > 2 and not self.envDistributed) else False
        self.testBatchPerGPU = 1
        self.netInitType = 'xavier'  # normal, xavier, orthogonal, kaiming,default

        self.netInitGain = 0.2
        if self.envDistributed:
            self.testBatch = self.testBatchPerGPU * self.envWorldSize
        elif self.envParallel:
            self.testBatch = self.testBatchPerGPU * self.envNumGPUs
        else:  # CPU
            self.testBatch = self.testBatchPerGPU

        # self.netCheck = True
        self.netCheck = False

        # ----------------------------------------------------------------------------------------------------

        self.setRandSeed = 2020

        # path and init------------------------------------------------------------------------------------------
        self.envWorkers = 4 if self.envDistributed else 0
        self.pathExp, self.pathEvents, self.pathState, self.pathGif = self.expInit()

        # train config------------------------------------------------------------------------------------------
        if self.envRank == 0:
            self.trainLogger = self.logInit()

            if self.step in [1, 2]:
                checkpoints = torch.load(self.pathWeight, map_location=torch.device('cpu'))
                try:
                    totalIter = checkpoints['totalIter']
                except Exception as e:
                    totalIter = 0
                self.trainWriter = SummaryWriter(self.pathEvents, purge_step=totalIter)
            else:
                self.trainWriter = SummaryWriter(self.pathEvents)

        self.trainMaxSave = 2

        # finally ------------------------------------------------------------------------------------------------
        if 0 == self.envRank:
            self.record()

    def set_gpu_ids(self):
        str_ids = self.gpu_ids
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

    def expInit(self):
        if self.step not in [1, 2]:
            # if self.envRank == 0:
            now = datetime.now().strftime("%Y%m%d%H%M")
            pathExp = str(Path(self.pathOut) / '{}_{}'.format(self.a_name, now))
            pathEvents = str(Path(pathExp) / 'events')
            pathState = str(Path(pathExp) / 'state')
            pathGif = str(Path(pathExp) / 'gif')

            FT.mkPath(pathExp)
            FT.mkPath(pathEvents)
            FT.mkPath(pathState)
            FT.mkPath(pathGif)
        else:
            pathExp = str(Path(self.pathOut) / self.pathExp)
            assert Path(pathExp).is_dir(), pathExp
            pathEvents = str(Path(pathExp) / 'events')
            pathState = str(Path(pathExp) / 'state')
            pathGif = str(Path(pathExp) / 'gif')
        return pathExp, pathEvents, pathState, pathGif

    def logInit(self):
        logger = logging.getLogger(__name__)
        logfile = str(Path(self.pathExp) / 'log.txt')
        fh = logging.FileHandler(logfile, mode='a')
        formatter = logging.Formatter("%(asctime)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def getEnviron(self, key: str, default):

        out = os.environ[key] if all([key in os.environ, self.envDistributed]) else default
        if isinstance(default, int):
            out = int(out)
        elif isinstance(default, str):
            out = str(out)
        return out

    def record(self):
        logger_ = logging.getLogger(__name__ + 'sub')
        logging.basicConfig(level=logging.INFO)
        path = os.path.join(self.pathExp, 'config.txt')
        fh = logging.FileHandler(path, mode='w')
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        logger_.addHandler(fh)

        args = vars(self)
        logger_.info('-------------Config-------------------')
        for k, v in sorted(args.items()):
            logger_.info('{} = {}'.format(k, v))
        logger_.info('--------------End---------------------')
