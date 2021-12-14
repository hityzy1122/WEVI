import os
import logging
import torch
from torch.backends import cudnn
from datetime import datetime
from lib import fileTool as FT
from pathlib import Path
from tensorboardX import SummaryWriter


class Config(object):
    def __init__(self, gpuList=None, envDistributed=False, expName='gopro_gn_'):
        super(Config, self).__init__()

        # always change---------------------------------------------------------------------------------------
        self.a_name = expName

        # 1:train stage2
        #   lr=5e-5

        # 2:continue train stage if killed
        #   lr=5e-5

        self.step = 1
        self.snapShot = 10

        self.pathOut = './output/'
        self.pathExp = 'Demo_train_on_lowfps_S1'  # dir name for experiment of stage 1
        self.pathWeight = f'./output/{self.pathExp}/state/bestEVI_epoch100.pth'  # path for weight trained in stage 1

        self.dump = False  # save image results or not
        self.outPathS2 = './output/Demo_train_on_lowfps_S1/Real_S2/'  # path to save image results

        self.trainSize = (176, 240)

        self.valScale = 1
        self.valNumInter = 3

        self.netCheck = False

        self.lrInit = 5e-4
        self.trainEpoch = 5000

        self.trainBatchPerGPU = 8
        self.testBatchPerGPU = 2 * self.trainBatchPerGPU
        if self.dump:
            self.testBatchPerGPU = 1

        self.lrGamma = 0.999  # decay scale for exp scheduler

        # network param-------------------------------------------------------------------------------------------
        self.netActivate = 'leakyrelu'  # prelu, relu, leakyrelu, swish

        self.netNorm = 'group'  # instance, lrn, bn, identity, group

        self.netInitType = 'xavier'  # normal, xavier, orthogonal, kaiming,default

        self.netInitGain = 0.2
        self.pathTrainEvent = str(Path(__file__).parents[2] / 'dataset/fastDVS_dataset/train')
        self.pathValEvent = str(Path(__file__).parents[2] / 'dataset/fastDVS_dataset/test')

        # optimizer-----------------------------------------------------------------------------------------------

        self.optPolicy = 'Adam'  # adam, sgd
        self.optBetas = [0.9, 0.999]
        self.optDecay = 0
        self.optMomentum = 0.995

        self.lrPolicy = 'exp'  # step, multistep, cosine, plateau, exp
        self.lrdecayIter = 100
        self.lrMilestones = [100, 150]

        self.trainMean = 0
        self.trainStd = 1

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

        # self.testBatchPerGPU = self.trainBatchPerGPU

        if self.envDistributed:
            self.trainBatch = self.trainBatchPerGPU * self.envWorldSize
            self.testBatch = self.testBatchPerGPU * self.envWorldSize
        elif self.envParallel:
            self.trainBatch = self.trainBatchPerGPU * self.envNumGPUs
            self.testBatch = self.testBatchPerGPU * self.envNumGPUs
            # self.testBatch = self.testBatchPerGPU
        else:  # CPU
            self.trainBatch = self.trainBatchPerGPU
            self.testBatch = self.testBatchPerGPU
            # self.valScale = 0.8
            self.valScale = 1
            self.trainSize = (64, 64)

        self.trainVisual = False

        # ----------------------------------------------------------------------------------------------------

        self.setRandSeed = 2021

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

        self.trainMaxSave = 10

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
        if 0 == self.step:
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
