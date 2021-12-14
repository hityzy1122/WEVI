import ast
import argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, default='configEVI',
                    help='Algorithm configuration file name')

parser.add_argument('--initNode', type=str, required=False, default='HK-IDC2-10-1-75-52',
                    help='Node for init')

parser.add_argument('--gpuList', type=str, required=False, default='{"SingleNode":"0"}',
                    help='gpuList for init')

parser.add_argument('--reuseGPU', type=int, required=False, default=0,
                    help='reuseGPU or not')

parser.add_argument('--envDistributed', type=int, required=False, default=0,
                    help='reuseGPU or not')

parser.add_argument('--expName', type=str, required=False, default='Demo_train_on_lowfps',
                    help='reuseGPU or not')

args = parser.parse_args()
config = import_module('configs.' + args.config)
gpuList = ast.literal_eval(args.gpuList)

reuseGPU = args.reuseGPU
if not reuseGPU:
    gpuList = None
cfg = config.Config(gpuList, int(args.envDistributed), str(args.expName))

import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback

from lib import fileTool as flLib
from lib import metrics as mrcLib
from lib import checkTool as ckLib
from lib import distribTool as distLib
from lib import lossTool as lossLib
from lib.dlTool import getOptimizer, getScheduler
from lib.fitTool import getAccParam, getAccFlow
from lib.warp import forwadWarp
from lib.pwcNet.pwcNet import PWCDCNet
from lib.visualTool import viz
import torch
import torch.nn as nn
from torch.distributed import init_process_group
from torch.backends import cudnn

from models.Generator import Generator
from dataloader import eventReader

from apex import amp
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model


class Trainer(object):
    def __init__(self, check: bool = False):

        if 0 == cfg.envRank:
            self.writer = cfg.trainWriter
            self.logger = cfg.trainLogger
            self.fileBuffer = flLib.fileBuffer(cfg.trainMaxSave)
            self.fileBest = flLib.fileBuffer(cfg.trainMaxSave)
            self.valiBuffer = flLib.fileBuffer(cfg.trainMaxSave)

        self.envDistributed = cfg.envDistributed
        self.envParallel = cfg.envParallel
        self.envUseApex = cfg.envUseApex if not self.envParallel else False

        self.visual = cfg.trainVisual
        self.check = check

        self.lastEpoch = 0
        self.epoch = 0
        self.totalIter = 0
        self.bestPSNR = 0

        self.device0 = torch.device('cuda:{}'.format(cfg.envLocalRank)) if cfg.envNumGPUs > 0 else torch.device('cpu')

        self.dataTrainSampler, self.dataTrain = eventReader.createEventVITrain(cfg)

        self.Generator = Generator(cfg).to(self.device0)
        self.flownet = PWCDCNet(cfg=cfg).to(self.device0).eval()
        self.setRequiresGrad([self.flownet], False)

        self.optim = getOptimizer(self.Generator, cfg)
        self.scheduler = getScheduler(self.optim, cfg)

        self.fwarp = forwadWarp()

        self.initState()

        if cfg.envDistributed:
            self.Generator = convert_syncbn_model(self.Generator)
            self.Generator = DistributedDataParallel(self.Generator, delay_allreduce=True)
            self.flownet = convert_syncbn_model(self.flownet)
            self.flownet = DistributedDataParallel(self.flownet, delay_allreduce=True)
        elif cfg.envParallel:
            self.Generator = nn.DataParallel(self.Generator)
            self.flownet = nn.DataParallel(self.flownet)

    def trainingEVI(self):
        # init---------------------------------
        # self.optim.zero_grad()
        # self.optim.step()
        try:
            for epoch in range(self.lastEpoch, cfg.trainEpoch):

                self.epoch = epoch + 1
                if 0 == cfg.envRank:
                    self.logger.info('{} Epoch:{}'.format(args.expName, self.epoch).center(100, '='))
                    self.logger.info('\n')

                self.trainEpochEVI()

                if self.epoch % cfg.snapShot == 0:

                    if 0 == cfg.envRank:

                        self.saveState(bestModel=True)
                        path = str(Path(cfg.pathState) / f'bestEVI_epoch{self.epoch}.pth')
                        self.logger.info('Saving model in ' + path + '\n')

        except Exception as e:
            if 0 == cfg.envRank:
                self.logger.error(traceback.format_exc(limit=100))
                self.saveState(bestModel=False)
                path = str(Path(cfg.pathState) / f'epochAuto{self.epoch}.pth')
                self.logger.info('Saving model in ' + path + '\n')

    def trainEpochEVI(self):
        self.Generator.train()

        lr = self.updateLR()
        if self.dataTrainSampler is not None:
            self.dataTrainSampler.set_epoch(self.epoch)
        torch.cuda.empty_cache()

        if 0 == cfg.envRank:
            lossEVI = mrcLib.AverageMeter()
            pbar = tqdm(total=len(self.dataTrain))
        else:
            lossEVI = None

        for iter, (I0, It, I1, F0t, F1t, Et) in enumerate(self.dataTrain):

            I0: torch.Tensor = I0.to(self.device0)
            It: torch.Tensor = It.to(self.device0)
            I1: torch.Tensor = I1.to(self.device0)

            F0t: torch.Tensor = F0t.to(self.device0)
            F1t: torch.Tensor = F1t.to(self.device0)

            Et: torch.Tensor = Et.to(self.device0)

            with torch.no_grad():
                I0t = self.fwarp(I0, F0t)
                I1t = self.fwarp(I1, F1t)

            if cfg.netCheck:
                ckLib.checkTrainInput(I0t[0:1], I1t[0:1], It[0:1], Et[0:1])

            output = self.Generator(I0t, I1t, Et)

            loss = lossLib.CharbonnierLoss(output, It)

            self.optim.zero_grad()
            loss.backward()

            if self.check:
                ckLib.checkGrad(self.Generator)

            self.optim.step()

            self.totalIter += 1
            #

            if cfg.envDistributed:
                loss = distLib.reduceTensorMean(cfg, loss)
                loss = loss.detach().cpu().item()
            else:
                loss = loss.detach().cpu().item()

            if all([lossEVI is not None]):
                lossEVI.update(loss)

                pbar.set_description("lrEVI={:.7f}  lossEVI={:.5f} ".format(lr, lossEVI.avg))

                pbar.update(1)
            del loss

        if 0 == cfg.envRank:
            pbar.close()
            self.logger.info("\n")
            self.logger.info(
                "epoch={}  lrEVI={:.7f}  lossEVI={:.5f}".format(self.epoch, lr, lossEVI.avg).center(100, ' '))
            self.writer.add_scalar('lossEVI', lossEVI.avg, self.epoch)
            self.writer.add_scalar('lr', lr, self.epoch)

    def initState(self):
        if cfg.step in [1, 2, 3]:
            weightPath = str(cfg.pathWeight)
            if 0 == cfg.envRank:
                self.logger.info('Loading from ' + weightPath)
            checkpoints = torch.load(weightPath, map_location=torch.device('cpu'))

            self.lastEpoch = checkpoints['epoch']
            self.totalIter = checkpoints['totalIter']

            try:
                self.optim.load_state_dict(checkpoints['optim'])

                for param_group in self.optim.param_groups:
                    param_group['lr'] = cfg.lrInit

            except Exception as e:
                if 0 == cfg.envRank:
                    self.logger.error(traceback.format_exc(limit=100))
                    self.logger.info('loading state dict for optimazerEVI failed'.center(100, '!'))

            try:
                amp.load_state_dict(checkpoints['amp'])
            except Exception as e:
                if 0 == cfg.envRank:
                    self.logger.error(traceback.format_exc(limit=100))
                    self.logger.info('amp is not defined in pretained state'.center(100, '!'))
        else:
            if 0 == cfg.envRank:
                self.logger.info('training from scratch')

    def saveState(self, bestModel=True):
        if hasattr(self.Generator, 'module'):  # parallel may add module
            weight = self.Generator.module.state_dict()
        else:
            weight = self.Generator.state_dict()

        if self.envUseApex:
            stateDict = {'epoch': self.epoch,
                         'totalIter': self.totalIter,
                         'Generator': weight,
                         'optim': self.optim.state_dict(),
                         'amp': amp.state_dict(),
                         'bestPSNR': self.bestPSNR
                         }
        else:
            stateDict = {'epoch': self.epoch,
                         'totalIter': self.totalIter,
                         'Generator': weight,
                         'optim': self.optim.state_dict(),
                         'bestPSNR': self.bestPSNR
                         }
        if bestModel:
            savePath = str(Path(cfg.pathState) / f'bestEVI_epoch{self.epoch}.pth')
            self.fileBest(savePath)
        else:
            savePath = str(Path(cfg.pathState) / f'epochEVI{self.epoch}.pth')
            self.fileBuffer(savePath)

        torch.save(stateDict, savePath)

    def updateLR(self):
        # self.warmUpScheduler.step(epoch=self.epoch)
        self.scheduler.step(self.epoch)
        return self.optim.param_groups[0]['lr']

    def setRequiresGrad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    set_random_seed(cfg.setRandSeed)

    # if cudnn.enabled:
    #     cudnn.benchmark = True
    if cfg.envDistributed:
        assert cudnn.enabled, "Amp requires cudnn backend to be enabled."

        torch.cuda.set_device(cfg.envLocalRank)
        init_process_group(backend='nccl',
                           init_method='tcp://' + args.initNode + ':5801',
                           world_size=cfg.envWorldSize,
                           rank=cfg.envRank)

        distLib.synchronize()
    trainer = Trainer(check=cfg.netCheck)
    trainer.trainingEVI()
