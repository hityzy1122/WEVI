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

parser.add_argument('--expName', type=str, required=False, default='DVS_S2FullHard_gn_',
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

import torch
import torch.nn as nn
from torch.distributed import init_process_group
from torch.backends import cudnn

from models.Generator import Generator
from models.attnRefine import refineNet
from dataloader import eventReader
from skimage.measure import compare_ssim
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
        self.dataValiSampler, self.dataValidation = eventReader.createEventVIVali(cfg)

        self.Generator = Generator(cfg=cfg).to(self.device0)
        self.flownet = PWCDCNet(cfg=cfg).to(self.device0)
        self.refineNet = refineNet(cfg=cfg).to(self.device0)

        self.setRequiresGrad([self.flownet, self.Generator], False)
        self.Generator.eval()
        self.setRequiresGrad([self.refineNet], True)

        self.optim = getOptimizer(self.refineNet, cfg)
        self.scheduler = getScheduler(self.optim, cfg)

        self.fwarp = forwadWarp()

        self.initState()
        self.SSIM = mrcLib.SSIM()

        if cfg.envDistributed:
            self.Generator = convert_syncbn_model(self.Generator)
            self.Generator = DistributedDataParallel(self.Generator, delay_allreduce=True)
            self.flownet = convert_syncbn_model(self.flownet)
            self.flownet = DistributedDataParallel(self.flownet, delay_allreduce=True)

            self.refineNet = convert_syncbn_model(self.refineNet)
            self.refineNet = DistributedDataParallel(self.refineNet, delay_allreduce=True)
        elif cfg.envParallel:
            self.Generator = nn.DataParallel(self.Generator)
            self.flownet = nn.DataParallel(self.flownet)
            self.refineNet = nn.DataParallel(self.refineNet)

    def trainingEVI(self):
        # init---------------------------------
        # self.optim.zero_grad()
        # self.optim.step()
        try:
            self.bestPSNR = self.validation()

            for epoch in range(self.lastEpoch, cfg.trainEpoch):

                self.epoch = epoch + 1
                if 0 == cfg.envRank:
                    self.logger.info('{} Epoch:{}'.format(args.expName, self.epoch).center(100, '='))
                    self.logger.info('\n')

                self.trainEpochEVI()

                if self.epoch % cfg.snapShot == 0:
                    psnrVali = self.validation()

                    if 0 == cfg.envRank:
                        self.saveState(bestModel=True)
                        path = str(Path(cfg.pathState) / 'bestEVI_epoch{}.pth').format(self.epoch)
                        self.logger.info('Saving model in ' + path + '\n')

        except Exception as e:
            if 0 == cfg.envRank:
                self.logger.error(traceback.format_exc(limit=100))
                self.saveState(bestModel=False)
                path = str(Path(cfg.pathState) / 'epochAuto{}.pth'.format(self.epoch))
                self.logger.info('Saving model in ' + path + '\n')

    def trainEpochEVI(self):
        self.refineNet.train()

        lr = self.updateLR()
        if self.dataTrainSampler is not None:
            self.dataTrainSampler.set_epoch(self.epoch)
        torch.cuda.empty_cache()

        if 0 == cfg.envRank:
            lossRec = mrcLib.AverageMeter()
            pbar = tqdm(total=len(self.dataTrain))
        else:
            lossRec = None

        for iter, (I0, It, I1, F0t, F1t, E0, Et, E1) in enumerate(self.dataTrain):

            I0: torch.Tensor = I0.to(self.device0)
            It: torch.Tensor = It.to(self.device0)
            I1: torch.Tensor = I1.to(self.device0)

            F0tOut: torch.Tensor = F0t.to(self.device0)
            F1tOut: torch.Tensor = F1t.to(self.device0)

            E0: torch.Tensor = E0.to(self.device0)
            Et: torch.Tensor = Et.to(self.device0)
            E1: torch.Tensor = E1.to(self.device0)

            with torch.no_grad():
                I0t = self.fwarp(I0, F0tOut)
                I1t = self.fwarp(I1, F1tOut)

            # if cfg.netCheck:
            #     ckLib.checkTrainInput(I0t, I1t, It, Et)
            with torch.no_grad():
                ItStage1, ST4x, ST2x, ST1x = self.Generator(I0t, I1t, Et)

                IE0 = torch.cat([I0.detach(), E0.detach()], dim=1)
                IEt = torch.cat([ItStage1.detach(), Et.detach()], dim=1)
                IE1 = torch.cat([I1.detach(), E1.detach()], dim=1)

            ItStage2 = self.refineNet(IE0, IEt, IE1, ST4x, ST2x, ST1x)

            recLoss = lossLib.CharbonnierLoss(ItStage2, It.detach())

            loss = recLoss

            self.optim.zero_grad()
            loss.backward()

            if self.check:
                ckLib.checkGrad(self.refineNet)

            self.optim.step()

            self.totalIter += 1

            if cfg.envDistributed:
                recLoss = distLib.reduceTensorMean(cfg, recLoss)
                recLoss = recLoss.detach().cpu().item()
            else:
                recLoss = recLoss.detach().cpu().item()


            if all([lossRec is not None]):
                lossRec.update(recLoss)

                pbar.set_description(f"lrEVI={lr:.7f}  lossRec={lossRec.avg:.5f} ")

                pbar.update(1)

        if 0 == cfg.envRank:
            pbar.close()
            self.logger.info("\n")
            self.logger.info("epoch={}  lrEVI={:.7f}  lossRec={:.5f}"
                             .format(self.epoch, lr, lossRec.avg).center(100, ' '))
            self.writer.add_scalar('lossRec', lossRec.avg, self.epoch)
            self.writer.add_scalar('lr', lr, self.epoch)

    def validation(self):
        self.refineNet.eval()
        if 0 == cfg.envRank:
            self.logger.info("totalIter={} Doing validation".format(self.totalIter).center(100, '-'))
            self.logger.info("\n")
            qbar = tqdm(total=len(self.dataValidation))

        with torch.no_grad():
            for num, (IV, ET, E0, E1, targetPath) in enumerate(self.dataValidation):
                if cfg.netCheck:
                    cv2.namedWindow('1', 0)
                    ckLib.checkValInput(IV, ET)

                if cfg.dump:
                    pklPath = str(targetPath[0][0])
                    allParts = Path(pklPath).parts
                    dirName, pklname = allParts[-2], allParts[-1]
                    pklIdx = int(Path(pklname).stem) + 1

                    I0_ = IV[1]
                    I0_ = ((I0_[0].clamp(-1, 1) + 1) * 127.5).permute([1, 2, 0]).byte().cpu().numpy()

                    I0Name = Path(cfg.outPathS2) / Path(dirName) / Path('{:04d}_00.png'.format(pklIdx))
                    flLib.mkPath(I0Name.parent)
                    cv2.imwrite(str(I0Name), I0_)

                I_1, I0, I1, I2 = [i.to(self.device0) for i in IV]

                E0 = E0.to(self.device0)
                E1 = E1.to(self.device0)

                IE0 = torch.cat([I0, E0], dim=1)
                IE1 = torch.cat([I1, E1], dim=1)

                F0_1 = self.flownet(I0, I_1)
                F01 = self.flownet(I0, I1)
                a0, b0 = getAccParam(F0_1, F01)

                F12 = self.flownet(I1, I2)
                F10 = self.flownet(I1, I0)
                a1, b1 = getAccParam(F12, F10)

                intNum = len(ET)
                NameList = ['01', '02', '03']
                for idxT, Et in enumerate(ET):

                    Et = Et.to(self.device0)
                    timeF = (idxT + 1.0) / (intNum + 1.0)
                    F0tOut, F1tOut = getAccFlow(a0=a0, b0=b0, a1=a1, b1=b1, t=timeF, device=self.device0)

                    Nt, Ct, Ht, Wt = Et.shape
                    I0t = self.fwarp(I0, F0tOut)
                    I1t = self.fwarp(I1, F1tOut)
                    # if cfg.netCheck:
                    #     ckLib.checkIE(I0, I1, It, Et)

                    ItStage1, ST4x, ST2x, ST1x = self.Generator(I0t, I1t, Et)

                    # Et = Et[:, 0::2, ...] + Et[:, 1::2, ...]
                    IEt = torch.cat([ItStage1.detach(), Et], dim=1)

                    ItStage2 = self.refineNet(IE0, IEt, IE1, ST4x, ST2x, ST1x)

                    # --------------------------------------------------------------------------------
                    if cfg.dump:
                        ItName = Path(cfg.outPathS2) / Path(dirName) / Path(
                            '{:04d}_{}.png'.format(pklIdx, NameList[idxT]))
                        flLib.mkPath(Path(ItName).parent)
                        It_ = ((ItStage2[0].clamp(-1, 1) + 1) * 127.5).permute([1, 2, 0]).byte().cpu().numpy()
                        cv2.imwrite(str(ItName), It_)
                    # --------------------------------------------------------------------------------

                if 0 == cfg.envRank:
                    qbar.update(1)

        if 0 == cfg.envRank:
            self.logger.info("\n")

        if 0 == cfg.envRank:
            qbar.close()
            self.logger.info("\n")
            self.logger.info("Done".format(self.totalIter).center(100, '-'))

        cfg.dump = False
        return 0

    def initState(self):
        if cfg.step in [1, 2, 3]:
            # weightPath = str(Path(cfg.pathState) / 'bestModel.pth')
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
            weightR = self.refineNet.module.state_dict()
        else:
            weight = self.Generator.state_dict()
            weightR = self.refineNet.state_dict()

        if self.envUseApex:
            stateDict = {'epoch': self.epoch,
                         'totalIter': self.totalIter,
                         'Generator': weight,
                         'attnRefine': weightR,
                         'optim': self.optim.state_dict(),
                         'amp': amp.state_dict(),
                         'bestPSNR': self.bestPSNR
                         }
        else:
            stateDict = {'epoch': self.epoch,
                         'totalIter': self.totalIter,
                         'Generator': weight,
                         'attnRefine': weightR,
                         'optim': self.optim.state_dict(),
                         'bestPSNR': self.bestPSNR
                         }
        if bestModel:
            savePath = str(Path(cfg.pathState) / 'bestEVI_epoch{}.pth').format(self.epoch)
            self.fileBest(savePath)
        else:
            savePath = str(Path(cfg.pathState) / 'epochEVI{}.pth'.format(self.epoch))
            self.fileBuffer(savePath)

        torch.save(stateDict, savePath)

    def updateLR(self):
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
