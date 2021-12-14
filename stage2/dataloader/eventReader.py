import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import random
from lib import fileTool as filLib
import numpy as np
from pathlib import Path
import pickle
import cv2
from dataloader.dataloaderBase import DistributedSamplerVali


# train-------------------------------------------------------------------
class eventReaderTrain(data.Dataset):
    def __init__(self, cfg=None):
        super(eventReaderTrain, self).__init__()
        self.cfg = cfg
        self.eventPath = cfg.pathTrainEvent
        self.numPerSamples = 10

        self.eventGroups = filLib.getAllFiles(self.eventPath, 'pkl')
        self.len = len(self.eventGroups)
        self.size = cfg.trainSize

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        with open(self.eventGroups[index], 'rb') as f:
            record = pickle.load(f)

        I0 = record['I0']
        It = record['It']
        I1 = record['I1']

        F0t = record['F0t']
        F1t = record['F1t']

        E0 = record['E0']
        Et = record['Et']
        E1 = record['E1']

        C, H, W = It.shape
        # crop-------------------------------------------------------------------------------
        # i = random.randint(0, H - self.size[0])
        i = 0
        # j = random.randint(0, W - self.size[1])
        j = 0

        I0 = I0[:, i: i + self.size[0], j:j + self.size[1]]
        It = It[:, i: i + self.size[0], j:j + self.size[1]]
        I1 = I1[:, i: i + self.size[0], j:j + self.size[1]]

        F0t = F0t[:, i: i + self.size[0], j:j + self.size[1]]
        F1t = F1t[:, i: i + self.size[0], j:j + self.size[1]]

        E0 = E0[:, i: i + self.size[0], j:j + self.size[1]]
        Et = Et[:, i: i + self.size[0], j:j + self.size[1]]
        E1 = E1[:, i: i + self.size[0], j:j + self.size[1]]
        # to Tensor-------------------------------------------------------------------------------
        I0 = torch.from_numpy(I0.copy()).float() / 127.5 - 1  # (-1, 1)
        I0 = I0.repeat((3, 1, 1))

        It = torch.from_numpy(It.copy()).float() / 127.5 - 1  # (-1, 1)
        It = It.repeat((3, 1, 1))

        I1 = torch.from_numpy(I1.copy()).float() / 127.5 - 1  # (-1, 1)
        I1 = I1.repeat((3, 1, 1))

        F0t = torch.from_numpy(F0t.copy()).float()
        F1t = torch.from_numpy(F1t.copy()).float()

        E0 = torch.from_numpy(E0.copy()).float()
        Et = torch.from_numpy(Et.copy()).float()
        E1 = torch.from_numpy(E1.copy()).float()

        return I0, It, I1, F0t, F1t, E0, Et, E1

    def getFramGroups(self):
        framDirs = filLib.getAllFiles(self.eventPath, 'pkl')
        framDirs.sort(key=lambda x: int(Path(x).stem))
        return framDirs


def createEventVITrain(cfg=None):
    trainDataset = eventReaderTrain(cfg)

    if cfg.envDistributed:
        trainSampler = DistributedSampler(trainDataset, num_replicas=cfg.envWorldSize, rank=cfg.envRank)

        trainLoader = data.DataLoader(dataset=trainDataset,
                                      batch_size=cfg.trainBatchPerGPU,
                                      shuffle=False,
                                      num_workers=cfg.envWorkers,
                                      pin_memory=False,  # False if memory is not enough
                                      drop_last=False,
                                      sampler=trainSampler
                                      )
        return trainSampler, trainLoader
    else:
        trainLoader = data.DataLoader(dataset=trainDataset,
                                      batch_size=cfg.trainBatch,
                                      shuffle=True,
                                      num_workers=cfg.envWorkers,
                                      pin_memory=False,  # False if memory is not enough
                                      drop_last=False
                                      )
        return None, trainLoader


# Vali---------------------------------------------------------------------------------------
class eventReaderVali(data.Dataset):
    def __init__(self, cfg=None):
        super(eventReaderVali, self).__init__()
        self.cfg = cfg
        self.eventPath = cfg.pathValEvent
        self.scale = cfg.valScale

        self.eventGroups = filLib.getAllFiles(self.eventPath, 'pkl')
        self.len = len(self.eventGroups)
        # self.transform = self._transformInit()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        with open(self.eventGroups[index], 'rb') as f:
            record = pickle.load(f)

        IV = record['IV']
        # IT = record['IT']
        ET = record['ET']

        E0 = record['E0']
        E1 = record['E1']

        targetPath = [str(record['targetPath'])]

        IV = [self.processImg(array=i, scale=self.scale) for i in IV]
        # IT = [self.processImg(array=i, scale=self.scale) for i in IT]
        ET = [self.processE(array=i, scale=self.scale) for i in ET]

        E0 = self.processE(array=E0, scale=self.scale)
        E1 = self.processE(array=E1, scale=self.scale)

        return IV, ET, E0, E1, targetPath

    def processImg(self, array, scale=1.0):
        tensor = torch.from_numpy(array).float() / 127.5 - 1
        tensor = tensor.repeat([3, 1, 1])
        tensor = tensor[:, 0:176, ...]
        if self.scale != 1:
            tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=scale).squeeze(0)
        return tensor

    def processE(self, array, scale=1.0):
        tensor = torch.from_numpy(array).float()
        tensor = tensor[:, 0:176, ...]
        if self.scale != 1:
            tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=scale).squeeze(0)
        return tensor.clamp(-10, 10)

    def getFramGroups(self):
        framDirs = filLib.getAllFiles(self.eventPath, 'pkl')
        framDirs.sort(key=lambda x: int(Path(x).stem))
        return framDirs


def createEventVIVali(cfg=None):
    ValiDataset = eventReaderVali(cfg)

    if cfg.envDistributed:
        valiSampler = DistributedSamplerVali(ValiDataset, num_replicas=cfg.envWorldSize, rank=cfg.envRank)

        valiLoader = data.DataLoader(dataset=ValiDataset,
                                     batch_size=cfg.testBatchPerGPU,
                                     shuffle=False,
                                     num_workers=cfg.envWorkers,
                                     pin_memory=False,  # False if memory is not enough
                                     drop_last=False,
                                     sampler=valiSampler
                                     )
        return valiSampler, valiLoader
    else:
        valiLoader = data.DataLoader(dataset=ValiDataset,
                                     batch_size=cfg.testBatch,
                                     shuffle=False,
                                     num_workers=cfg.envWorkers,
                                     pin_memory=False,  # False if memory is not enough
                                     drop_last=False
                                     )
        return None, valiLoader


# Test ----------------------------------------------------------------------------------------------
class eventReaderTest(data.Dataset):
    def __init__(self, cfg=None):
        super(eventReaderTest, self).__init__()
        self.cfg = cfg
        self.eventPath = cfg.pathTestEvent
        self.scale = cfg.testScale

        self.eventGroups = filLib.getAllFiles(self.eventPath, 'pkl')
        self.len = len(self.eventGroups)
        # self.transform = self._transformInit()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fileName = self.eventGroups[index]
        with open(fileName, 'rb') as f:
            record = pickle.load(f)

        IV = record['IV']
        IT = record['IT']
        ET = record['ET']

        IV = [self.processImg(array=i, scale=self.scale) for i in IV]
        IT = [self.processImg(array=i, scale=self.scale) for i in IT]
        ET = [self.processE(array=i, scale=self.scale) for i in ET]

        return fileName, IV, IT, ET

    def processImg(self, array, scale=1.0):
        tensor = torch.from_numpy(array).float() / 127.5 - 1
        if self.scale != 1:
            tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=scale).squeeze(0)
        return tensor

    def processE(self, array, scale=1.0):
        tensor = torch.from_numpy(array).float()
        if self.scale != 1:
            tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=scale).squeeze(0)
        return tensor

    def getFramGroups(self):
        framDirs = filLib.getAllFiles(self.eventPath, 'pkl')
        framDirs.sort(key=lambda x: int(Path(x).stem))
        return framDirs


def createEventVITest(cfg=None):
    testDataset = eventReaderTest(cfg)

    if cfg.envDistributed:
        testSampler = DistributedSamplerVali(testDataset, num_replicas=cfg.envWorldSize, rank=cfg.envRank)

        testLoader = data.DataLoader(dataset=testDataset,
                                     batch_size=cfg.testBatchPerGPU,
                                     shuffle=False,
                                     num_workers=cfg.envWorkers,
                                     pin_memory=False,  # False if memory is not enough
                                     drop_last=False,
                                     sampler=testSampler
                                     )
        return testSampler, testLoader
    else:
        valiLoader = data.DataLoader(dataset=testDataset,
                                     batch_size=cfg.testBatch,
                                     shuffle=False,
                                     num_workers=cfg.envWorkers,
                                     pin_memory=False,  # False if memory is not enough
                                     drop_last=False
                                     )
        return None, valiLoader


def testTrain():
    import configs.configEVI as config
    import cv2
    cfg = config.Config({'SingleNode': '0'}, False)
    sample, Testloader = createEventVITrain(cfg)
    cv2.namedWindow('1', 0)
    for trainIndex, (I0t, I1t, It, Et) in enumerate(Testloader, 0):
        I0t = ((I0t[0] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
        I1t = ((I1t[0] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
        It = ((It[0] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)

        print('check I0t, I1t')
        cv2.imshow('1', np.concatenate([I1t, I0t], axis=1))
        cv2.waitKey(0)

        cv2.imshow('1', np.concatenate([It, It], axis=1))
        cv2.waitKey(0)

        Et = Et[0].cpu().numpy().astype(np.float32)

        for eIdx, p in enumerate(Et):
            eventImg = p
            eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                np.uint8)
            eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

            img = It.copy()

            img[:, :, 0][p != 0] = 0

            img[:, :, 2][p > 0] = 255
            img[:, :, 1][p < 0] = 255

            cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255),
                        5,
                        cv2.LINE_AA)

            cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
            cv2.waitKey(0)


def checkTest():
    import configs.configTest as config
    import cv2
    cfg = config.Config({'SingleNode': '0'}, False)
    sample, Testloader = createEventVITest(cfg)
    cv2.namedWindow('1', 0)
    for trainIndex, (fileName, IV, IT, ET) in enumerate(Testloader, 0):
        for Iv in IV:
            Iv = ((Iv[0] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
            cv2.imshow('1', Iv)
            cv2.waitKey(0)
        for It in IT:
            It = ((It[0] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
            cv2.imshow('1', It)
            cv2.waitKey(0)

        ET = [i[0].cpu().numpy().astype(np.float32) for i in ET]
        for It, Et in zip(IT, ET):
            It = ((It[0] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
            for eIdx, p in enumerate(Et):
                eventImg = p
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = It.copy()

                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(0)


if __name__ == '__main__':
    # testTrain()
    checkTest()
