import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import random
from lib import fileTool as filLib
import numpy as np
from pathlib import Path
import pickle
from dataloader.dataloaderBase import DistributedSamplerVali
import cv2

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

        Et = record['Et']

        C, H, W = It.shape

        # i = random.randint(0, H - self.size[0])
        i = 0
        # j = random.randint(0, W - self.size[1])
        j = 0

        I0 = I0[:, i: i + self.size[0], j:j + self.size[1]]
        It = It[:, i: i + self.size[0], j:j + self.size[1]]
        I1 = I1[:, i: i + self.size[0], j:j + self.size[1]]

        F0t = F0t[:, i: i + self.size[0], j:j + self.size[1]]
        F1t = F1t[:, i: i + self.size[0], j:j + self.size[1]]

        Et = Et[:, i: i + self.size[0], j:j + self.size[1]]

        I0 = torch.from_numpy(I0.copy()).float() / 127.5 - 1
        I0 = I0.repeat((3, 1, 1))

        It = torch.from_numpy(It.copy()).float() / 127.5 - 1
        It = It.repeat((3, 1, 1))

        I1 = torch.from_numpy(I1.copy()).float() / 127.5 - 1
        I1 = I1.repeat((3, 1, 1))

        F0t = torch.from_numpy(F0t.copy()).float()
        F1t = torch.from_numpy(F1t.copy()).float()

        Et = torch.from_numpy(Et.copy()).float().clamp(-10, 10)

        return I0, It, I1, F0t, F1t, Et

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



def checkTrain():
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


if __name__ == '__main__':
    checkTrain()
