import cv2
import sys
import os

sys.path.append('../')
from lib import fileTool as FT
from lib.pwcNet.pwcNet import PWCDCNet
from lib.forwardWarpTorch.forwardWarp import forwadWarp
import torch
from pathlib import Path
from DVSBase import ESIMReader
import numpy as np
import pickle

from tqdm import tqdm
from multiprocessing import Pool, RLock, freeze_support
from functools import partial
import multiprocessing

"""
    zip related intensity, events, flows and warped intensity
"""


def getallSamples(allFiles, numInter=9):
    subSetSize = 5  # [0,1,2,3,4], [1,2,3,4,5], [2,3,4,5,6]
    length = len(allFiles)
    sampleList = []
    for start in range(length):
        try:
            lowT = allFiles[start]
            upT = allFiles[start + subSetSize]
            sampleList.append(allFiles[start:start + subSetSize])
        except Exception as e:
            break

    return sampleList


def zipFrameEvent(EventDir: str, TargetDir: str, numEvFea=20, numInter=9, vis=False):
    curProc = multiprocessing.current_process()
    targetParent = Path(TargetDir) / Path(EventDir).stem
    gpuId = (curProc._identity[0] - 1) % 8
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuId)
    FT.mkPath(targetParent)

    allFiles = FT.getAllFiles(EventDir, 'pkl')
    allFiles.sort(key=lambda x: int(Path(x).stem), reverse=False)
    allSamples = getallSamples(allFiles=allFiles, numInter=3)
    flowNet = PWCDCNet().cuda().eval()

    warp = forwadWarp()

    if vis:
        cv2.namedWindow('1', 0)

    pbar = tqdm(total=len(allSamples), position=int(curProc._identity[0]))
    for sIdx, sample in enumerate(allSamples):
        targetPath = Path(targetParent) / '{:07d}.pkl'.format(sIdx)
        if targetPath.is_file():
            pbar.set_description('{}: GPU:{}'.format(Path(EventDir).stem, gpuId))
            pbar.update(1)
            continue

        ESIM = ESIMReader(fileNames=sample)

        I = []

        trainImgIdxs = [1, 2, 3]

        # get images --------------------------------------------------------------------------------
        if vis:
            for imgPath in ESIM.pathImgStart:
                cv2.imshow('1', imgPath)
                cv2.waitKey(100)

        for fidx in trainImgIdxs:
            imgPath = ESIM.pathImgStart[fidx]

            I.append(imgPath)
            if vis:
                cv2.imshow('1', I[-1])
                cv2.waitKey(200)

        # get E0 --------------------------------------------------------------------------------------------
        fCenter = ESIM.tImgStart[1]  # 1
        fStart = ESIM.tImgStart[0]  # 0
        fStop = ESIM.tImgStart[2]  # 2

        tEvents = np.linspace(start=fStart, stop=fCenter, num=numEvFea // 2 + 1, endpoint=True).tolist() + \
                  np.linspace(start=fCenter, stop=fStop, num=numEvFea // 2 + 1, endpoint=True).tolist()[1::]

        E0 = np.zeros([numEvFea, ESIM.height, ESIM.width]).astype(np.int8)

        for eIdx in range(numEvFea):
            eStart, eEnd = tEvents[eIdx], tEvents[eIdx + 1]

            p = ESIM.aggregEvent(tStart=eStart, tStop=eEnd, P=None)

            E0[eIdx, ...] = p

            if vis:
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)
                # imgPath = ESIM.pathImgStart[15 + eIdx // 2]
                imgPath = ESIM.pathImgStart[1]
                img = cv2.cvtColor(imgPath.copy(), cv2.COLOR_GRAY2BGR)
                # img = I[1].copy()
                img[:, :, 0][E0[eIdx, ...] != 0] = 0

                img[:, :, 2][E0[eIdx, ...] > 0] = 255
                img[:, :, 1][E0[eIdx, ...] < 0] = 255

                cv2.putText(img, '{}_{}'.format(1, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(300)
        E0 = E0.astype(np.int8)

        # get Et -----------------------------------------------------------------------------------
        fCenter = ESIM.tImgStart[2]  # 20
        fStart = ESIM.tImgStart[1]  # 15
        fStop = ESIM.tImgStart[3]  # 25

        tEvents = np.linspace(start=fStart, stop=fCenter, num=numEvFea // 2 + 1, endpoint=True).tolist() + \
                  np.linspace(start=fCenter, stop=fStop, num=numEvFea // 2 + 1, endpoint=True).tolist()[1::]

        Et = np.zeros([numEvFea, ESIM.height, ESIM.width]).astype(np.int8)

        for eIdx in range(numEvFea):
            eStart, eEnd = tEvents[eIdx], tEvents[eIdx + 1]

            p = ESIM.aggregEvent(tStart=eStart, tStop=eEnd, P=None)

            Et[eIdx, ...] = p

            if vis:
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                # imgPath = ESIM.pathImgStart[15 + eIdx // 2]
                imgPath = ESIM.pathImgStart[2]
                img = cv2.cvtColor(imgPath.copy(), cv2.COLOR_GRAY2BGR)
                # img = I[1].copy()
                img[:, :, 0][Et[eIdx, ...] != 0] = 0

                img[:, :, 2][Et[eIdx, ...] > 0] = 255
                img[:, :, 1][Et[eIdx, ...] < 0] = 255

                cv2.putText(img, '{}_{}'.format(1, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(300)
        Et = Et.astype(np.int8)

        # get E1 ------------------------------------------------------------------------------------------------
        fCenter = ESIM.tImgStart[3]  # 30
        fStart = ESIM.tImgStart[2]  # 25
        fStop = ESIM.tImgStart[4]  # 35

        tEvents = np.linspace(start=fStart, stop=fCenter, num=numEvFea // 2 + 1, endpoint=True).tolist() + \
                  np.linspace(start=fCenter, stop=fStop, num=numEvFea // 2 + 1, endpoint=True).tolist()[1::]

        E1 = np.zeros([numEvFea, ESIM.height, ESIM.width]).astype(np.int8)

        for eIdx in range(numEvFea):
            eStart, eEnd = tEvents[eIdx], tEvents[eIdx + 1]

            p = ESIM.aggregEvent(tStart=eStart, tStop=eEnd, P=None)

            E1[eIdx, ...] = p

            if vis:
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)
                # imgPath = ESIM.pathImgStart[15 + eIdx // 2]
                imgPath = ESIM.pathImgStart[3]
                img = cv2.cvtColor(imgPath.copy(), cv2.COLOR_GRAY2BGR)
                # img = I[1].copy()
                img[:, :, 0][E1[eIdx, ...] != 0] = 0

                img[:, :, 2][E1[eIdx, ...] > 0] = 255
                img[:, :, 1][E1[eIdx, ...] < 0] = 255

                cv2.putText(img, '{}_{}'.format(1, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(300)
        E1 = E1.astype(np.int8)

        I0 = I[0].transpose([2, 0, 1])
        It = I[1].transpose([2, 0, 1])
        I1 = I[2].transpose([2, 0, 1])

        with torch.no_grad():

            I0Cuda = torch.from_numpy(I0 / 255.0).cuda().unsqueeze(0).float()
            ItCuda = torch.from_numpy(It / 255.0).cuda().unsqueeze(0).float()
            I1Cuda = torch.from_numpy(I1 / 255.0).cuda().unsqueeze(0).float()

            F0t = flowNet(I0Cuda, ItCuda, iters=20, test_mode=True)

            F1t = flowNet(I1Cuda, ItCuda, iters=20, test_mode=True)

            if vis:
                It_ = (warp(I0Cuda, F0t))[0].cpu().numpy().transpose([1, 2, 0])
                It__ = (warp(I1Cuda, F1t))[0].cpu().numpy().transpose([1, 2, 0])

                cv2.imshow('1', It.transpose([1, 2, 0]))
                cv2.waitKey(200)

                cv2.imshow('1', It_)
                cv2.waitKey(200)

                cv2.imshow('1', It__)
                cv2.waitKey(200)

        if vis:
            cv2.imshow('1', I0.transpose([1, 2, 0]))
            cv2.waitKey(200)

            cv2.imshow('1', It.transpose([1, 2, 0]))
            cv2.waitKey(200)

            cv2.imshow('1', I1.transpose([1, 2, 0]))
            cv2.waitKey(200)

        record = {'I0': I0, 'It': It, 'I1': I1,
                  'Et': Et, 'E0': E0, 'E1': E1,
                  'F0t': F0t[0, ...].cpu().numpy(), 'F1t': F1t[0, ...].cpu().numpy()
                  }

        with open(targetPath, 'wb') as fs:
            pickle.dump(record, fs)
            pbar.set_description('{}: GPU:{}'.format(Path(EventDir).stem, gpuId))
            pbar.update(1)
    pbar.close()


def batchZip(srcDir, dstDir, numEvFea, numInter, vis, poolSize=1):
    allSubDirs = FT.getSubDirs(srcDir)
    kernelFunc = partial(zipFrameEvent, TargetDir=dstDir, numEvFea=numEvFea, numInter=numInter, vis=vis)

    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(processes=poolSize, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(func=kernelFunc, iterable=allSubDirs)
    p.close()
    p.join()


def mainServer():
    # dirs of simulated events
    srcDir = '/mnt/lustre/yuzhiyang/dataset/fastDVS/event/train'

    # dirs of output samples
    dstDir = '/mnt/lustre/yuzhiyang/dataset/fastDVS/train'

    # channels of event feature
    numEvFea = 8
    # number of frames to be interpolated
    numInter = 9
    vis = False
    poolSize = 20

    batchZip(srcDir=srcDir,
             dstDir=dstDir,
             numEvFea=numEvFea,
             numInter=numInter,
             vis=vis,
             poolSize=poolSize)


def mainLocal():
    # dirs of simulated events
    srcDir = '../dataset/fastDVS_process/train'

    # dirs of output samples
    dstDir = '../dataset/fastDVS_dataset/train'

    # channels of event feature
    numEvFea = 8
    # number of frames to be interpolated
    numInter = 9
    vis = False
    poolSize = 1

    batchZip(srcDir=srcDir,
             dstDir=dstDir,
             numEvFea=numEvFea,
             numInter=numInter,
             vis=vis,
             poolSize=poolSize)


def check():
    srcDir = '../dataset/fastDVS_dataset/train/'
    allSequences = FT.getSubDirs(srcDir)
    for sequence in allSequences:
        allFiles = FT.getAllFiles(sequence, 'pkl')
        allFiles.sort(reverse=False)
        cv2.namedWindow('1', 0)

        for file in allFiles:
            with open(file, 'rb') as fs:
                aSample = pickle.load(fs)

            print('check I')

            cv2.imshow('1', aSample['I0'].transpose([1, 2, 0]))
            cv2.waitKey(200)
            cv2.imshow('1', aSample['It'].transpose([1, 2, 0]))
            cv2.waitKey(200)
            cv2.imshow('1', aSample['I1'].transpose([1, 2, 0]))
            cv2.waitKey(200)

            print('check E0')
            for eIdx, p in enumerate(aSample['E0']):
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = cv2.cvtColor(aSample['I0'].transpose([1, 2, 0]).copy(), cv2.COLOR_GRAY2BGR)
                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)

            print('check Et')
            for eIdx, p in enumerate(aSample['Et']):
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = cv2.cvtColor(aSample['It'].transpose([1, 2, 0]).copy(), cv2.COLOR_GRAY2BGR)
                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)

            print('check E1')
            for eIdx, p in enumerate(aSample['E1']):
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = cv2.cvtColor(aSample['I1'].transpose([1, 2, 0]).copy(), cv2.COLOR_GRAY2BGR)
                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)


if __name__ == '__main__':
    # srun -p Pixel --nodelist=SH-IDC1-10-5-31-31 --cpus-per-task=22 --gres=gpu:8 python mainGetDVSTrain_02.py
    # mainServer()
    mainLocal()
    # check()
    pass
