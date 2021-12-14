import cv2
import sys

sys.path.append('../')
from tqdm import tqdm
from lib import fileTool as FT
# from lib.pwcNet.pwcNet import PWCDCNet
# from lib.RAFT.raftNet import RAFT
# from lib.forwardWarpTorch.forwardWarp import forwadWarp
# import torch
from pathlib import Path
from DVSBase import ESIMReader
import numpy as np
from lib.visualTool import viz
import imageio
import pickle
from functools import partial
import multiprocessing
import os
from multiprocessing import Pool, RLock, freeze_support

"""
    zip related intensity, events, flows and warped intensity
"""


def getallSamples(allFiles):
    subSetSize = 4  # [0,1,2,3]
    length = len(allFiles)
    # allFiles = list(range(length))
    sampleList = []

    for start in range(0, length):
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
    allSamples = getallSamples(allFiles=allFiles)

    fIdxListV = [0, 1, 2, 3]
    # fIdxListT = [10, 11, 12]

    if vis:
        cv2.namedWindow('1', 0)
    pbar = tqdm(total=len(allSamples), position=int(curProc._identity[0]))
    for sIdx, sample in enumerate(allSamples):
        IV = []
        IT = []
        ET = []

        targetPath = Path(targetParent) / '{:07d}.pkl'.format(sIdx)
        # if targetPath.is_file():
        #     pbar.set_description('{}: GPU:{}'.format(Path(EventDir).stem, gpuId))
        #     pbar.update(1)
        #     continue

        ESIM = ESIMReader(fileNames=sample)

        # get I0, I10, I20, I30
        for fIdxV in fIdxListV:
            imgPath = ESIM.pathImgStart[fIdxV]
            img = imgPath.copy()
            IV.append(img.transpose([2, 0, 1]))
            if vis:
                cv2.imshow('1', img)
                cv2.waitKey(100)

        tF_1, tF0, tF1, tF2 = ESIM.tImgStart[0], ESIM.tImgStart[1], ESIM.tImgStart[2], ESIM.tImgStart[3]

        fStarts = np.linspace(start=tF_1, stop=tF0, num=5, endpoint=True).tolist()[1:-1]
        fCenters = np.linspace(start=tF0, stop=tF1, num=5, endpoint=True).tolist()[1:-1]
        fStops = np.linspace(start=tF1, stop=tF2, num=5, endpoint=True).tolist()[1:-1]
        for fStart, fCenter, fStop in zip(fStarts, fCenters, fStops):
            # for fIdx in fIdxListT:
            #     fCenter = ESIM.tImgStart[fIdx]
            #     fStart = ESIM.tImgStart[fIdx - 5]
            #     fStop = ESIM.tImgStart[fIdx + 5]

            tEvents = np.linspace(start=fStart, stop=fCenter, num=numEvFea // 2 + 1, endpoint=True).tolist() + \
                      np.linspace(start=fCenter, stop=fStop, num=numEvFea // 2 + 1, endpoint=True).tolist()[1::]
            relateEvents = np.zeros([numEvFea, ESIM.height, ESIM.width]).astype(np.int8)

            for eIdx in range(numEvFea):
                eStart, eEnd = tEvents[eIdx], tEvents[eIdx + 1]

                p = ESIM.aggregEvent(tStart=eStart, tStop=eEnd, P=None)

                relateEvents[eIdx, ...] = p
            ET.append(relateEvents.astype(np.int8))

        # get E0 ------------------------------------------------------------------------------------
        fCenter = ESIM.tImgStart[1]
        fStart = ESIM.tImgStart[0]
        fStop = ESIM.tImgStart[2]
        tEvents = np.linspace(start=fStart, stop=fCenter, num=numEvFea // 2 + 1, endpoint=True).tolist() + \
                  np.linspace(start=fCenter, stop=fStop, num=numEvFea // 2 + 1, endpoint=True).tolist()[1::]

        E0 = np.zeros([numEvFea, ESIM.height, ESIM.width]).astype(np.int8)

        for eIdx in range(numEvFea):
            eStart, eEnd = tEvents[eIdx], tEvents[eIdx + 1]

            p = ESIM.aggregEvent(tStart=eStart, tStop=eEnd, P=None)

            E0[eIdx, ...] = p
        #  get E1------------------------------------------------------------------------------------
        fCenter = ESIM.tImgStart[2]
        fStart = ESIM.tImgStart[1]
        fStop = ESIM.tImgStart[3]
        tEvents = np.linspace(start=fStart, stop=fCenter, num=numEvFea // 2 + 1, endpoint=True).tolist() + \
                  np.linspace(start=fCenter, stop=fStop, num=numEvFea // 2 + 1, endpoint=True).tolist()[1::]

        E1 = np.zeros([numEvFea, ESIM.height, ESIM.width]).astype(np.int8)

        for eIdx in range(numEvFea):
            eStart, eEnd = tEvents[eIdx], tEvents[eIdx + 1]

            p = ESIM.aggregEvent(tStart=eStart, tStop=eEnd, P=None)

            E1[eIdx, ...] = p

        if vis:
            for imgIdx, Et in enumerate(ET):
                Img = cv2.cvtColor(IV[1].transpose([1, 2, 0]), cv2.COLOR_GRAY2BGR)
                for eIdx, p in enumerate(Et):
                    eventImg = p.astype(np.float32)
                    eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                        np.uint8)
                    eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                    if vis:
                        img = Img.copy()
                        img[:, :, 0][p != 0] = 0

                        img[:, :, 2][p > 0] = 255
                        img[:, :, 1][p < 0] = 255

                        cv2.putText(img, '{}_{}'.format(imgIdx, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255),
                                    5,
                                    cv2.LINE_AA)

                        cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                        cv2.waitKey(300)

            for eIdx, p in enumerate(E0):

                Img = IV[1]
                Img = cv2.cvtColor(Img.transpose([1, 2, 0]), cv2.COLOR_GRAY2BGR)
                # for eIdx, p in enumerate(Et):

                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                if vis:
                    img = Img.copy()
                    img[:, :, 0][p != 0] = 0

                    img[:, :, 2][p > 0] = 255
                    img[:, :, 1][p < 0] = 255

                    cv2.putText(img, '{}_{}'.format('I0', eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255),
                                5,
                                cv2.LINE_AA)

                    cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                    cv2.waitKey(300)

            for eIdx, p in enumerate(E1):

                Img = IV[2]
                Img = cv2.cvtColor(Img.transpose([1, 2, 0]), cv2.COLOR_GRAY2BGR)
                # for eIdx, p in enumerate(Et):

                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                if vis:
                    img = Img.copy()
                    img[:, :, 0][p != 0] = 0

                    img[:, :, 2][p > 0] = 255
                    img[:, :, 1][p < 0] = 255

                    cv2.putText(img, '{}_{}'.format('I1', eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255),
                                5,
                                cv2.LINE_AA)

                    cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                    cv2.waitKey(300)

        record = {'IV': IV, 'IT': IT, 'ET': ET, 'E0': E0, 'E1': E1, 'targetPath': targetPath}
        with open(targetPath, 'wb') as fs:
            pickle.dump(record, fs)
        pbar.update(1)
    pbar.close()


def batchZip(srcDir, dstDir, numEvFea, numInter, vis, poolSize=1):
    # srcDir = '/mnt/lustre/yuzhiyang/dataset/GoPro_public/event/simEvents/'

    # zipFrameEvent(srcDir, dstDir, imgDir, vis=True)
    allSubDirs = FT.getSubDirs(srcDir)
    kernelFunc = partial(zipFrameEvent, TargetDir=dstDir,
                         numEvFea=numEvFea, numInter=numInter, vis=vis)

    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(processes=poolSize, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(func=kernelFunc, iterable=allSubDirs)
    p.close()
    p.join()
    # for subDir in allSubDirs:
    #     zipFrameEvent(subDir, dstDir, vis=True)


def mainServer():
    # dirs of simulated events
    srcDir = '/mnt/lustre/yuzhiyang/dataset/fastDVS/event/oneTest'

    # dirs of output samples
    dstDir = '/mnt/lustre/yuzhiyang/dataset/fastDVS/oneTest'

    # channels of event feature
    numEvFea = 8
    # number of frames to be interpolated
    numInter = 9
    vis = False
    poolSize = 8

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
    dstDir = '../dataset/fastDVS_dataset/test'

    # channels of event feature
    numEvFea = 8
    # number of frames to be interpolated
    numInter = 3
    vis = False
    poolSize = 1

    batchZip(srcDir=srcDir,
             dstDir=dstDir,
             numEvFea=numEvFea,
             numInter=numInter,
             vis=vis,
             poolSize=poolSize)


def check():
    srcDir = '/home/sensetime/data/event/DVS/slomoDVS3/test20'
    allFiles = FT.getAllFiles(srcDir, 'pkl')
    allFiles.sort(reverse=False)
    allNum = len(allFiles)
    cv2.namedWindow('1', 0)
    for file in allFiles:
        with open(file, 'rb') as fs:
            record = pickle.load(fs)
        IV = record['IV']
        IT = record['IT']
        ET = record['ET']
        print('check IV')
        for img in IV:
            cv2.imshow('1', img.transpose([1, 2, 0]))
            cv2.waitKey(200)
        print('check IT')
        for img in IT:
            cv2.imshow('1', img.transpose([1, 2, 0]))
            cv2.waitKey(200)
        print('check Et')
        for imgIdx, (Img, Et) in enumerate(zip(IT, ET)):
            # Et = (Et[0::2, ...] + Et[1::2, ...])[3:7, ::]
            Img = Img.transpose([1, 2, 0])
            for eIdx, p in enumerate(Et):
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = Img.copy()
                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}_{}'.format(imgIdx, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)
        print('check E0, E1')
        for imgIdx, (Img, Et) in enumerate(zip(IV[1:-1], [record['E0'], record['E1']])):
            # Et = (Et[0::2, ...] + Et[1::2, ...])[3:7, ::]
            Img = Img.transpose([1, 2, 0])
            for eIdx, p in enumerate(Et):
                eventImg = p.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = Img.copy()
                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}_{}'.format(imgIdx, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)


if __name__ == '__main__':
    # srun -p Pixel --nodelist=SH-IDC1-10-5-30-138 --cpus-per-task=22 --gres=gpu:8 python mainGetDVSTest_03.py

    # mainServer()
    mainLocal()
    # check()
    pass
