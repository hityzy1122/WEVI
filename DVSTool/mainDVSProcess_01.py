import sys

sys.path.append('../')
import cv2
import lib.fileTool as FT
import numpy as np
import pickle
from pathlib import Path
from DVSBase import DVSReader, ESIMReader
from tqdm import tqdm
from multiprocessing import Pool, RLock, freeze_support
from functools import partial
import multiprocessing
import os


def simulate(dvsFile: str, outPath):
    curProc = multiprocessing.current_process()
    dvs = DVSReader(dvsFile)
    outDir = Path(outPath) / Path(dvsFile).stem
    FT.mkPath(outDir)
    pbar = tqdm(total=len(dvs.tImg), position=int(curProc._identity[0]))
    for idx in range(len(dvs.tImg) - 1):
        img = dvs.Img[idx]
        tImgStart = dvs.tImg[idx]
        tImgStop = dvs.tImg[idx + 1]
        sliceIdx = (np.array(dvs.tE) >= tImgStart) & (np.array(dvs.tE) < tImgStop)

        tE = np.array(dvs.tE)[sliceIdx].tolist()
        xE = np.array(dvs.xE)[sliceIdx].tolist()
        yE = np.array(dvs.yE)[sliceIdx].tolist()
        pE = np.array(dvs.pE)[sliceIdx].tolist()

        recordEvent = {'tE': tE, 'xE': xE, 'yE': yE, 'pE': pE,
                       'tImgStart': tImgStart, 'tImgStop': tImgStop,
                       'pathImgStart': img}
        targetPath = Path(outDir) / Path('{:07d}.pkl'.format(idx))

        with open(targetPath, 'wb') as fs:
            pickle.dump(recordEvent, fs)

        pbar.set_description('{}'.format(Path(dvsFile).stem))
        pbar.update(1)
    pbar.clear()
    pbar.close()


def batchESIM(dirPath: str, outPath, poolSize=4):
    allDvsFiles = FT.getAllFiles(dirPath, 'aedat4')
    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(processes=poolSize, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

    kernelFunc = partial(simulate, outPath=outPath)
    p.map(func=kernelFunc, iterable=allDvsFiles)
    p.close()
    p.join()


def mainDVSServer():
    # dataPath = '/mnt/lustre/yuzhiyang/dataset/slomeDVS/aedat4/train'
    dataPath = '/mnt/lustre/yuzhiyang/dataset/fastDVS/aedat4/oneTest/'
    # outPath = '/mnt/lustre/yuzhiyang/dataset/slomeDVS/event/train'
    outPath = '/mnt/lustre/yuzhiyang/dataset/fastDVS/event/oneTest/'
    batchESIM(dirPath=dataPath, outPath=outPath, poolSize=2)


def mainDVSLocal():
    dataPath = '../dataset/aedat4/train'
    outPath = '../dataset/fastDVS_process/train'
    batchESIM(dirPath=dataPath, outPath=outPath, poolSize=1)


def check():
    channel = 1
    cv2.namedWindow('1', 0)
    eventDirs = '/home/sensetime/data/event/DVS/slomoDVS/event'
    allsubEventDirs = FT.getSubDirs(eventDirs)
    for eventDir in allsubEventDirs:
        allEvents = FT.getAllFiles(eventDir, 'pkl')
        for fIdx, evePath in enumerate(allEvents):
            esim = ESIMReader(evePath)
            # relateEvents = np.zeros([2, esim['height'], esim['width']], np.int8)
            tEvents = np.linspace(start=esim.tImgStart[0], stop=esim.tImgStop[0], num=channel + 1,
                                  endpoint=True).tolist()
            for eIdx in range(channel):
                eStart = tEvents[eIdx]
                eStop = tEvents[eIdx + 1]
                relateEvents = esim.aggregEvent(eStart, eStop)

                eventImg = relateEvents.astype(np.float32)
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min() + 1e-5) * 255.0).astype(
                    np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = cv2.cvtColor(esim.pathImgStart[0].copy(), cv2.COLOR_GRAY2BGR)
                img[:, :, 0][relateEvents != 0] = 0

                img[:, :, 2][relateEvents > 0] = 255
                img[:, :, 1][relateEvents < 0] = 255

                cv2.putText(img, '{}_{}'.format(fIdx, eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)


if __name__ == '__main__':
    # srun -p Pixel --nodelist= --cpus-per-task=22 --job-name=Train python mainESIM.py
    # mainDVSServer()
    mainDVSLocal()
    # check()
    # mainSimGoproTest()
