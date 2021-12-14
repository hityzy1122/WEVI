import sys

sys.path.append('../')
from lib import fileTool as FT
from pathlib import Path
from tqdm import tqdm
from functools import partial
import multiprocessing
import os
from multiprocessing import Pool, RLock, freeze_support


def sortFunc(name: str):
    name = Path(name).stem
    return int(name)


def splitFile(testFile, dstDir):
    curProc = multiprocessing.current_process()
    # allFiles = FT.getAllFiles(subDir)

    # allFiles.sort(key=sortFunc)
    # num = len(allFiles)
    # testSubset = allFiles[0:num // 3]
    # pbar = tqdm(total=len(allFiles), position=int(curProc._identity[0]))
    # for test in allFiles:
    fileName = Path(testFile).stem
    dirName = str(Path(dstDir) / Path(fileName))

    targetName = dirName.replace('/train', '/test')
    FT.mkPath(str(Path(targetName).parent))
    FT.movFile(dirName, targetName)
    # pbar.update(1)
    # pbar.clear()
    # pbar.close()


def batchRun(srcDir, dstDir, poolSize=1):
    # srcDir = '/mnt/lustre/yuzhiyang/dataset/GoPro_public/event/simEvents/'

    # zipFrameEvent(srcDir, dstDir, imgDir, vis=True)
    allAedats = FT.getAllFiles(srcDir, 'aedat4')
    kernelFunc = partial(splitFile, dstDir=dstDir)

    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(processes=poolSize, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(func=kernelFunc, iterable=allAedats)
    p.close()
    p.join()
    # for subDir in allSubDirs:
    #     zipFrameEvent(subDir, dstDir, vis=True)


def mainServer():
    # dirs of simulated events
    srcDir = '/mnt/lustre/yuzhiyang/dataset/slomoDVS/aedat4/test'

    # dirs of output samples
    dstDir = '/mnt/lustre/yuzhiyang/dataset/slomoDVS3/event/train'

    poolSize = 20

    batchRun(srcDir=srcDir,
             dstDir=dstDir,
             poolSize=poolSize)


def mainLocal():
    # dirs of simulated events
    srcDir = '/home/sensetime/data/event/DVS/slomoDVS/event/train'

    # dirs of output samples
    dstDir = '/home/sensetime/data/event/DVS/slomoDVS/event/test'

    poolSize = 1

    batchRun(srcDir=srcDir,
             dstDir=dstDir,
             poolSize=poolSize)


if __name__ == '__main__':
    mainServer()
    # mainLocal()
