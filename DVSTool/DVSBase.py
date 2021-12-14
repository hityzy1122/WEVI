from dv import AedatFile
import numpy as np
import torch
import cv2
import pickle


class DVSReader(object):
    def __init__(self, fileName):
        super(DVSReader, self).__init__()
        self.C = 3

        with AedatFile(fileName) as f:
            self.height, self.width = f['events'].size
            self.Events = np.hstack([packet for packet in f['events'].numpy()])
            self.tE = self.Events['timestamp']
            self.xE = self.Events['x']
            self.yE = self.Events['y']
            self.pE = 2 * self.Events['polarity'] - 1

            tImg = []
            Img = []
            # Img H*W*1
            for packet in f['frames']:
                tImg.append(packet.timestamp)
                Img.append(packet.image)

            # self.tImg = np.hstack(tImg)
            self.tImg = tImg
            # self.Img = np.expand_dims(np.dstack(Img).transpose([2, 0, 1]), axis=1)
            self.Img = Img

    def aggregEvent(self, tStart=0, tStop=1e20, P=1):
        reverse = False
        if tStart >= tStop:
            reverse = True
            tStart, tStop = tStop, tStart

        if P is not None:
            sliceIdx = (self.tE >= tStart) & (self.tE < tStop) & (self.pE == P)
        else:
            sliceIdx = (self.tE >= tStart) & (self.tE < tStop)

        target = torch.zeros((self.height, self.width)).half().to(self.pE.device)

        if not (1 in sliceIdx):
            return target.cpu().char().numpy()

        # tSlice = self.tE[sliceIdx]
        xSlice = self.xE[sliceIdx]
        ySlice = self.yE[sliceIdx]
        pSlice = self.pE[sliceIdx]

        index = ySlice * self.width + xSlice

        target.put_(index=index, source=pSlice, accumulate=True)
        target = target.clamp(-10, 10)

        # print(target.max().cpu().item(), target.min().cpu().item())
        if reverse:
            target = -target
        return target.cpu().char().numpy()


class ESIMReader(object):
    def __init__(self, fileNames=None):
        super(ESIMReader, self).__init__()
        tE = []
        xE = []
        yE = []
        pE = []
        self.pathImgStart = []
        self.pathImgStop = []
        self.tImgStart = []
        self.tImgStop = []
        if not isinstance(fileNames, list):
            fileNames = [fileNames]
        for fileName in fileNames:
            fs = open(fileName, 'rb')
            record: dict = pickle.load(fs)
            tE.extend(record['tE'])
            xE.extend(record['xE'])
            yE.extend(record['yE'])
            pE.extend(record['pE'])
            pathImgStart = record['pathImgStart']
            self.pathImgStart.append(pathImgStart)
            self.tImgStart.append(record['tImgStart'])
            self.tImgStop.append(record['tImgStop'])

            fs.close()

            # self.numStep = 10
            # self.tE = torch.from_numpy(np.array(record['tE'])).cuda()
        self.tE = torch.from_numpy(np.array(tE)).cuda()
        self.xE = torch.from_numpy(np.array(xE)).long().cuda()
        self.yE = torch.from_numpy(np.array(yE)).long().cuda()
        self.pE = torch.from_numpy(2 * np.array(pE) - 1).half().cuda()

        self.height = record.get('height', 180)
        self.width = record.get('width', 240)

    def aggregEvent(self, tStart=0, tStop=1e20, P=None):
        reverse = False
        if tStart >= tStop:
            reverse = True
            tStart, tStop = tStop, tStart

        if P is not None:
            sliceIdx = (self.tE >= tStart) & (self.tE < tStop) & (self.pE == P)
        else:
            sliceIdx = (self.tE >= tStart) & (self.tE < tStop)

        target = torch.zeros((self.height, self.width)).half().to(self.pE.device)

        if not (1 in sliceIdx):
            return target.cpu().char().numpy()

        # tSlice = self.tE[sliceIdx]
        xSlice = self.xE[sliceIdx]
        ySlice = self.yE[sliceIdx]
        pSlice = self.pE[sliceIdx]

        index = ySlice * self.width + xSlice

        target.put_(index=index, source=pSlice, accumulate=True)
        target = target.clamp(-10, 10)

        # print(target.max().cpu().item(), target.min().cpu().item())
        if reverse:
            target = -target
        return target.cpu().char().numpy()


if __name__ == '__main__':
    fileName = '/home/sensetime/research/research_vi/EventTool/dataset/1.aedat4'
    event = DVSReader(fileName)
    cv2.namedWindow('1', 0)
    subimg = event.Img[0::10]
    for img in subimg:
        cv2.imshow('1', img)
        cv2.waitKey(0)
