# only use for python2

from __future__ import division
from __future__ import print_function
from glob import glob
import numpy as np
import os.path as op
import struct
import copy
import cv2
import os
from tqdm import tqdm
import aedat

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loadaerdat(datafile='.aedat', length=0):
    aeLen = 8
    readMode = '>II'
    td = 0.000001

    sizeX = 240
    sizeY = 180
    x0 = 1
    y0 = 1
    xmask = 0x003ff000
    xshift = 12
    ymask = 0x7fc00000
    yshift = 22
    pmask = 0x800
    pshift = 11
    eventtypeshift = 31
    adcmask = 0x3ff

    frame = np.zeros([6, sizeX, sizeY], dtype=np.int32)

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    print('file size', length)

    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == '#':
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []
    frames = []
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen
    pbar = tqdm(total=length)
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        eventtype = (addr >> eventtypeshift)

        # parse event's data
        if eventtype == EVT_DVS:  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift

            timestamps.append(ts)
            xaddr.append(sizeX - x_addr)
            yaddr.append(sizeY - y_addr)
            pol.append(a_pol)

        if eventtype == EVT_APS:  # this is an APS packet
            x1 = sizeX
            y1 = sizeY

            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            adc_data = addr & adcmask
            read_reset = (addr >> 10) & 3

            if x_addr >= x0 and x_addr < x1 and y_addr >= y0 and y_addr < y1:
                if (read_reset == 0):  # is reset read
                    frame[0, x_addr, y_addr] = adc_data
                    frame[4, x_addr, y_addr] = ts  # resetTsBuffer;
                if (read_reset == 1):  # is read signal
                    # print "read", read_reset
                    frame[1, x_addr, y_addr] = adc_data
                    frame[3, x_addr, y_addr] = ts  # readTsBuffer;

            if (read_reset == 0) and x_addr == 0 and y_addr == 0:
                frame[2, :, :] = frame[0, :, :] - frame[1, :, :]
                frame[5, :, :] = frame[3, :, :] - frame[4, :, :]
                frames.append(frame)
                frame = np.zeros([6, sizeX, sizeY], dtype=np.int32)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen
        pbar.update(aeLen)
    pbar.close()

    try:
        print('read %i (~ %.2fM) AE events, duration= %.2fs' % (
            len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
        n = 5
        print('showing first %i:' % (n))
        print('timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s' % (
            timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
    except:
        print('failed to print statistics')

    return timestamps, xaddr, yaddr, pol, frames


def save_data(timestamps, xaddr, yaddr, pol, frames, save_frame=True, save_event=True, save_path=''):
    if save_frame:
        start = timestamps[0]
        f = open(op.join(save_path, 'images.txt'), 'w')
        mkdir(os.path.join(save_path, 'images'))

        for idx, frame in enumerate(frames[1:]):
            img_path = op.join(save_path, 'images/%06d.png' % (idx + 1))
            img = frame[2]
            img[img < 0] = 0
            img = (np.rot90(img, 1) / np.power(2, 10) * 255.0).astype('uint8')
            time = (np.max(frame[3]) - start) * 1e-6

            # put the writing as the end
            cv2.imwrite(img_path, img)
            f.write('%.6f' % time + ' ' + 'images/%06d.png' % (idx + 1) + '\n')
        f.close()

    if save_event:
        f = open(op.join(save_path, 'events.txt'), 'w')
        start = timestamps[0]
        timestamps = ['%.6f' % ((x - start) * 1e-6) for x in timestamps]
        xaddr = np.array(xaddr) - 1
        yaddr = np.array(yaddr) - 1
        for line in zip(timestamps, xaddr.tolist(), yaddr.tolist(), pol):
            f.write(' '.join([str(item) for item in line]) + '\n')
        f.close()


if __name__ == '__main__':
    # filename = 'DAVIS240C-2020-08-18T11-46-15 0800-00000000-0.aedat'
    # save_data(*loadaerdat(filename))
    import aer

    # read all at once
    events = aer.AEData("DAVIS240C-2020-08-18T11-46-15 0800-00000000-0.aedat")
    pass
