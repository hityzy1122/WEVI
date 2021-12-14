import os
import lib.fileTool as FT
from pathlib import Path
import cv2

ffmpegPath = '/usr/bin/ffmpeg'
import numpy as np


def video2Frame(vPath: str, fdir: str, H: int = None, W: int = None):
    FT.mkPath(fdir)
    if H is None or W is None:
        os.system('{} -y -i {} -vsync 0 -qscale:v 2 {}/%07d.png'.format(ffmpegPath, vPath, fdir))
    else:
        os.system('{} -y -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%07d.jpg'.format(ffmpegPath, vPath, W, H, fdir))


def frame2Video(fdir: str, vPath: str, fps: int, H: int = None, W: int = None, ):
    if H is None or W is None:
        # os.system('{} -y -r {} -f image2 -i {}/%*.png -vcodec libx264 -crf 18 -pix_fmt yuv420p {}'
        #           .format(ffmpegPath, fps, fdir, vPath))

        os.system('{} -y -r {} -f image2 -i {}/%6d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p {}'
                  .format(ffmpegPath, fps, fdir, vPath))
    else:
        os.system('{} -y -r {} -f image2 -s {}x{} -i {}/%*.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'
                  .format(ffmpegPath, fps, W, H, fdir, vPath))


def slomo(vPath: str, dstPath: str, fps):
    os.system(
        '{} -y -r {} -i {}  -strict -2 -vcodec libx264 -c:a aac -crf 18 {}'.format(ffmpegPath, fps, vPath, dstPath))


def downFPS(vPath: str, dstPath: str, fps):
    os.system(
        '{} -i {}  -strict -2 -r {} {}'.format(ffmpegPath, vPath, fps, dstPath))


def downSample(vPath: str, dstPath: str, H, W):
    os.system(
        '{} -i {}  -strict -2 -s {}x{}  {}'.format(ffmpegPath, vPath, H, W, dstPath))


def batchVideo2Frames(videosDir, outDir):
    allVideos = FT.getAllFiles(videosDir)
    for video in allVideos:
        targetPath = str(Path(outDir) / Path(video).stem)

        video2Frame(video, targetPath)


def roteResize(videoPath, outPath):
    allsubDirs = FT.getSubDirs(videoPath)
    for subDir in allsubDirs:
        allFrames = FT.getAllFiles(subDir)
        for idx, frame in enumerate(allFrames):
            targetPath = frame.replace(videoPath, outPath)
            FT.mkPath(Path(targetPath).parent)
            if idx <= 5:
                # FT.delPath(frame)
                continue
            img: np.ndarray = cv2.imread(frame)
            H, W, C = img.shape
            if H > W:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, dsize=(640, 360))
            cv2.imwrite(targetPath, img)
            pass


if __name__ == '__main__':
    videoDir = '/home/sensetime/data/VideoInterpolation/highfps/adobe_240fps/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos'
    outDir = '/home/sensetime/data/VideoInterpolation/highfps/adobe_240fps/adobe240_frames'
    outPath = '/home/sensetime/data/VideoInterpolation/highfps/adobe_240fps/adobe240_frames_small'
    # batchVideo2Frames(videoDir, outDir)
    roteResize(outDir, outPath)
    # framePath = '/home/sensetime/data/event/outputTest/pencil2/'
    # framePath = '/home/sensetime/data/VideoInterpolation/highfps/gopro_yzy/output'
    # video = '/home/sensetime/data/event/outputvideo/pencil2.mp4'
    # video2Frame(video, framePath)

    # video = '/home/sensetime/data/VideoInterpolation/highfps/goPro_240fps/train/GOPR0372_07_00/out.mp4'
    # framePath = '/home/sensetime/data/VideoInterpolation/highfps/goPro_240fps/train/GOPR0372_07_00'
    # frame2Video(framePath, video, 60)

    # vPath = '/media/sensetime/Elements/0721   /0716_video/1.avi'
    # dstPath = '/media/sensetime/Elements/0721/0716_video/1_.mp4'
    # downFPS(vPath, dstPath, 8)
