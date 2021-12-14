import lib.imgTool_ as trans
import cv2
import imageio
from pathlib import Path
from lib.fileTool import mkPath
import numpy as np
import torch


def visualContinuousFrames(imgrendListOut, gtframeTList, path):
    mkPath(path)
    totalList = []
    for t, (imgrendOut, gtframeT) in enumerate(zip(imgrendListOut, gtframeTList)):
        imgrendOut = trans.ToCVImage(imgrendOut)

        gtframeT = trans.ToCVImage(gtframeT)

        imgrendOut = cv2.putText(imgrendOut, "rdOut{}".format(t + 1), (10, 20),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        gtframeT = cv2.putText(gtframeT, "gt{}".format(t + 1), (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        imgcat = trans.makeGrid([gtframeT, imgrendOut], shape=[1, 2])
        totalList.append(imgcat)

    imageio.mimsave(str(Path(path) / 'Input.gif'), totalList, 'GIF', duration=0.5)


def checkGrad(net):
    for parem in list(net.named_parameters()):
        if parem[1].grad is not None:
            print(parem[0] + ' \t shape={}, \t mean={}, \t std={}\n'.format(parem[1].shape,
                                                                            parem[1].grad.abs().mean().cpu().item(),
                                                                            parem[1].grad.abs().std().cpu().item()))


def checkImgList(imgList):
    cv2.namedWindow('1', 0)
    N, C, H, W = imgList[0].shape
    for n in N:
        for imgn in imgList:
            img: torch.Tensor = (imgn[n] - imgn[n].min()) / (imgn[n].max() - imgn[n].min())
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            cv2.imshow('1', img)
            cv2.waitKey(0)


def checkIE(I0t: torch.Tensor, I1t: torch.Tensor, It: torch.Tensor, Et: torch.Tensor):
    # N, F, C, H, W = img.shape
    N, C, H, W = Et.shape

    cv2.namedWindow('1', 0)

    for n in range(N):
        img0t = ((I0t[n].float() - I0t[n].min()) / (I0t[n].max() - I0t[n].min()) * 255).cpu().numpy().transpose(
            [1, 2, 0]).astype(np.uint8)

        img1t = ((I1t[n].float() - I1t[n].min()) / (I1t[n].max() - I1t[n].min()) * 255).cpu().numpy().transpose(
            [1, 2, 0]).astype(np.uint8)
        I = ((It[n].float() - It[n].min()) / (It[n].max() - It[n].min()) * 255).cpu().numpy().transpose(
            [1, 2, 0]).astype(np.uint8)
        cv2.imshow('1', np.concatenate([img0t, img1t], axis=1))
        cv2.waitKey(100)
        cv2.imshow('1', np.concatenate([I, I], axis=1))
        cv2.waitKey(100)

        E = Et[n].cpu().numpy().astype(np.float32)

        for eIdx, p in enumerate(E):
            eventImg = p
            eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
                np.uint8)
            eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

            img = I.copy()

            img[:, :, 0][p != 0] = 0

            img[:, :, 2][p > 0] = 255
            img[:, :, 1][p < 0] = 255

            cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255),
                        5,
                        cv2.LINE_AA)

            cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
            cv2.waitKey(100)


def checkValInput(IV, ET):
    N, C, H, W = IV[0].shape
    for n in range(N):
        for Iv in IV:
            I = ((Iv[n].float() - Iv[n].min()) / (Iv[n].max() - Iv[n].min()) * 255).cpu().numpy().transpose(
                [1, 2, 0]).astype(np.uint8)
            cv2.imshow('1', I)
            cv2.waitKey(0)

        E = [i[n].cpu().numpy().astype(np.float32) for i in ET]
        for It, Et in zip([IV[1], IV[1], IV[1]], E):
            I = ((It[n].float() - It[n].min()) / (It[n].max() - It[n].min()) * 255).cpu().numpy().transpose(
                [1, 2, 0]).astype(np.uint8)
            for eIdx, p in enumerate(Et):
                eventImg = p
                eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(np.uint8)
                eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

                img = I.copy()

                img[:, :, 0][p != 0] = 0

                img[:, :, 2][p > 0] = 255
                img[:, :, 1][p < 0] = 255

                cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            5,
                            cv2.LINE_AA)

                cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
                cv2.waitKey(100)


def checkTrainInput(I0ts, I1ts, Its, Ets):
    cv2.namedWindow('1', 0)
    N, C, H, W = I0ts.shape
    for n in range(N):
        I0t = ((I0ts[n] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
        I1t = ((I1ts[n] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)
        It = ((Its[n] + 1) * 127.5).cpu().numpy().transpose([1, 2, 0]).astype(np.uint8)

        print('check I0t, I1t')
        cv2.imshow('1', np.concatenate([I1t, I0t], axis=1))
        cv2.waitKey(0)

        cv2.imshow('1', np.concatenate([It, It], axis=1))
        cv2.waitKey(0)

        Et = Ets[0].cpu().numpy().astype(np.float32)

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
            cv2.waitKey(100)


def checkRefineInput(blur: torch.Tensor, refL: torch.Tensor, refR: torch.Tensor):
    cv2.namedWindow('1', 0)
    blur = (blur[0]).cpu().numpy()
    refL = (refL[0]).cpu().numpy()
    refR = (refR[0]).cpu().numpy()

    for eIdx, p in enumerate(blur[3::, ...]):
        eventImg = p
        eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
            np.uint8)
        eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

        img = ((blur[0:3, ...] + 1) * 127.5).copy().transpose([1, 2, 0]).astype(np.uint8)

        img[:, :, 0][p != 0] = 0

        img[:, :, 2][p > 0] = 255
        img[:, :, 1][p < 0] = 255

        cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA)

        cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
        cv2.waitKey(0)

    for eIdx, p in enumerate(refL[3::, ...]):
        eventImg = p
        eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
            np.uint8)
        eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

        img = ((refL[0:3, ...] + 1) * 127.5).copy().transpose([1, 2, 0]).astype(np.uint8)

        img[:, :, 0][p != 0] = 0

        img[:, :, 2][p > 0] = 255
        img[:, :, 1][p < 0] = 255

        cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA)

        cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
        cv2.waitKey(0)
    for eIdx, p in enumerate(refR[3::, ...]):
        eventImg = p
        eventImg = ((eventImg - eventImg.min()) / (eventImg.max() - eventImg.min()) * 255.0).astype(
            np.uint8)
        eventImg = cv2.cvtColor(eventImg, cv2.COLOR_GRAY2BGR)

        img = ((refR[0:3, ...] + 1) * 127.5).copy().transpose([1, 2, 0]).astype(np.uint8)

        img[:, :, 0][p != 0] = 0

        img[:, :, 2][p > 0] = 255
        img[:, :, 1][p < 0] = 255

        cv2.putText(img, '{}'.format(eIdx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA)

        cv2.imshow('1', np.concatenate([img.astype(np.uint8), eventImg], axis=1))
        cv2.waitKey(0)
