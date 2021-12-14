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


def checkInput(I0t, I1t, It, Et):
    cv2.namedWindow('1', 0)
    for bidx in range(I0t.size(0)):
        It_ = ((It[bidx, 0] + 1) * 127.5).cpu().byte().numpy()
        I0t_ = ((I0t[bidx, 0] + 1) * 127.5).cpu().byte().numpy()
        I1t_ = ((I1t[bidx, 0] + 1) * 127.5).cpu().byte().numpy()
        # Et_ = ((Et[bidx, 0] + 1) * 127.5).cpu().byte().numpy()
        Et_ = Et.cpu().float().numpy()
        cv2.imshow('1', It_)
        cv2.waitKey(0)

        cv2.imshow('1', I0t_)
        cv2.waitKey(0)

        cv2.imshow('1', I1t_)
        cv2.waitKey(0)

        Pos = Et_[bidx, 0::2, ...]
        Neg = Et_[bidx, 1::2, ...]

        for eIdx in range(4):
            imgCV = np.zeros((180, 240, 3), dtype=np.uint8)
            imgCV[..., 0] = It_
            imgCV[..., 1] = It_
            imgCV[..., 2] = It_

            pPos = Pos[eIdx, ...]
            pNeg = Neg[eIdx, ...]
            eventGray = pPos + pNeg
            eventGray = ((eventGray - eventGray.min()) / (eventGray.max() - eventGray.min())) * 255
            eventGray = cv2.cvtColor(eventGray, cv2.COLOR_BGR2RGB).astype(np.uint8)

            imgCV[..., 2][pPos > 0] = 255
            imgCV[..., 1][pNeg < 0] = 255

            cv2.imshow('1', np.concatenate([imgCV, eventGray], axis=1))
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def checkEvents(imgs: torch.Tensor, events: torch.Tensor):
    # N, F, C, H, W = img.shape
    N, C, H, W = events.shape

    Pos = events[:, 0::2, ...]
    # Neg = -events[:, 1::2, :, ...]
    Neg = -events[:, 1::2, ...]

    cv2.namedWindow('1', 0)

    for n in range(N):
        for eIdx in range(C // 2):
            pPos = Pos[n, eIdx, ...].detach().cpu().numpy()
            pNeg = Neg[n, eIdx, ...].detach().cpu().numpy()

            img = np.zeros([H, W, 3])
            imgSample = imgs[n, 0, ...].detach().cpu().numpy()
            imgSample = ((imgSample - imgSample.min()) / (imgSample.max() - imgSample.min()) * 255).astype(np.uint8)

            img[:, :, -1] = imgSample
            img[:, :, 1] = imgSample
            img[:, :, 0] = imgSample

            img[:, :, -1][pPos > 0] = 255
            img[:, :, 1][pNeg > 0] = 255

            # cv2.putText(img, '{}_{}'.format(fIdx, eIdx), (20, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, '{}_{}'.format(n, eIdx), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('1', img.astype(np.uint8))
            cv2.waitKey(100)

        pPos = Pos[n, ...].sum(dim=0).detach().cpu().numpy()
        pNeg = Neg[n, ...].sum(dim=0).detach().cpu().numpy()

        img = np.zeros([H, W, 3])
        imgSample = imgs[n, 0, ...].detach().cpu().numpy()
        imgSample = ((imgSample - imgSample.min()) / (imgSample.max() - imgSample.min()) * 255).astype(np.uint8)

        img[:, :, -1] = imgSample
        img[:, :, 1] = imgSample
        img[:, :, 0] = imgSample

        img[:, :, -1][pPos > 0] = 255
        img[:, :, 1][pNeg > 0] = 255

        # cv2.putText(img, '{}_{}'.format(fIdx, eIdx), (20, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, 'final_{}'.format(n), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('1', img.astype(np.uint8))
        cv2.waitKey(100)
