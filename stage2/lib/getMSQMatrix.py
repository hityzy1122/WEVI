import numpy as np
import pickle


def getGaussianW(shape: tuple):
    sigma = 0.5
    H, W = shape
    x = np.linspace(0, H - 1, H).astype(np.float32)
    y = np.linspace(0, W - 1, W).astype(np.float32)
    xv, yv = np.meshgrid(x, y)
    xv: np.ndarray = xv - H // 2
    yv: np.ndarray = yv - W // 2
    WeightMatrix: np.ndarray = np.exp(-(np.square(xv) + np.square(yv)) / (2 * np.square(sigma))) / np.square(
        sigma * np.sqrt(np.pi * 2))

    # WeightMatrix /= WeightMatrix.sum()

    return np.diag(WeightMatrix.flatten())


def getArray():
    P: np.ndarray = np.array([[0.5, 0.5, -1, -1, 1],
                              [0, 0.5, 0, -1, 1],
                              [0.5, 0.5, 1, -1, 1],
                              [0.5, 0, -1, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0.5, 0, 1, 0, 1],
                              [0.5, 0.5, -1, 1, 1],
                              [0, 0.5, 0, 1, 1],
                              [0.5, 0.5, 1, 1, 1]]).astype(np.float32)
    W = getGaussianW((3, 3))
    C = np.linalg.inv(P.T @ W @ P) @ P.T @ W
    np.save('matrixC.npy', C)
    pass


if __name__ == '__main__':
    getArray()
