import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros_like(img)
        r = int((self.wndw_size - 1) / 2)


        # range kernel
        scaleFactor_r = -1 / (2 * self.sigma_r * self.sigma_r)
        LUTr = np.exp(np.linspace(0, 1, 256) ** 2 * scaleFactor_r)  # all possible value


        # spatial kernel
        scaleFactor_s = -1 / (2 * self.sigma_s * self.sigma_s)
        x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        LUTs = np.exp((x * x + y * y) * scaleFactor_s)


        # # plot spatial kernel
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.plot_wireframe(x, y, LUTs)
        # plt.show()

        # implement joint_bilateral_filter
        rows, cols, _ = img.shape
        for row in range(self.pad_w, self.pad_w + rows):
            for col in range(self.pad_w, self.pad_w + cols):
                # for r in self.wndw_size**2:
                if padded_guidance.ndim == 3:
                    Wsr = LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1), 0] - padded_guidance[row, col, 0])] * \
                          LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1), 1] - padded_guidance[row, col, 1])] * \
                          LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1), 2] - padded_guidance[row, col, 2])] * \
                          LUTs
                else:
                    Wsr = LUTr[abs(padded_guidance[(row - r):(row + r + 1), (col - r):(col + r + 1)] - padded_guidance[row, col])] * LUTs

                # Wsr shape   (19,19)
                # padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 0] shape   (19,19)
                sum_Wsr = np.sum(Wsr)
                output[row-self.pad_w, col-self.pad_w, 0] = np.sum(Wsr * padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 0]) / sum_Wsr
                output[row-self.pad_w, col-self.pad_w, 1] = np.sum(Wsr * padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 1]) / sum_Wsr
                output[row-self.pad_w, col-self.pad_w, 2] = np.sum(Wsr * padded_img[(row - r): (row + r + 1), (col - r):(col + r + 1), 2]) / sum_Wsr

        return np.clip(output, 0, 255).astype(np.uint8)