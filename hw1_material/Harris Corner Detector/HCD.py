import numpy as np
import cv2
import matplotlib.pyplot as plt


class Harris_corner_detector(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def detect_harris_corners(self, img, Iy=None):
        ### TODO ####
        # Step 1: Smooth the image by Gaussian kernel
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.5)
        Blur = cv2.GaussianBlur(img, (3, 3), 1.5)

        # Step 2: Calculate Ix, Iy (1st derivative of image along x and y axis)
        # - Function: cv2.filter2D (kernel = [[1.,0.,-1.]] for Ix or [[1.],[0.],[-1.]] for Iy)
        Ix = cv2.filter2D(Blur, -1, np.array([[1.,0.,-1.]]))
        Iy = cv2.filter2D(Blur, -1, np.array([[1.],[0.],[-1.]]))

        # Step 3: Compute Ixx, Ixy, Iyy (Ixx = Ix*Ix, ...)
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        # Step 4: Compute Sxx, Sxy, Syy (weighted summation of Ixx, Ixy, Iyy in neighbor pixels)
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.)
        Sxx = cv2.GaussianBlur(Ixx, (3, 3), 1.0)
        Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.0)
        Syy = cv2.GaussianBlur(Iyy, (3, 3), 1.0)

        # Step 5: Compute the det and trace of matrix M (M = [[Sxx, Sxy], [Sxy, Syy]])
        detM = Sxx * Syy - Sxy * Sxy
        traceM = Sxx + Syy
        # Step 6: Compute the response of the detector for each pixel by det/(trace+1e-12)
        response = detM / (traceM+1e-12)

        return response


    def post_processing(self, response):
        ### TODO ###
        # Step 1: Thresholding
        ZPresponse = cv2.copyMakeBorder(response, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

        # w, h = response.shape
        # candidates = []
        # for row in range(w):
        #     for col in range(h):
        #         if ZPresponse[row][col] > self.threshold:
        #             candidates.append([row, col])
        candidates_row = np.where(ZPresponse > self.threshold)[0]   # get row index of which pixel's value > threshold
        candidates_col = np.where(ZPresponse > self.threshold)[1]   # get col index of which pixel's value > threshold
        candidates = list(map(list, zip(candidates_row, candidates_col)))  # zip each point and convert to list of list

        # Step 2: Find local maximum
        local_max = []
        for row, col in candidates:
            center = ZPresponse[row,col]
            window = ZPresponse[(row-2):(row+3), (col-2):(col+3)]
            if window.max() == center:
                local_max.append([row-2, col-2])

        return local_max
