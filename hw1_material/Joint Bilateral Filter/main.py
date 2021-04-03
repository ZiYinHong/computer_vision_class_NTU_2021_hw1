import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
import time
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ### TODO ###
    # create R,G,B list and read sigma_s , sigma_r
    R = []
    G = []
    B = []
    with open(args.setting_path) as f:
        for line in f.readlines()[1:]:
            s = line.split(',')
            if 'sigma_s' in line:
                sigma_s = int(s[1])
                sigma_r = float(s[3])
            else:
                R.append(float(s[0]))
                G.append(float(s[1]))
                B.append(float(s[2]))



    # create 1~5 guidances
    h, w = img_gray.shape
    guidance = np.zeros([h, w, len(R)+1])
    for i in range(len(R)):
        guidance[:, :, i] = (R[i] * img_rgb[:, :, 0] + G[i] * img_rgb[:, :, 1] + B[i] * img_rgb[:, :, 2])
    guidance[:, :, i+1] = img_gray
    # img_gray dtype : uint8 , guidance dtype : float64



    # # plot guidances
    # guidances_count = guidance.shape[-1]
    # plt.figure(num=' guidances ', figsize=(2, 3))
    # for i in range(guidances_count):
    #     plt.subplot(2, 3, i + 1)
    #     if i == guidances_count - 1:
    #         plt.title("img_gray")
    #     else:
    #         plt.title("guidance" + str(i + 1))
    #     plt.imshow(guidance[:, :, i], cmap='gray')
    # plt.show()



    # create JBF class
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)

    # implement bilateral_filter : gt
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)

    # implement joint_bilateral_filter on guidence 1~6
    guidances_count = guidance.shape[-1]
    guidance_jbf_out = np.zeros([h, w, 3, guidances_count])
    t0 = time.time()
    print("loading....")
    for i in range(guidances_count):
        guidance_jbf_out[:, :, :, i] = JBF.joint_bilateral_filter(img_rgb, guidance[:, :, i]).astype(np.uint8)
    print('[Time] %.4f sec' % (time.time() - t0))



    # # plot joint_bilateral_filter result
    # guidance_jbf_out_counts = guidance_jbf_out.shape[-1]  # guidance_jbf_out_counts = guidances_count
    # plt.figure(num=' jbf ', figsize=(2, 3))
    # for i in range(guidance_jbf_out_counts):
    #     plt.subplot(2, 3, i + 1)
    #     if i == guidance_jbf_out_counts - 1:
    #         plt.title("img_gray_jbf_out")
    #     else:
    #         plt.title("guidance" + str(i + 1) + "_jbf_out")
    #     plt.imshow(guidance_jbf_out[:, :, :, i].astype(np.uint8))
    # plt.show()



    # Compute L1-norm with gt
    guidance_jbf_out_counts = guidance_jbf_out.shape[-1]
    L1_norm = []
    for i in range(guidance_jbf_out_counts):
        L1_norm.append(np.sum(np.abs(bf_out.astype('int32') - guidance_jbf_out[:, :, :, i].astype('int32'))))
        # bf_out.dtype = uint8,  guidance_jbf_out = float64
    print("L1_norm = ", L1_norm)



    # Compute the highest and lowest cost guidance
    max_index = L1_norm.index(max(L1_norm))
    min_index = L1_norm.index(min(L1_norm))



    # plot the result
    plt.subplot(2, 3, 1)
    plt.title('bf_out')
    plt.imshow(bf_out.astype(np.uint8))

    plt.subplot(2, 3, 2)
    plt.title('highest cost guidance image_ guidance'+ str(max_index+1))
    plt.imshow(guidance[:, :, max_index].astype(np.uint8), cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('highest cost JBF image')
    plt.imshow(guidance_jbf_out[:, :, :, max_index].astype(np.uint8))

    plt.subplot(2, 3, 4)
    plt.title('lowest cost guidance image_ guidance'+ str(min_index+1))
    plt.imshow(guidance[:, :, min_index].astype(np.uint8), cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title('lowest cost JBF image')
    plt.imshow(guidance_jbf_out[:, :, :, min_index].astype(np.uint8))

    plt.show()


if __name__ == '__main__':
    main()
