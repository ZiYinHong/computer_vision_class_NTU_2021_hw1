import numpy as np
import cv2
import argparse
from HCD import Harris_corner_detector


def main():
    parser = argparse.ArgumentParser(description='main function of Harris corner detector')
    parser.add_argument('--threshold', default=100., type=float, help='threshold value to determine corner')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    ### TODO ###
    # create HCD class
    HCD = Harris_corner_detector(args.threshold)
    response = HCD.detect_harris_corners(img_gray)
    result = HCD.post_processing(response)

    # draw the result
    for x, y in result:
        result_img = cv2.circle(img, (y, x), 1, (0, 0,255), 1)

    cv2.imshow("result_img", result_img)
    if (cv2.waitKey(0) & 0xFF == 27):
        cv2.destroyAllWindows()
    cv2.imwrite("result_img.png", result_img)

if __name__ == '__main__':
    main()