import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalize_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    num_of_pixels = img.size
    ratio = np.zeros(256)
    transf_map = np.zeros(256)
    processed = img.copy()
    j = 0
    for i in hist:
        if j > 0:
            ratio[j] = i / num_of_pixels + ratio[j - 1]
        else:
            ratio[j] = i / num_of_pixels
        transf_map[j] = round(ratio[j] * 255)
        j = j + 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            processed[i][j] = transf_map[img[i][j]]
    return processed



if __name__ == "__main__":
    img = cv2.imread("./Project1_2.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("project1_2.jpg", img)
    # plt.hist(img.flatten(), 256, [0, 256])

    # 均衡化
    # equ = cv2.equalizeHist(img)
    equ = equalize_hist(img)
    cv2.imshow("result", equ)
    cv2.imwrite("./result_equlizeHish.jpg",equ)
    # 分块均衡
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cll = clahe.apply(img)
    # cv2.imshow("result2", cll)

    # #### 频域图
    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum)

    plt.show()
    cv2.waitKey()
