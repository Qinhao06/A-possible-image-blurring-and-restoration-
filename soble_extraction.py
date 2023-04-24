import cv2
import numpy as np




if __name__ == "__main__":
    print("处理开始")
    img = cv2.imread("./Project1_1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("original image", img)
    print("关闭图像进行处理")
    cv2.waitKey()
    result = sobel(img)
    cv2.imwrite('./result_sobel_extraction.jpg', result)
    cv2.imshow("result", result)
    cv2.waitKey()


