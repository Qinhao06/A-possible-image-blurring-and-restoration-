import numpy as np
import cv2


def median_Blur_gray(img, filiter_size=3):
    image_copy = np.copy(img).astype(np.float32)
    processed = np.zeros_like(image_copy)
    mid = int(filiter_size / 2)

    for i in range(mid, image_copy.shape[0] - mid):
        for j in range(mid, image_copy.shape[1] - mid):
            temp = []
            for m in range(i - mid, i + mid + 1):
                for n in range(j - mid, j + mid + 1):
                    if m - mid < 0 or m + mid + 1 > image_copy.shape[0] or n - mid < 0 or n + mid + 1 > \
                            image_copy.shape[1]:
                        temp.append(0)
                    else:
                        temp.append(image_copy[m][n])
                    # count += 1
            temp.sort()
            processed[i][j] = temp[(int(filiter_size * filiter_size / 2) + 1)]
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    return processed


# sobel 算子
def sobel(img):
    image_copy = np.copy(img).astype(np.float32)
    sobel_image_x = np.zeros_like(image_copy)
    sobel_image_y = np.zeros_like(image_copy)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for i in range(image_copy.shape[0] - 2):
        for j in range(image_copy.shape[1] - 2):
            sobel_image_x[i, j] = np.abs(np.sum(image_copy[i:i + 3, j:j + 3] * sobel_x))
            sobel_image_y[i, j] = np.abs(np.sum(image_copy[i:i + 3, j:j + 3] * sobel_y))

    # sobel_image = np.sqrt(sobel_image_x*sobel_image_x + sobel_image_y*sobel_image_y).astype(np.int8)

    sobel_image = (sobel_image_x + sobel_image_y) * 0.5
    sobel_image = np.clip(sobel_image, 0, 255).astype(np.uint8)
    return sobel_image


if __name__ == "__main__":
    print("处理开始")
    img = cv2.imread("./Project1_1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("impulse_noise", img)
    cv2.waitKey()

    result_do_noise = median_Blur_gray(img)
    cv2.imwrite('./result_for_do_noise.jpg', result_do_noise)
    cv2.imshow("result", result_do_noise)
    cv2.waitKey()

    result_extraction = sobel(result_do_noise)
    cv2.imwrite('./result_extraction.jpg', result_extraction)
    cv2.imshow('result_extraction', result_extraction)
    cv2.waitKey()
