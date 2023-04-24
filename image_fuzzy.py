import cv2
import numpy as np


def process_image(imag, input_number, is_color):
    image = np.copy(imag).astype(np.float32)
    shape = image.shape
    shape3 = 1
    if is_color == 1:
        shape3 = 3
    cnt_input = 0
    for i in range(shape3):
        for j in range(shape[0]):
            for m in range(shape[1] - 1):
                image[j][m + 1][i] = image[j][m + 1][i] - input_number[cnt_input] * 0.1 * (
                            image[j][m + 1][i] - image[j][m][i])
                cnt_input = cnt_input + 1
                if cnt_input >= len(input_number):
                    cnt_input = 0
    return image, image.astype(np.uint8)


def reverse_img(imag, input_number, is_color):
    image = np.copy(imag).astype(np.float32)
    image_copy = np.copy(imag).astype(np.float32)
    shape = image.shape
    shape3 = 1
    if is_color == 1:
        shape3 = 3
    cnt_input = 0
    for i in range(shape3):
        for j in range(shape[0]):
            for m in range(shape[1] - 1):
                image[j][m + 1][i] = (image_copy[j][m + 1][i] - 0.1 * input_num[cnt_input] * image_copy[j][m][i]) / (
                            1 - 0.1 * input_num[cnt_input])
                cnt_input = cnt_input + 1
                if cnt_input >= len(input_number):
                    cnt_input = 0
    print(image)
    return image, image.astype(np.uint8)


if __name__ == '__main__':
    print('process on:')
    img = cv2.imread('./original_picture.png', cv2.IMREAD_COLOR)
    cv2.imshow('original image', img)
    print('输入数组为：')
    input_str = '345678'
    input_str = list(input_str)
    input_num = [int(x) for x in input_str]
    print(input_num)
    result_float, result_int = process_image(img, input_number=input_num, is_color=1)
    print('图片形状为：', img.shape)
    cv2.imshow('processed_fuzzy_result', result_int)
    cv2.imwrite('./processed_fuzzy_result.jpg', result_int)
    reverse_float, reverse_int = reverse_img(result_float, input_number=input_num, is_color=1)
    cv2.imshow('reverse_fuzzy', reverse_int)
    cv2.imwrite('./reverse_fuzzy.jpg', reverse_int)
    cv2.waitKey()
