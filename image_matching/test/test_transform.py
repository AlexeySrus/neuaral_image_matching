import numpy as np
import cv2

img = cv2.imread('../../data/images/night_series/7.png', 1)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image2', cv2.WINDOW_NORMAL)

def get_rectanle_points(size):
    return np.array([
        [size, size],
        [2*size, size],
        [2*size, 2*size],
        [size, 2*size]
    ]).astype('float32')


def transform_rectanle(rect, max_shift):
    return np.array([
        [coord + np.random.randint(-max_shift, max_shift) for coord in point]
        for point in rect
    ]).astype('float32')


gen_rect = get_rectanle_points(1000)
print(gen_rect)
transform_rect = transform_rectanle(gen_rect, 50)
print(transform_rect)

kernel = cv2.getPerspectiveTransform(gen_rect, transform_rect)

print(kernel)

w, h, _ = img.shape
t_img = cv2.warpPerspective(img, kernel, (h, w), borderMode=cv2.BORDER_REFLECT)

inv_kernel = np.linalg.inv(kernel)
print(inv_kernel)

inv_t_img = cv2.warpPerspective(
    t_img, inv_kernel, (h, w), borderMode=cv2.BORDER_REFLECT
)

cv2.imshow('Image', t_img)
cv2.imshow('Image2', inv_t_img)

cv2.waitKey(0)
cv2.destroyAllWindows()