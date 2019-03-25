import numpy as np
import cv2
from image_matching.architectures.match_model import MatchModel
from image_matching.model.model import Model

img1 = cv2.imread('../../data/images/night_series/7.png', 1)
img2 = cv2.imread('../../data/images/night_series/8.png', 1)

cv2.namedWindow('Image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image2', cv2.WINDOW_NORMAL)

matcher = Model(MatchModel(), 'cpu')
matcher.load('../../data/weights/model4/model-700.trh')

# matched_image, M = matcher.predict(img1, img2, [224, 224])

# print(M)

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)


cv2.waitKey(0)
cv2.destroyAllWindows()