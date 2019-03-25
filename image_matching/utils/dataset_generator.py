import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from image_matching.utils.image_utils import (crop_image_by_center,
                                              resize_image, image_to_quadrate)


class VideoFramesGenerator:
    def __init__(self, video_source):
        self.video_source = video_source
        self.frames_count = 0
        self.video_capture = None

        self.open_video_source()

    def open_video_source(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_capture = cv2.VideoCapture(self.video_source)

        if self.video_capture is None:
            assert ValueError('Can\'t open video: {}'.format(self.video_source))

        self.video_capture.set(cv2.CAP_PROP_FOURCC, fourcc)

    def __len__(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))

    def get_resolution(self):
        return (
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )

    def get_next_frame(self):
        ret, frame = self.video_capture.read()

        frame_type = 'next'

        if not ret:
            self.video_capture.release()
            self.open_video_source()
            ret, frame = self.video_capture.read()
            if not ret:
                raise ValueError(
                    'Cant\'t read frame from reopen video source: {}'.format(
                        self.video_source
                    )
                )
            frame_type = 'new'

        return {
            'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            'type': frame_type
        }

    def __getitem__(self, idx):
        pass

    def __del__(self):
        self.video_capture.release()


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


def load_image(path):
    img = cv2.imread(path, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class TransformFramesDataset(Dataset):
    def __init__(self, images_path, shape,
                 transform_rect_size, transform_deviate):
        self.images_pathes = [
            images_path + '/' + image_name
            for image_name in os.listdir(images_path)
        ]
        self.shape = shape

        self.rect_size = transform_rect_size
        self.rect_dev = transform_deviate

    def generate_transform_matrix(self):
        rect = get_rectanle_points(self.rect_size)
        rect_dev = transform_rectanle(rect, self.rect_dev)
        return cv2.getPerspectiveTransform(rect, rect_dev)

    def __len__(self):
        return len(self.images_pathes)

    def process_to_tensor(self, img):
        t_img = image_to_quadrate(img, self.shape)
        return torch.FloatTensor(t_img).permute(2, 0, 1) / 255.0

    def __getitem__(self, idx):
        transform_matrix = self.generate_transform_matrix()

        i = random.randint(0, len(self.images_pathes) - 1)
        img = load_image(self.images_pathes[i])
        w, h, _ = img.shape
        transform_img = cv2.warpPerspective(
            img,
            transform_matrix,
            (h, w),
            borderMode=cv2.BORDER_REFLECT
        )

        return (
            self.process_to_tensor(img),
            self.process_to_tensor(transform_img),
            torch.FloatTensor(transform_matrix)
        )


class TransformFramesDatasetByVideo(Dataset):
    def __init__(self, video_generator, shape,
                 transform_rect_size, transform_deviate):
        self.video = video_generator
        self.shape = shape

        self.rect_size = transform_rect_size
        self.rect_dev = transform_deviate

    def generate_transform_matrix(self):
        rect = get_rectanle_points(self.rect_size)
        rect_dev = transform_rectanle(rect, self.rect_dev)
        return cv2.getPerspectiveTransform(rect, rect_dev)

    def __len__(self):
        return 100000

    def process_to_tensor(self, img):
        t_img = image_to_quadrate(img, self.shape)
        return torch.FloatTensor(t_img).permute(2, 0, 1) / 255.0

    def __getitem__(self, idx):
        transform_matrix = self.generate_transform_matrix()

        img = self.video.get_next_frame()['frame']
        w, h, _ = img.shape
        transform_img = cv2.warpPerspective(
            img,
            transform_matrix,
            (h, w),
            borderMode=cv2.BORDER_REFLECT
        )

        transform_matrix = np.linalg.inv(transform_matrix)

        return (
            self.process_to_tensor(img),
            self.process_to_tensor(transform_img),
            torch.FloatTensor(transform_matrix)
        )




