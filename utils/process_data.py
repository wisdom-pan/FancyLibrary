import hashlib
from PIL import Image
import torch
import numpy as np
import cv2

class Utils:
    @staticmethod
    def get_image_md5(img: Image.Image):
        img_byte_array = img.tobytes()
        hash_md5 = hashlib.md5()
        hash_md5.update(img_byte_array)
        hex_digest = hash_md5.hexdigest()
        return hex_digest

    @staticmethod
    def calculate_md5_from_binary(binary_data):
        hash_md5 = hashlib.md5()
        hash_md5.update(binary_data)
        return hash_md5.hexdigest()

    @staticmethod
    def save_image(image: Image.Image, path: str):
        image.save(path)

    @staticmethod
    def load_image(path: str):
        return Image.open(path).convert('RGB')

    @staticmethod
    def extract_frames_from_video(video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(image)
        cap.release()
        return frames

    @staticmethod
    def save_numpy_array(array, path: str):
        np.save(path, array)

    @staticmethod
    def load_numpy_array(path: str):
        return np.load(path)

    @staticmethod
    def calculate_similarity(query_rep, doc_reps):
        doc_reps_cat = torch.stack([torch.Tensor(i) for i in doc_reps], dim=0)
        similarities = torch.matmul(query_rep, doc_reps_cat.T)
        return similarities
