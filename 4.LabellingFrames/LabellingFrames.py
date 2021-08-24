import dlib
from facenet_pytorch import MTCNN
import torch
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os
import threading
import time
import random
import shutil


def do_recognize_person(all_face_img):
    try:
        all_verified_identity = []
        df = DeepFace.find(all_face_img,
                                  db_path="A:/PYTHON/FinalBachelorProject/FaceDataBase",
                                  enforce_detection=False,
                                  detector_backend='retinaface',
                                  # detector_backend='mtcnn',
                                  model_name='ArcFace',
                                  model=model,
                                  distance_metric="euclidean_l2")
        print(df)
        print()
        # filtered_faced = df[df['ArcFace_euclidean_l2'] <= 1]

        closest_identity = df.loc[df['ArcFace_euclidean_l2'].idxmin()]
        print(closest_identity['identity'])
        verified_identity = closest_identity['identity'].split('\\')[-1].split('/')[0]
        print()
        print(verified_identity)
        return verified_identity

    except (ValueError, cv2.error):
        print('No Face Detected')
        return None


def face_recognition(addr, n=5):
    all_img = os.listdir(addr)
    selected_img = random.sample(all_img, min(n, len(all_img)))
    predictions = []
    final_pred = None
    print(addr)
    for s in selected_img:
        face_img_addr = os.path.join(addr, s)
        # face_img = cv2.imread(face_img_addr)
        p = do_recognize_person(face_img_addr)
        if p is not None:
            predictions.append(p)
    print(predictions, len(predictions))

    print()
    print(len(predictions), max(predictions, key=predictions.count))
    if len(predictions) > 0:
        final_pred = max(predictions, key=predictions.count)
    if final_pred is not None and n / 2 < predictions.count(final_pred):
        return final_pred
    else:
        return None


if __name__ == '__main__':
    print('dlib version:', dlib.__version__)

    model = DeepFace.build_model('ArcFace')
    # print(dlib.cuda.get_device())
    # print('dlib using cuda:', dlib.DLIB_USE_CUDA)

    base_addr = 'A:\\PYTHON\\FinalBachelorProject\\MTCNNFrames'

    num_of_sources = len(os.listdir(base_addr))
    source_digits = len(str(num_of_sources))

    selected_channels = ['1']
    selected_videos = [1]

    for root, dirs, files in os.walk(base_addr):
        root_list = root.split('\\')
        if root_list[-1] == 'unlabeled' and int(root_list[-2]) in selected_videos and \
                root_list[-3] in selected_channels:
            # print(root, dirs)
            for f in dirs:
                file_addr = os.path.join(root, f)
                # print(file_addr, len(os.listdir(file_addr)))
                person = face_recognition(file_addr, n=5)
                if person is not None:
                    dest = file_addr.replace('unlabeled', 'labeled\\{identity}'.format(identity=person))
                    shutil.move(file_addr, dest)
                    print(2, 'source:', file_addr, 'dest:', dest)
                    print()
