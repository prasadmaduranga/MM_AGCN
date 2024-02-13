import sys
sys.path.extend(['../'])

import pickle
import argparse
from tqdm import tqdm
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import cv2
import numpy as np
import os
# from data_gen.preprocess import pre_normalization


# https://arxiv.org/pdf/1604.02808.pdf, Section 3.2
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]

max_body_true = 2
max_body_kinect = 4
max_body=2
num_joint = 25
max_frame = 300

frame_rate = 30
resnet_encoder_dims = 1000



class ImageEncoder:
    def __init__(self, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            lambda img: ImageEncoder.custom_resize(img, (256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size

    @staticmethod
    def custom_resize(img, size, interpolation=Image.BILINEAR):
        return img.resize(size, interpolation)


    def encode_images(self, frames):
        imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        tensors = [self.transform(img) for img in imgs]
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            encodings = self.model(batch_tensor)
        return encodings.cpu().numpy()

encoder = ImageEncoder()

def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file):
    cap = cv2.VideoCapture(file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)
    video_encodings = np.zeros((frame_count, resnet_encoder_dims))

    frames = []
    indices = []
    for i in range(0, frame_count, max(1, step)):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            indices.append(i)
            if len(frames) == encoder.batch_size:
                encodings = encoder.encode_images(frames)
                for j, index in enumerate(indices):
                    video_encodings[index, :] = encodings[j]
                frames, indices = [], []

    if frames:
        encodings = encoder.encode_images(frames)
        for j, index in enumerate(indices):
            video_encodings[index, :] = encodings[j]

    cap.release()
    return video_encodings


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    full_sample_name_list = []
    full_sample_label_list = []

    label_file_path = '{}/{}_rgb_joint_label.pkl'.format(out_path, part)
    data_file_path = '{}/{}_data_rgb_joint.npy'.format(out_path, part)

    if os.path.exists(label_file_path):
        with open(label_file_path, 'rb') as f:
            existing_sample_name, existing_sample_label = pickle.load(f)
        full_sample_name_list.extend(existing_sample_name)
        full_sample_label_list.extend(existing_sample_label)


    for filename in sorted(os.listdir(data_path)):
        if filename.startswith('.'):
            continue
        if filename in ignored_samples:
            continue

        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])


        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()


        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)



    if os.path.exists(data_file_path):
        existing_data = np.load(data_file_path)
    else:
        existing_data = np.zeros((0, max_frame, resnet_encoder_dims), dtype=np.float32)


    fp = np.zeros((len(sample_label), max_frame,resnet_encoder_dims), dtype=np.float32)

    # Fill in the data tensor `fp` one training example a time
    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s))
        fp[i, 0:data.shape[0], :] = data

    # fp = pre_normalization(fp)
    fp = np.concatenate((existing_data, fp), axis=0) if existing_data.size else fp

    full_sample_name_list.extend(sample_name)
    full_sample_label_list.extend(sample_label)

    # Save updated data
    with open(label_file_path, 'wb') as f:
        pickle.dump((full_sample_name_list, list(full_sample_label_list)), f)
    np.save(data_file_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')

    parser.add_argument('--data_path', default='./data/NTURGB_60/raw/')
    parser.add_argument('--ignored_sample_path',default=None)
    parser.add_argument('--out_folder', default='./data/NTURGB_60/')
    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    # folder_list = ['nturgbd_rgb_s001', 'nturgbd_rgb_s002', 'nturgbd_rgb_s003', 'nturgbd_rgb_s004','nturgbd_rgb_s005',
    #                'nturgbd_rgb_s006', 'nturgbd_rgb_s007', 'nturgbd_rgb_s008', 'nturgbd_rgb_s009','nturgbd_rgb_s010',
    #                  'nturgbd_rgb_s011', 'nturgbd_rgb_s012', 'nturgbd_rgb_s013', 'nturgbd_rgb_s014','nturgbd_rgb_s015',
    #                  'nturgbd_rgb_s016', 'nturgbd_rgb_s017']
    folder_list = ['nturgbd_rgb_s001']
    arg = parser.parse_args()

    for folder in folder_list:
        folder_data_path = os.path.join(arg.data_path, folder)
        for b in benchmark:
            for p in part:
                out_path = os.path.join(arg.out_folder, b)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                print(b, p)
                gendata(
                    folder_data_path,
                    out_path,
                    arg.ignored_sample_path,
                    benchmark=b,
                    part=p)
