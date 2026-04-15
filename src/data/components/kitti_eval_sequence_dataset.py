"""Sliding-window raw KITTI sequences for offline eval testers (not Lightning training)."""

import glob

import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.utils.kitti_utils import read_pose_from_text


class KittiEvalSequenceDataset:
    """One KITTI drive, sliced into windows of ``opt.seq_len`` frames for model evaluation."""

    def __init__(self, opt, folder):
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len
        self.folder = folder
        self.load_data()

    def load_data(self):
        image_dir = self.data_dir + "/sequences/"
        imu_dir = self.data_dir + "/imus/"
        pose_dir = self.data_dir + "/poses/"

        self.img_paths = glob.glob("{}{}/image_2/*.png".format(image_dir, self.folder))
        self.imus = sio.loadmat("{}{}.mat".format(imu_dir, self.folder))["imu_data_interp"]
        self.poses, self.poses_rel = read_pose_from_text("{}{}.txt".format(pose_dir, self.folder))
        self.img_paths.sort()

        self.img_paths_list, self.poses_list, self.imus_list = [], [], []
        start = 0
        n_frames = len(self.img_paths)
        while start + self.seq_len < n_frames:
            self.img_paths_list.append(self.img_paths[start : start + self.seq_len])
            self.poses_list.append(self.poses_rel[start : start + self.seq_len - 1])
            self.imus_list.append(self.imus[start * 10 : (start + self.seq_len - 1) * 10 + 1])
            start += self.seq_len - 1
        self.img_paths_list.append(self.img_paths[start:])
        self.poses_list.append(self.poses_rel[start:])
        self.imus_list.append(self.imus[start * 10 :])

    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, i):
        image_path_sequence = self.img_paths_list[i]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.opt.img_h, self.opt.img_w))
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        imu_sequence = torch.FloatTensor(self.imus_list[i])
        gt_sequence = self.poses_list[i][:, :6]
        return image_sequence, imu_sequence, gt_sequence
