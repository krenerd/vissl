from vissl.data.data_helper import get_mean_image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import pydicom as dicom
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img=dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data=(data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

def load_df():
    TRAIN_DF = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv"
    VERTEBRAE_PRED = "/kaggle/input/rsna2022individualsegmap/train_segmented_hardlabel.csv"
    df = pd.read_csv(TRAIN_DF)
    df_train_slices = pd.read_csv(VERTEBRAE_PRED)

    df = df_train_slices.set_index('StudyInstanceUID').join(df.set_index('StudyInstanceUID'),
                                                          rsuffix='_fracture').reset_index().copy()

    # remove `1.2.826.0.1.3680043.20574` (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344862)
    df = df.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)
    return df


class VisslFractureDataset(Dataset):
    """
    add documentation on how this dataset works
    Args:
        add docstrings for the parameters
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(VisslFractureDataset, self).__init__()
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "fracture"
        ], "data_source must be either disk_filelist or disk_folder or my_data_source"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        # implement anything that data source init should do

        self.img_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images"
        self.get_image = True
        if split == "TRAIN":
            self.df = load_df()
            self._num_samples = len(self.df) # set the length of the dataset
        else:
            print(split)
            input()
            self._num_samples = 0

    def _get_image(self, uid, slice_idx):
        path = os.path.join(self.img_path, uid, f'{slice_idx}.dcm')
        try:
            img = load_dicom(path)[0]
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            return Image.fromarray(img.astype('uint8'), 'RGB')
        except Exception as ex:
            print(ex)
            input()
            return None

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx: int):
        """
        implement how to load the data corresponding to idx element in the dataset
        from your data source
        """
        uid = self.df.iloc[idx].StudyInstanceUID
        slice_idx = self.df.iloc[idx].Slice

        img = self._get_image(uid, slice_idx)
        if img is None:
            return None, False

        # is_success should be True or False indicating whether loading data was successful or failed
        # loaded data should be Image.Image if image data
        return img, True
