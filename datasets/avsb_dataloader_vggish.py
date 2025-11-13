import os
from wave import _wave_params
import torch
import torch.nn as nn

import warnings
warnings.simplefilter("ignore", UserWarning)

from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as audio_T
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('..')
# from configs.avsb_config import cfg
from torchvggish import vggish_input

import pdb
import ipdb
from tqdm import tqdm
import json


def crop_resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not img_is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    # print("crop for train. set")
    return img

def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

def load_mask_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

def load_image_in_PIL_to_Tensor(path, split='train', mode='RGB', crop_size=1024, transform=None, crop_img_and_mask=True, transform_initial=None):
    img_PIL = Image.open(path).convert(mode)
    img_PIL = resize_img(crop_size,
                                 img_PIL, img_is_mask=False)
    if transform:
        img_tensor = transform(img_PIL)
        img_initial_tensor = transform_initial(img_PIL)
        return img_tensor, img_initial_tensor
    return img_PIL

def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB', crop_size=1024, crop_img_and_mask=True):
    color_mask_PIL = Image.open(path).convert(mode)
    color_mask_PIL = resize_img(
                crop_size, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label)  # [H, W]
    color_label = color_label.unsqueeze(0)
    # binary_mask = (color_label != (cfg_avs.NUM_CLASSES-1)).float()
    # return color_label, binary_mask # both [1, H, W]
    return color_label  # both [1, H, W]

def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete

class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train', args=None):
        super(S4Dataset, self).__init__()
        self.split = split
        self.args = args
        if self.split == 'train':
            self.mask_num = 1
        else:
            self.mask_num = 5 # if self.split == 'train' else 5
        
        df_all = pd.read_csv(args.config['AVSSBench']['meta_csv_path'], sep=',')
        self.df_split = df_all[df_all['split'] == split]
        self.df_split = self.df_split   #[:20]

        if args.local_rank == 0:
            print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        
        if (self.split == 'train'):
            df_all = pd.read_csv(args.config['AVSBenchS4']['ANNO_CSV_SHUFFLE'], sep=',')
            self.df_split = df_all[df_all['split'] == split]
            len_set = (len(self.df_split)) * args.trainset_ratio
            len_set = int(len_set)
            self.df_split = self.df_split
            self.df_split = self.df_split[df_all['name'] != 'vhUM-UOKpSk'] #[:2]vhUM-UOKpSk_4000_9000

        else:
            df_all = pd.read_csv(args.config['AVSBenchS4']['ANNO_CSV'], sep=',')
            self.df_split = df_all[df_all['split'] == split]
            self.df_split = self.df_split  #[:2]

        if args.local_rank == 0:
            print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.args.inp_size, self.args.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.args.inp_size, self.args.inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.initial_transform = transforms.Compose([
            transforms.Resize((self.args.inp_size, self.args.inp_size)),
            transforms.ToTensor(),
        ])

        self.AmplitudeToDB = audio_T.AmplitudeToDB()
        self.crop_size = args.config['AVSBenchS4']['crop_size']
        self.crop_img_and_mask = args.config['AVSBenchS4']['crop_img_and_mask']
        self.v2_pallete = get_v2_pallete(
            args.config['AVSSBench']['label_idx_path'], num_cls=args.config['AVSSBench']['num_class'])

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(self.args.config['AVSBenchS4']['DIR_IMG'], self.split, category, video_name)
        mask_base_path = os.path.join(self.args.config['AVSBenchS4']['DIR_MASK'])
        all_video_names = os.listdir(mask_base_path)
        video_name_for_class = [video for video in all_video_names if video.startswith(video_name)]
        color_mask_base_path = os.path.join(mask_base_path, video_name_for_class[0], 'labels_rgb')

        audio_lm_path = os.path.join(self.args.config['AVSBenchS4']['DIR_AUDIO_LOG_MEL'], self.split, category, video_name + '.pkl')
        audio_log_mel = load_audio_lm(audio_lm_path) 
        
        imgs, initial_imgs = [], []
        for img_id in range(1, 1+self.mask_num):
            img, initial_img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform, transform_initial = self.initial_transform)
            imgs.append(img)
            initial_imgs.append(initial_img)
        
        audio_path = os.path.join(
            mask_base_path, video_name_for_class[0], 'audio.wav')

        labels = []
        mask_path_list = sorted(os.listdir(color_mask_base_path))
        for mask_path in mask_path_list:
            if not mask_path.endswith(".png"):
                mask_path_list.remove(mask_path)
        mask_num = len(mask_path_list)
        if self.split != 'train':
            assert mask_num == 5

        mask_num = len(mask_path_list)
        for mask_id in range(self.mask_num):
            mask_path = os.path.join(
                color_mask_base_path, "%d.png" % (mask_id))
            # mask_path =  os.path.join(color_mask_base_path, mask_path_list[mask_id])
            color_label = load_color_mask_in_PIL_to_Tensor(
                mask_path, v_pallete=self.v2_pallete, split=self.split, crop_size=self.crop_size, crop_img_and_mask=self.crop_img_and_mask)
            # print('color_label.shape: ', color_label.shape)
            labels.append(color_label)

        imgs_tensor = torch.stack(imgs, dim=0)
        labels_tensor = torch.stack(labels, dim=0)
        initial_imgs_tensor = torch.stack(initial_imgs, dim=0)

        return imgs_tensor, initial_imgs_tensor, audio_path, labels_tensor, video_name
        

    def __len__(self):
        return len(self.df_split)




