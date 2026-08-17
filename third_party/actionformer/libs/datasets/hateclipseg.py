import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("hateclipseg")
class HateClipSegDataset(Dataset):
    """HateClipSeg temporal localization, in ActionFormer's THUMOS-14 data contract.

    Byte-identical to `thumos14.py` except for three project-specific points, each of
    which is stated in `idea-stage/R16_DETBASE_FREEZE.md`:
      1. `tiou_thresholds` are the paper's {0.3, 0.5, 0.7}, not THUMOS's linspace(0.3,0.7,5).
      2. videos with **zero** offensive segments are kept (they are legal background-only
         training and evaluation videos here, unlike THUMOS); `truncate_feats` is therefore
         called with `has_action=False` for them, otherwise it spins 200 trials and returns
         a garbage crop.
      3. an empty `annotations` list yields a (0,2) segment array rather than `None`, so a
         background-only video still reaches the loss as pure background.
    """

    def __init__(
        self,
        is_training,
        split,
        feat_folder,
        json_file,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        input_dim,
        num_classes,
        file_prefix,
        file_ext,
        force_upsampling,
    ):
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.file_prefix = file_prefix if file_prefix is not None else ''
        self.file_ext = file_ext
        self.json_file = json_file

        self.split = split
        self.is_training = is_training

        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes, (len(label_dict), num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict

        self.db_attributes = {
            'dataset_name': 'hateclipseg',
            'tiou_thresholds': np.array([0.3, 0.5, 0.7]),
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        label_dict = {}
        for key, value in json_db.items():
            for act in value['annotations']:
                label_dict[act['label']] = act['label_id']
        # a split may legitimately contain no annotation of some class
        if len(label_dict) == 0:
            label_dict = {'offensive': 0}

        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            feat_file = os.path.join(
                self.feat_folder, self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            fps = self.default_fps if self.default_fps is not None else value['fps']
            duration = value.get('duration', 1e8)

            segments, labels = [], []
            for act in value['annotations']:
                segments.append(act['segment'])
                labels.append([label_dict[act['label']]])
            if len(segments) > 0:
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = np.zeros((0, 2), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

            dict_db += ({'id': key,
                         'fps': fps,
                         'duration': duration,
                         'segments': segments,
                         'labels': labels}, )
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        filename = os.path.join(
            self.feat_folder, self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        segments = torch.from_numpy(
            video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
        )
        labels = torch.from_numpy(video_item['labels'])

        data_dict = {'video_id': video_item['id'],
                     'feats': feats,
                     'segments': segments,
                     'labels': labels,
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': self.num_frames}

        if self.is_training:
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset,
                self.crop_ratio, has_action=(segments.shape[0] > 0)
            )
        return data_dict
