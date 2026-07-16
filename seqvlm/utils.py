'''
数据加载
'''
import base64
from collections import defaultdict
from datetime import datetime
import json
import logging
import numpy as np
import os
import sys
import torch

from PIL import Image
from io import BytesIO


def load_seg_inst(scene_id):
    '''
    把 Mask3D proposal 转成：
    ins_labels: proposal 类别
    ins_locs:   proposal 3D bbox, [cx, cy, cz, sx, sy, sz]
    ins_scores: proposal 置信度
    center:     scene center
    '''
    root_dir = '../data/mask3d/'
    data = np.load(os.path.join(root_dir, scene_id + '.npz'), allow_pickle=True)
    ins_labels = list(data['ins_labels'])
    if 'ins_scores' in data.files:
        ins_scores = [float(x) for x in data['ins_scores']]
    else:
        ins_scores = [1.0] * len(ins_labels)
    
    ins_locs = []
    scene_pc = []
    for obj in data['ins_pcds']:
        if obj.shape[0] == 0:
            obj = np.zeros((1, 6))
        obj_pcd = obj[:, :3]
        scene_pc.append(obj_pcd)
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        ins_locs.append(np.concatenate([obj_center, obj_size], 0))

    scene_pc = np.concatenate(scene_pc, 0)
    center = (scene_pc.max(0) + scene_pc.min(0)) / 2
    
    return ins_labels, ins_locs, ins_scores, center


def load_pc(scene_id):
    root_dir = '../data/referit3d/scan_data'
    pcds, _, _, instance_labels = torch.load(
        os.path.join(root_dir, 'pcd_with_global_alignment', '%s.pth' % scene_id))
    inst_to_name = json.load(open(os.path.join(root_dir, 'instance_id_to_name', '%s.json' % scene_id)))

    obj_labels = []
    inst_locs = []
    obj_ids = []
    
    for i, obj_label in enumerate(inst_to_name):
        if obj_label in ['wall', 'floor', 'ceiling']:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, 'scene: %s, obj %d' % (scene_id, i)
        
        obj_pcd = pcds[mask]
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        obj_labels.append(obj_label)
        obj_ids.append(i)

    return obj_ids, obj_labels, inst_locs


def calc_iou(box_a, box_b):
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    
    return 1.0 * intersection / union


class _Tee:
    """Write to multiple streams (terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        if not data:
            return
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return getattr(self.streams[0], 'isatty', lambda: False)()


_log_file = None

# dataset -> (full_log_prefix, metrics_log_prefix)
_LOG_PREFIX = {
    'scanrefer': ('scanrefer_full', 'visprog_scanrefer'),
    'nr3d': ('nr3d_full', 'nr3d'),
}


def setup_run_logging(dataset: str, timestamp: str = None):
    """
    Split logging into two files under ../logs/:
      - full log: all stdout/stderr (scanrefer_full_{ts}.log / nr3d_full_{ts}.log)
      - metrics log: periodic logger.info only (visprog_scanrefer_{ts}.log / nr3d_{ts}.log)

    Returns (metrics_logger, full_log_path, metrics_log_path).
    """
    global _log_file

    if dataset not in _LOG_PREFIX:
        raise ValueError(f"dataset must be one of {list(_LOG_PREFIX)}, got {dataset!r}")

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    full_prefix, metrics_prefix = _LOG_PREFIX[dataset]
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    os.makedirs(log_dir, exist_ok=True)

    full_log_path = os.path.join(log_dir, f'{full_prefix}_{timestamp}.log')
    metrics_log_path = os.path.join(log_dir, f'{metrics_prefix}_{timestamp}.log')

    if _log_file is not None:
        _log_file.close()
    _log_file = open(full_log_path, 'w', encoding='utf-8', buffering=1)

    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    metrics_logger = logging.getLogger(f'seq-vlm-metrics-{dataset}')
    metrics_logger.handlers.clear()
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.propagate = False
    metrics_handler = logging.FileHandler(metrics_log_path, mode='w', encoding='utf-8')
    metrics_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    metrics_logger.addHandler(metrics_handler)

    print(f'Full log: {full_log_path}')
    print(f'Metrics log: {metrics_log_path}')
    return metrics_logger, full_log_path, metrics_log_path


def encode_image_to_base64(image_path):
    with Image.open(image_path) as image:
        buf = BytesIO()
        image.save(buf, format='JPEG')
        byte_data = buf.getvalue()
        base64_str = base64.b64encode(byte_data).decode('utf-8')
        return base64_str