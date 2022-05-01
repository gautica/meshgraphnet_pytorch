import os
import torch
from torch_geometric.data import Dataset, Data, download_url
import numpy as np
import glob
import json
import tensorflow as tf
import functools


class MeshDataset(Dataset):
    def __init__(self,
                 config,
                 split='train',
                 transform=None,
                 pre_transform=None):
        self.split = split
        self.config = config
        self.dataset_name = config.dataset_name
        super().__init__(self.config.data_dir, transform, pre_transform)

    @property
    def raw_file_names(self): 
        return ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']

    @property
    def processed_file_names(self):
        return glob.glob(os.path.join(self.processed_dir, self.split, 'data_*.pt'))

    def triangles_to_edges(self, faces):
        """Computes mesh edges from triangles."""
        # collect edges from triangles
        edges = np.concatenate([faces[:, 0:2],
                               faces[:, 1:3],
                               np.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
        receivers = np.min(edges, axis=1)
        senders = np.max(edges, axis=1)
        packed_edges = np.stack([senders, receivers], axis=1)
        # remove duplicates and unpack
        unique_edges = np.unique(packed_edges, axis=0)
        senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
        # create two-way connectivity
        return np.stack([np.concatenate([senders, receivers], axis=0),
              np.concatenate([receivers, senders], axis=0)], axis=0)

    def _parse(self, proto, meta):
        """Parses a trajectory from tf.Example."""
        feature_lists = {k: tf.io.VarLenFeature(tf.string)
                       for k in meta['field_names']}
        features = tf.io.parse_single_example(proto, feature_lists)
        out = {}
        for key, field in meta['features'].items():
            data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
            data = tf.reshape(data, field['shape'])
            if field['type'] == 'static':
                pass
            elif field['type'] == 'dynamic_varlen':
                length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
                length = tf.reshape(length, [-1])
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
            elif field['type'] == 'dynamic':
                data = tf.transpose(data, perm=[1, 0, 2])       # (num_nodes, length_trajectory, feature_dim)
            elif field['type'] != 'dynamic':
                raise ValueError('invalid data format')
            out[key] = data
        return out

    def add_targets(self, ds, meta, target, add_history):
        """Adds target and optionally history fields to dataframe."""
        def fn(trajectory):
            out = {}
            for key, val in trajectory.items():
                if meta['features'][key]['type'] == 'dynamic':
                    out[key] = val[:, 1:-1]
                else:
                    out[key] = val
                if key == target:
                    if add_history:
                        out['prev_'+key] = val[:, 0:-2]
                    out['target_'+key] = val[:, 2:]
            return out
        return ds.map(fn, num_parallel_calls=8)

    def download(self):
        print(f'Download dataset {self.dataset_name} to {self.raw_dir}')
        for file in ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']:
            url = f"https://storage.googleapis.com/dm-meshgraphnets/{self.dataset_name}/{file}"
            download_url(url=url, folder=self.raw_dir)

    def process(self):
        with open(os.path.join(self.raw_dir, 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        ds = tf.data.TFRecordDataset(os.path.join(self.raw_dir, f'%s.tfrecord' % self.split))
        ds = ds.map(functools.partial(self._parse, meta=meta), num_parallel_calls=8)
        ds = self.add_targets(ds, meta, self.config.field, self.config.history)

        os.makedirs(os.path.join(self.processed_dir, self.split), exist_ok=True)
        for idx, data in enumerate(ds):
            d = {}
            for key, value in data.items():
                d[key] = torch.from_numpy(value.numpy()).squeeze(dim=0)
            cells = d['cells']
            edges = torch.from_numpy(self.triangles_to_edges(cells)).long()           # build edges
            data_transformed = Data(edge_index=edges, **d)
            torch.save(data_transformed, os.path.join(self.processed_dir, self.split, f'data_{idx}.pt')) # (#node, traj, feat_dim)

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.split, f'data_{idx}.pt'))
        return data
