import os
import mne
import torch
import numpy as np
import pandas as pd

from channels import stringify_channel_mapping
from collections import OrderedDict
from torch.utils.data import Dataset as TorchDataset
from utils import InstanceTransform, MappingDeep1010, TemporalInterpolation, To1020
from extras import *

def edf_to_array_epochs(path, label, format_type, tlen, overlap, data_max, data_min,
                        chns_consider, apply_winsor):
    if format_type == 'edf':
        new_raw = mne.io.read_raw_edf(path, preload=True)
        # TODO: Remove this hardcoding
        #assert new_raw.ch_names == ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental',
        #                            'Temp rectal', 'Event marker']
        #new_raw.rename_channels({'EEG Fpz-Cz': 'Fz', 'EEG Pz-Oz': 'Pz'})
    elif format_type == 'set':
        new_raw = mne.io.read_raw_eeglab(path, preload=True)
    else:
        print('Error: You must give a valid file format.')

    # Consider only some channels
    ch_to_remove = list(set(new_raw.ch_names) - set(chns_consider))
    new_raw.drop_channels(ch_to_remove)

    print('Processing the epochs of this record')
    epochs_one_rec = scz_BENDR_dataset(new_raw, tlen=tlen, apply_winsor=apply_winsor, overlap=overlap, label=label,
                                        data_max=data_max, data_min=data_min)
    is_first_epoch = True
    for ep_id in range(0, len(epochs_one_rec)):
        if is_first_epoch:
            # Consider only 3 last channels (Fz Cz Pz)
            all_eps_x = torch.unsqueeze(epochs_one_rec.__getitem__(ep_id)[0], dim=0)
            all_eps_y = torch.unsqueeze(epochs_one_rec.__getitem__(ep_id)[1], dim=0)
            is_first_epoch = False
        else:
            all_eps_x = torch.cat([all_eps_x, torch.unsqueeze(epochs_one_rec.__getitem__(ep_id)[0], dim=0)], 0)
            all_eps_y = torch.cat([all_eps_y, torch.unsqueeze(epochs_one_rec.__getitem__(ep_id)[1], dim=0)], 0)
    return all_eps_x, all_eps_y


def charge_all_data(directory, format_type, tlen, overlap, data_max, data_min,
                    chns_consider, labels_path, target_f, apply_winsor):
    array_epochs_all_subjects = []
    for root, folders, _ in os.walk(directory):
        for fold_day in sorted(folders):
            for root_day, _, files in os.walk(os.path.join(root, fold_day)):
                for file in sorted(files):
                    # TODO: HARDCODED!!
                    if file.endswith(format_type):
                        print('====================Processing record ' + str(file) + '======================')
                        subj_of_file = file[:5]
                        target_info = pd.read_csv('{}/{}_labels.csv'.format(labels_path, subj_of_file),
                                                  index_col=0, decimal=',')
                        target_info = target_info.to_dict()
                        target_info = target_info[target_f]
                        # TODO: hardcoded
                        label = target_info[file[:-13]]
                        array_epochs_subj_rec = edf_to_array_epochs(os.path.join(root_day, file), label, format_type, tlen, overlap,
                                                                data_max, data_min, chns_consider,apply_winsor)
                        array_epochs_all_subjects.append(array_epochs_subj_rec)
    return array_epochs_all_subjects


class standardDataset(TorchDataset):
    def __init__(self, X, y):
        if X.shape[0] != y.shape[0]:
            print('First dimesion of X and y must be the same')
            return

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        Xi = self.X[idx]
        yi = self.y[idx]
        # TODO: Is necesary to unsqueeze Xi (in dim 0) ??
        return Xi.float(), yi.float()


class scz_BENDR_dataset(TorchDataset):
    def __init__(self, raw: mne.io.Raw, data_max, data_min, tlen, apply_winsor, tmin=0, new_sfreq=256,
                 overlap=2, label=None):  # deep1010map,
        self.filename = raw.filenames[0].split('/')[-1]
        self.orig_sfreq = raw.info['sfreq']
        self.new_sfreq = new_sfreq
        self.ch_names = raw.ch_names
        self.apply_winsor = apply_winsor

        ch_list = []
        for i in raw.ch_names:
            ch_list.append([i, '2'])
        self.ch_list = np.array(ch_list, dtype='<U21')

        # Segment the recording
        '''
        if had_annotations:
            # TODO: HARDCODED
            ans = None
            for root, _, files in os.walk('/' + os.path.join(*raw.filenames[0].split('/')[:-1])):
                for file in sorted(files):
                    if file.startswith(raw.filenames[0].split('/')[-1][:6]) and file.endswith('Hypnogram.edf'):
                        ans = mne.read_annotations(os.path.join(root, file))
            ans.delete(-1)
            raw.set_annotations(ans)
            events = mne.events_from_annotations(raw, {'Sleep stage 1': 1, 'Sleep stage 2': 2,
                                                       'Sleep stage 3': 3, 'Sleep stage 4': 3,
                                                       'Sleep stage R': 4, 'Sleep stage W': 0})
            # events = mne.events_from_annotations(raw, event_id=self.event_ids, chunk_duration=None)[0]
            self.epochs = mne.Epochs(raw, events[0], tmin=tmin, tmax=tmin + tlen - 1 / self.orig_sfreq, preload=True,
                                     decim=1,
                                     baseline=None, reject_by_annotation=False)
            self.epoch_codes_to_class_labels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        else:
        '''
        self.epochs = mne.make_fixed_length_epochs(raw, id=label, duration=tlen, overlap=overlap)
        self.epochs.drop_bad()

        self.tlen = tlen
        self.transforms = list()
        self._different_deep1010s = list()
        # TODO: whats this?
        self.return_mask = False
        self.data_max = data_max
        self.data_min = data_min

        # Maps particular channels to a consistent index from each loaded trial
        xform = MappingDeep1010(channels=self.ch_list, data_max=data_max, data_min=data_min,
                                return_mask=self.return_mask)
        self.add_transform(xform)
        self._add_deep1010(self.ch_names, xform.mapping.numpy(), [])
        self.new_sequence_len = int(tlen * self.new_sfreq)

        # Over- or Under- sampling to match the desired sampling freq
        """
        WARNING: if downsample the signal below a suitable Nyquist freq given the low-pass filtering
        of the original singal, it will cause aliasing artifacts (ruined data)
        """
        self.add_transform(TemporalInterpolation(self.new_sequence_len, new_sfreq=self.new_sfreq))

        print("Constructed {} channel maps".format(len(self._different_deep1010s)))
        for names, deep_mapping, unused, count in self._different_deep1010s:
            print('=' * 20)
            print("Used by {} recordings:".format(count))
            print(stringify_channel_mapping(names, deep_mapping))
            print('-' * 20)
            print("Excluded {}".format(unused))
            print('=' * 20)

        # Only conserve 19 channels of the 10/20 International System + 1 scale channel
        self.add_transform(To1020())
        self._safe_mode = False

    def __getitem__(self, index):
        ep = self.epochs[index]

        # TODO Could have a speedup if not using ep, but items, but would need to drop bads?
        x = ep.get_data() #(list(range(len(self.ch_names))))
        if len(x.shape) != 3 or 0 in x.shape:
            print("I don't know why: This  `filename` index{}/{}".format(index, len(self)))
            print(self.epochs.info['description'])
            print("Using trial {} in place for now...".format(index - 1))
            return self.__getitem__(index - 1)


        # 3 MAD threshold Winsorising
        x = torch.from_numpy(x.squeeze(0)).float()
        if self.apply_winsor:
            for i in range(x.shape[-2]):
                assert len(x.shape) == 2
                mad = MAD(x[i, :])
                med = np.median(x[i, :])
                # Winsorising
                x[i, :] = np.clip(x[i, :], med - 3 * mad, med + 3 * mad)

        y = torch.tensor(ep.events[0, -1]).squeeze().long()
        return self._execute_transforms(x, y)

    def __len__(self):
        return len(self.epochs)

        # def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        #    self.epochs = preprocessor(recording=self)
        #    if apply_transform:
        #        self.add_transform(preprocessor.get_transform())

    def __str__(self):
        return "{} trials | {} transforms".format(len(self), len(self.transforms))

    def event_mapping(self):
        """
        Maps the labels returned by this to the events as recorded in the original annotations or stim channel.
        Returns
        -------
        mapping : dict
        Keys are the class labels used by this object, values are the original event signifier.
        """
        return self.epoch_codes_to_class_labels

    def get_targets(self):
        return np.apply_along_axis(lambda x: self.epoch_codes_to_class_labels[x[0]], 1,
                                   self.epochs.events[list(range(len(self.epochs))), -1, np.newaxis]).squeeze()

    def add_transform(self, transform_item):
        """
        Add a transformation that is applied to every fetched item in the dataset
        Parameters
        ----------
        transform : BaseTransform
                    For each item retrieved by __getitem__, transform is called to modify that item.
        """
        if isinstance(transform_item, InstanceTransform):
            self.transforms.append(transform_item)

    def _add_deep1010(self, ch_names: list, deep1010map: np.ndarray, unused_ch):
        for i, (old_names, old_map, unused_ch, count) in enumerate(self._different_deep1010s):
            if np.all(deep1010map == old_map):
                self._different_deep1010s[i] = (old_names, old_map, unused_ch, count + 1)
                return
        self._different_deep1010s.append((ch_names, deep1010map, unused_ch, 1))

    def _execute_transforms(self, *x):
        for transform in self.transforms:
            assert isinstance(transform, InstanceTransform)
            if transform.only_trial_data:
                new_x = transform(x[0])
                if isinstance(new_x, (list, tuple)):
                    x = (*new_x, *x[1:])
                else:
                    x = (new_x, *x[1:])
            else:
                x = transform(*x)

            if self._safe_mode:
                for i in range(len(x)):
                    if torch.any(torch.isnan(x[i])):
                        raise ValueError('error')
        return x