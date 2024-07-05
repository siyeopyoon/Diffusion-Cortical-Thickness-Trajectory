# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels



# Dataset subclass that loads personalized datasets (numpy) from the specified directory

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.dataset=[]
        textfile = []
        with open(path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into words using space as the delimiter
                words = line.split()

                textfile.append(words)
        pids = []
        for i in range(1, len(textfile)):
            pid = textfile[i][0]
            if pid not in pids:
                pids.append(pid)

        train_set_num = int(len(pids) * 0.8)

        train_set_pid = pids[:train_set_num]
        valid_set_pid = pids[train_set_num:]

        train_data = []
        for i in range(1, len(textfile)):

            pid = textfile[i][0]

            scantime = textfile[i][1]
            if scantime == 'bl':
                scantime = 0.0
            elif scantime == 'm03':
                scantime = 0.3 / 4.8
            elif scantime == 'm06':
                scantime = 0.6 / 4.8
            elif scantime == 'm12':
                scantime = 1.2 / 4.8
            elif scantime == 'm18':
                scantime = 1.6 / 4.8
            elif scantime == 'm24':
                scantime = 2.4 / 4.8
            elif scantime == 'm36':
                scantime = 3.6 / 4.8
            elif scantime == 'm48':
                scantime = 1.0

            sicktype = textfile[i][2]

            if sicktype == 'AD':
                sicktype = 0.0
            elif sicktype == 'CN':
                sicktype = 0.25
            elif sicktype == 'EMCI':
                sicktype = 0.50
            elif sicktype == 'LMCI':
                sicktype = 0.75
            elif sicktype == 'SMC':
                sicktype = 1.0

            value = textfile[i][3:]
            numeric_values = [float(x) for x in value]
            np_vector = np.array(numeric_values)
            if pid in train_set_pid:
                train_data.append((pid, scantime, sicktype, np_vector))
            else:
                valid_set_pid.append((pid, scantime, sicktype, np_vector))


        for pid in train_set_pid:
            pid_pairs = []
            for t_data_point in train_data:
                if pid == t_data_point[0]:
                    pid_pairs.append(t_data_point)
            pid_pairs = sorted(pid_pairs, key=lambda x: x[1])

            for i in range(len(pid_pairs) - 1):
                for j in range(i + 1, len(pid_pairs), 1):
                    self.dataset.append({'src':pid_pairs[i], 'trg':pid_pairs[j]})

    def patcher(self,src,trg):
        set_resolution=96
        src_thickness=np.zeros(96)
        trg_thickness = np.zeros(96)

        resolution_cortical = src[3].shape[0]  # src_thickness
        wing=(set_resolution-resolution_cortical)//2

        src_thickness[wing:wing+resolution_cortical]=src[3]
        trg_thickness[wing:wing+resolution_cortical]=trg[3]

        src_thickness = np.expand_dims(src_thickness, 0)
        trg_thickness = np.expand_dims(trg_thickness, 0)

        src_thickness = src_thickness * 0.1
        trg_thickness = trg_thickness * 0.1

        timegab=trg[1]-src[1] # times


        timegab = np.repeat(timegab, repeats=set_resolution, axis=0)
        sickname = np.repeat(src[2], repeats=set_resolution, axis=0)

        timegab = np.expand_dims(timegab, 0)
        sickname = np.expand_dims(sickname, 0)
        return src_thickness,trg_thickness,timegab,sickname

    def __getitem__(self, index: int):

        return self.patcher(self.dataset[index]['src'],self.dataset[index]['trg'])

    def __len__(self) -> int:
        return len(self.dataset)

class CustomDataset_valid(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.dataset=[]
        textfile = []
        with open(path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into words using space as the delimiter
                words = line.split()

                textfile.append(words)
        pids = []
        for i in range(1, len(textfile)):
            pid = textfile[i][0]
            if pid not in pids:
                pids.append(pid)

        train_set_num = int(len(pids) * 0.8)

        train_set_pid = pids[:train_set_num]
        valid_set_pid = pids[train_set_num:]

        train_data = []
        valid_data=[]
        for i in range(1, len(textfile)):

            pid = textfile[i][0]

            scantime = textfile[i][1]
            if scantime == 'bl':
                scantime = 0.0
            elif scantime == 'm03':
                scantime = 0.3 / 4.8
            elif scantime == 'm06':
                scantime = 0.6 / 4.8
            elif scantime == 'm12':
                scantime = 1.2 / 4.8
            elif scantime == 'm18':
                scantime = 1.6 / 4.8
            elif scantime == 'm24':
                scantime = 2.4 / 4.8
            elif scantime == 'm36':
                scantime = 3.6 / 4.8
            elif scantime == 'm48':
                scantime = 1.0

            sicktype = textfile[i][2]

            if sicktype == 'AD':
                sicktype = 0.0
            elif sicktype == 'CN':
                sicktype = 0.25
            elif sicktype == 'EMCI':
                sicktype = 0.50
            elif sicktype == 'LMCI':
                sicktype = 0.75
            elif sicktype == 'SMC':
                sicktype = 1.0

            value = textfile[i][3:]
            numeric_values = [float(x) for x in value]
            np_vector = np.array(numeric_values)
            if pid in train_set_pid:
                train_data.append((pid, scantime, sicktype, np_vector))
            else:
                valid_data.append((pid, scantime, sicktype, np_vector))


        for pid in valid_set_pid:
            pid_pairs = []
            for t_data_point in valid_data:
                if pid == t_data_point[0]:
                    pid_pairs.append(t_data_point)
            pid_pairs = sorted(pid_pairs, key=lambda x: x[1])

            for i in range(len(pid_pairs) - 1):
                for j in range(i + 1, len(pid_pairs), 1):
                    self.dataset.append({'src':pid_pairs[i], 'trg':pid_pairs[j]})

    def patcher(self,src,trg):
        set_resolution=96
        src_thickness=np.zeros(96)
        trg_thickness = np.zeros(96)

        resolution_cortical = src[3].shape[0]  # src_thickness
        wing=(set_resolution-resolution_cortical)//2

        src_thickness[wing:wing+resolution_cortical]=src[3]
        trg_thickness[wing:wing+resolution_cortical]=trg[3]

        src_thickness = np.expand_dims(src_thickness, 0)
        trg_thickness = np.expand_dims(trg_thickness, 0)

        src_thickness = src_thickness
        trg_thickness = trg_thickness

        timegab=trg[1]-src[1] # times


        timegab = np.repeat(timegab, repeats=set_resolution, axis=0)
        sickname = np.repeat(src[2], repeats=set_resolution, axis=0)

        timegab = np.expand_dims(timegab, 0)
        sickname = np.expand_dims(sickname, 0)
        return src_thickness,trg_thickness,timegab,sickname, src[0],src[1],trg[1],src[2]

    def __getitem__(self, index: int):

        return self.patcher(self.dataset[index]['src'],self.dataset[index]['trg'])

    def __len__(self) -> int:
        return len(self.dataset)

class CustomDataset_both_time(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.train_dataset=[]
        self.test_dataset=[]
        textfile = []
        with open(path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into words using space as the delimiter
                words = line.split()

                textfile.append(words)

        pids_train = []
        pids_test = []
        for i in range(1, len(textfile)):
            is_train = "train" ==textfile[i][0]
            pid = textfile[i][1]
            if is_train:

                if pid not in pids_train:
                    pids_train.append(pid)
            else:
                if pid not in pids_test:
                    pids_test.append(pid)


        train_data = []
        test_data = []


        for i in range(1, len(textfile)):

            is_train = "train" ==textfile[i][0]



            pid = textfile[i][1]

            scantime = textfile[i][2]
            if scantime == 'bl':
                scantime = 0.0
            elif scantime == 'm03':
                scantime = 0.3 / 3.6
            elif scantime == 'm06':
                scantime = 0.6 / 3.6
            elif scantime == 'm12':
                scantime = 1.2 / 3.6
            elif scantime == 'm18':
                scantime = 1.6 / 3.6
            elif scantime == 'm24':
                scantime = 2.4 / 3.6
            elif scantime == 'm36':
                scantime = 3.6 / 3.6

            baseline = textfile[i][3]
            sicktype = textfile[i][4]

            if sicktype == 'AD':
                sicktype = -1.0
            elif sicktype == 'CN':
                sicktype = 0.0
            elif sicktype == 'MCI':
                sicktype = 1.00

            age = float(textfile[i][5])
            age=age/100.0

            sex = textfile[i][6]
            if sex=="Male":
                sex=1.0
            elif sex=="Female":
                sex = -1.0

            value = textfile[i][7:]
            numeric_values = [float(x) for x in value]
            np_vector = np.array(numeric_values)
            if is_train:
                train_data.append((pid, scantime, sicktype,age,sex, np_vector))
            else:
                test_data.append((pid, scantime, sicktype,age, sex,np_vector))


        for pid in pids_train:
            pid_pairs = []
            for t_data_point in train_data:
                if pid == t_data_point[0]:
                    pid_pairs.append(t_data_point)
            pid_pairs = sorted(pid_pairs, key=lambda x: x[1])#sorted by scan date

            for i in range(len(pid_pairs) - 1): #from the previous visit
                for j in range(i + 1, len(pid_pairs), 1): # add the future visits
                    self.train_dataset.append({'src':pid_pairs[i], 'trg':pid_pairs[j]})

        for pid in pids_test:
            pid_pairs = []
            for t_data_point in test_data:
                if pid == t_data_point[0]:
                    pid_pairs.append(t_data_point)
            pid_pairs = sorted(pid_pairs, key=lambda x: x[1])  # sorted by scan date

            for i in range(len(pid_pairs) - 1):  # from the previous visit
                for j in range(i + 1, len(pid_pairs), 1):  # add the future visits
                    self.test_dataset.append({'src': pid_pairs[i], 'trg': pid_pairs[j]})


    def patcher(self,src,trg):
        set_resolution=72
        src_thickness=np.zeros(72)
        trg_thickness = np.zeros(72)

        resolution_cortical = src[5].shape[0]  # src_thickness
        wing=(set_resolution-resolution_cortical)//2

        src_thickness[wing:wing+resolution_cortical]=src[5]
        trg_thickness[wing:wing+resolution_cortical]=trg[5]

        src_thickness = np.expand_dims(src_thickness, 0)
        trg_thickness = np.expand_dims(trg_thickness, 0)

        src_thickness, trg_thickness = src_thickness, trg_thickness
        timegab = trg[1] - src[1]  # times
        sick_type = src[2]

        timegab = np.repeat(timegab, repeats=set_resolution, axis=0)
        sickname = np.repeat(sick_type, repeats=set_resolution, axis=0)
        age = np.repeat(src[3], repeats=set_resolution, axis=0)
        sex = np.repeat(src[4], repeats=set_resolution, axis=0)

        timegab = np.expand_dims(timegab, 0)
        sickname = np.expand_dims(sickname, 0)
        age = np.expand_dims(age, 0)
        sex = np.expand_dims(sex, 0)

        return src_thickness,trg_thickness,timegab,sickname,age,sex

    def __getitem__(self, index: int):

        return self.patcher(self.train_dataset[index]['src'],self.train_dataset[index]['trg'])

    def __len__(self) -> int:
        return len(self.train_dataset)


class CustomDataset_both_time_test_continous(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.train_dataset=[]
        self.test_dataset=[]
        textfile = []
        with open(path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into words using space as the delimiter
                words = line.split()

                textfile.append(words)

        pids_test = []
        for i in range(1, len(textfile)):
            is_train = "train" ==textfile[i][0]
            pid = textfile[i][1]
            if pid not in pids_test:
                pids_test.append(pid)

        test_data = []


        for i in range(1, len(textfile)):

            is_train = "train" ==textfile[i][0]



            pid = textfile[i][1]

            scantime = textfile[i][2]
            if scantime == 'bl':
                scantime = 0.0
            elif scantime == 'm03':
                scantime = 0.3 / 3.6
            elif scantime == 'm06':
                scantime = 0.6 / 3.6
            elif scantime == 'm12':
                scantime = 1.2 / 3.6
            elif scantime == 'm18':
                scantime = 1.6 / 3.6
            elif scantime == 'm24':
                scantime = 2.4 / 3.6
            elif scantime == 'm36':
                scantime = 3.6 / 3.6

            baseline = textfile[i][3]
            sicktype = textfile[i][4]

            if sicktype == 'AD':
                sicktype = -1.0
            elif sicktype == 'CN':
                sicktype = 0.0
            elif sicktype == 'MCI':
                sicktype = 1.00

            age = float(textfile[i][5])
            age=age/100.0

            sex = textfile[i][6]
            if sex=="Male":
                sex=1.0
            elif sex=="Female":
                sex = -1.0

            value = textfile[i][7:]
            numeric_values = [float(x) for x in value]
            np_vector = np.array(numeric_values)


            follow_steps=12
            for step in range (follow_steps):
                nextscantime=(0.3+0.3*float(step))/3.6
                self.test_dataset.append({'src': (pid, scantime, sicktype, age, sex, np_vector),'trg': (pid, nextscantime, sicktype, age, sex, np_vector)})

    def patcher(self,src,trg):
        set_resolution=72
        src_thickness=np.zeros(72)
        trg_thickness = np.zeros(72)

        resolution_cortical = src[5].shape[0]  # src_thickness
        wing=(set_resolution-resolution_cortical)//2

        src_thickness[wing:wing+resolution_cortical]=src[5]
        trg_thickness[wing:wing+resolution_cortical]=trg[5]

        src_thickness = np.expand_dims(src_thickness, 0)
        trg_thickness = np.expand_dims(trg_thickness, 0)

        src_thickness, trg_thickness = src_thickness, trg_thickness
        timegab = trg[1] - src[1]  # times

        timegab = np.repeat(timegab, repeats=set_resolution, axis=0)
        sickname = np.repeat(src[2], repeats=set_resolution, axis=0)
        age = np.repeat(src[3], repeats=set_resolution, axis=0)
        sex = np.repeat(src[4], repeats=set_resolution, axis=0)

        timegab = np.expand_dims(timegab, 0)
        sickname = np.expand_dims(sickname, 0)
        age = np.expand_dims(age, 0)
        sex = np.expand_dims(sex, 0)

        pid = src[0]
        src_time = src[1]*3.6
        trg_time= trg[1]*3.6
        dig = src[2]
        return src_thickness,trg_thickness,timegab,sickname,age,sex, pid, src_time, trg_time, dig

    def __getitem__(self, index: int):

        return self.patcher(self.test_dataset[index]['src'],self.test_dataset[index]['trg'])

    def __len__(self) -> int:
        return len(self.test_dataset)
