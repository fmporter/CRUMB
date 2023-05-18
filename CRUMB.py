from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class CRUMB(data.Dataset):
    """
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``CRUMB.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

     base_folder = 'CRUMB_batches'
    url = "http://www.jb.man.ac.uk/research/MiraBest/CRUMB/CRUMB_batches.tar.gz" 
    filename = "CRUMB_batches.tar.gz"
    tgz_md5 = 'a33c0564b99d66fb825e224a0392bc78'
    train_list = [
                  ['data_batch_1', '004e97220b29da803cf67e762ade4b52'],
                  ['data_batch_2', 'a05122141382c3ccec5d5c717a582b16'],
                  ['data_batch_3', 'aada5e8eab52732b3d171b158081bfa7'],
                  ['data_batch_4', 'ebc353fb9059dbeb44da28a50e6092bc'],
                  ['data_batch_5', '5d9459f61a710b27b3a790d3686fb14d'],
                  ['data_batch_6', '965c62bfff96acf83245e68ca42e0c10'],
                  ]

    test_list = [
                 ['test_batch', '0cd9c3869700b720f4adcadba79d793c'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': '58f77558538ea5cd398fea6300201332',
                }


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.filenames = []
        self.complete_labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                    self.filenames.extend(entry['filenames'])
                    self.complete_labels.extend(entry['complete_labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    self.filenames.extend(entry['filenames'])
                    self.filenames.extend(entry['complete_labels'])


        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            #print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
# ---------------------------------------------------------------------------------

class Not_MB(CRUMB):
    
    """
    Child class to load only sources not found in MiraBest or MB Hybrid
    """
    
    def __init__(self, *args, **kwargs):
        super(Not_MB, self).__init__(*args, **kwargs)
        
        #Only include sources which register "not present" for both MB (column 0) and Hyb (column 3)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.logical_and(np.transpose(full_labels)[0] == -1, 
                                                         np.transpose(full_labels)[3] == -1)))
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.targets = targets[include].tolist()
            self.complete_labels = full_labels[include].tolist()
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.logical_and(np.transpose(full_labels)[0] == -1, 
                                                      np.transpose(full_labels)[3] == -1)))
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.targets = targets[include].tolist()
            self.complete_labels = full_labels[include].tolist()
