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


    def __init__(self, root, labels='basic', train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.labels = labels
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
                
                # filename contains the full path; this cuts it down to just the coords
                for i in range(300):
                    
                    entry['filenames'][i] = entry['filenames'][i][31:]

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                    self.filenames.extend(entry['filenames'])
                    self.complete_labels.extend(entry['complete_labels'])
                else:
                    self.targets.extend(entry['fine_labels']) 
                    self.filenames.extend(entry['filenames'])
                    self.complete_labels.extend(entry['complete_labels'])


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

class CRUMB_MB(CRUMB):
    
    """
    Child class to load only sources found in MiraBest
    Flag included to load either basic labels or original labels from dataset
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_MB, self).__init__(*args, **kwargs)
        
        #Only include sources which register "present" for MB (column 0)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[0] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for MB labels.')
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[0] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for MB labels.')

# ---------------------------------------------------------------------------------

class CRUMB_FRDEEP(CRUMB):
    
    """
    Child class to load only sources found in FRDEEP
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_FRDEEP, self).__init__(*args, **kwargs)
        
        #Only include sources which register "present" for FRDEEP (column 1)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[1] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for FR-DEEP labels.')
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[1] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for FR-DEEP labels.')
            
# ---------------------------------------------------------------------------------

class CRUMB_AT17(CRUMB):
    
    """
    Child class to load only sources found in AT17
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_AT17, self).__init__(*args, **kwargs)
        
        #Only include sources which register "present" for AT17 (column 2)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[2] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for AT17 labels.')
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[2] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for AT17 labels.')
    
# ---------------------------------------------------------------------------------

class CRUMB_MBHyb(CRUMB):
    
    """
    Child class to load only sources found in MiraBest
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_MBHyb, self).__init__(*args, **kwargs)
        
        #Only include sources which register "present" for MB-Hyb (column 3)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[3] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for MB-Hyb labels.')
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[3] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'original':
                
                original_labels = np.transpose(np.transpose(full_labels)[0][include])
                self.targets = original_labels.tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels or \'original\' for MB-Hyb labels.')
    
# ---------------------------------------------------------------------------------

class CRUMB_CoMBo(CRUMB):
    
    """
    Child class to load "combo" of sources in MiraBest and MB Hybrid
    MiraBest labels take precedent over MB Hybrid labels by default
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_CoMBo, self).__init__(*args, **kwargs)
        
        #Include sources which register "present" for MB (column 0) or MB Hyb (column 3)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.logical_or(np.transpose(full_labels)[0] != -1, 
                                                         np.transpose(full_labels)[3] != -1)))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'MB':
                
                MB_labels = np.transpose(full_labels)[0][include]
                MBHyb_labels = np.transpose(full_labels)[3][include]
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if MB_labels[i] != -1:
                        
                        combined_labels[i] = MB_labels[i]
                        
                    elif MBHyb_labels[i] == 0:
                        
                        combined_labels[i] = 8
                        
                    else:
                        
                        combined_labels[i] = 9
                
                self.targets = combined_labels.astype(int).tolist()
                
            elif self.labels == 'MBHyb':
                
                MB_labels = np.transpose(np.transpose(full_labels)[0][include])
                MBHyb_labels = np.transpose(np.transpose(full_labels)[3][include])
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if MBHyb_labels[i] == 0:
                        
                        combined_labels[i] = 8
                        
                    elif MBHyb_labels[i] == 1:
                        
                        combined_labels[i] = 9
                        
                    else:
                        
                        combined_labels[i] = MB_labels[i]
                
                self.targets = combined_labels.astype(int).tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels, \'MB\' to prioritise MB labels in case of disagreement, or \'MBHyb\' to prioritise MBHyb labels in case of disagreement.')
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.transpose(full_labels)[0] != -1))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'MB':
                
                MB_labels = np.transpose(full_labels)[0][include]
                MBHyb_labels = np.transpose(full_labels)[3][include]
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if MB_labels[i] != -1:
                        
                        combined_labels[i] = MB_labels[i]
                        
                    elif MBHyb_labels[i] == 0:
                        
                        combined_labels[i] = 8
                        
                    else:
                        
                        combined_labels[i] = 9
                
                self.targets = combined_labels.astype(int).tolist()
                
            elif self.labels == 'MBHyb':
                
                MB_labels = np.transpose(np.transpose(full_labels)[0][include])
                MBHyb_labels = np.transpose(np.transpose(full_labels)[3][include])
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if MBHyb_labels[i] == 0:
                        
                        combined_labels[i] = 8
                        
                    elif MBHyb_labels[i] == 1:
                        
                        combined_labels[i] = 9
                        
                    else:
                        
                        combined_labels[i] = MB_labels[i]
                
                self.targets = combined_labels.astype(int).tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels, \'MB\' to prioritise MB labels in case of disagreement, or \'MBHyb\' to prioritise MBHyb labels in case of disagreement.')
    
# ---------------------------------------------------------------------------------

class CRUMB_NoMB(CRUMB):
    
    """
    Child class to load only sources not found in MiraBest or MB Hybrid
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_NoMB, self).__init__(*args, **kwargs)
        
        #Only include sources which register "not present" for both MB (column 0) and Hyb (column 3)
        
        if self.train:
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.logical_and(np.transpose(full_labels)[0] == -1, 
                                                         np.transpose(full_labels)[3] == -1)))
            
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'FRDEEP':
                
                FRDEEP_labels = np.transpose(full_labels)[1][include]
                AT17_labels = np.transpose(full_labels)[2][include]
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if FRDEEP_labels[i] != -1:
                        
                        combined_labels[i] = FRDEEP_labels[i]
                        
                    else: 
                        
                        combined_labels[i] = AT17_labels[i]
                
                self.targets = combined_labels.astype(int).tolist()
                
            elif self.labels == 'AT17':
                
                FRDEEP_labels = np.transpose(full_labels)[1][include]
                AT17_labels = np.transpose(full_labels)[2][include]
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if AT17_labels[i] != -1:
                        
                        combined_labels[i] = AT17_labels[i]
                    
                    else:
                        
                        combined_labels[i] = FRDEEP_labels[i]
                
                self.targets = combined_labels.astype(int).tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels, \'FRDEEP\' to prioritise FRDEEP labels in case of disagreement, or \'AT17\' to prioritise AT17 labels in case of disagreement.')
            
        else:
            
            full_labels = np.array(self.complete_labels)
            include = np.squeeze(np.where(np.logical_and(np.transpose(full_labels)[0] == -1, 
                                                      np.transpose(full_labels)[3] == -1)))
            targets = np.array(self.targets)
            self.data = self.data[include]
            self.filenames = list(self.filenames[i] for i in include)
            self.complete_labels = full_labels[include].tolist()
            
            if self.labels == 'basic':
                
                self.targets = targets[include].tolist()
            
            elif self.labels == 'FRDEEP':
                
                FRDEEP_labels = np.transpose(full_labels)[1][include]
                AT17_labels = np.transpose(full_labels)[2][include]
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if FRDEEP_labels[i] != -1:
                        
                        combined_labels[i] = FRDEEP_labels[i]
                        
                    else: 
                        
                        combined_labels[i] = AT17_labels[i]
                
                self.targets = combined_labels.astype(int).tolist()
                
            elif self.labels == 'AT17':
                
                FRDEEP_labels = np.transpose(full_labels)[1][include]
                AT17_labels = np.transpose(full_labels)[2][include]
                
                combined_labels = np.zeros(len(include))
                
                for i in range(len(include)):
                    
                    if AT17_labels[i] != -1:
                        
                        combined_labels[i] = AT17_labels[i]
                    
                    else:
                        
                        combined_labels[i] = FRDEEP_labels[i]
                
                self.targets = combined_labels.astype(int).tolist()
                
            else:
                
                print('Invalid label choice. Please select either \'basic\' for default CRUMB labels, \'FRDEEP\' to prioritise FRDEEP labels in case of disagreement, or \'AT17\' to prioritise AT17 labels in case of disagreement.')
                
# ---------------------------------------------------------------------------------

class CRUMB_4Class(CRUMB):
    
    """
    Child class to load basic label for a four-class system
    Classes: 0 (FRI), 1 (FRII), 2 (Bent), 3 (Hybrid)
    """
    
    def __init__(self, *args, **kwargs):
        super(CRUMB_4Class, self).__init__(*args, **kwargs)
        
        #Change any hybrid source targets to 3
        #Then find sources with labels 1, 2 or 4 in MB or 2 in AT17 and change their target to 2
        
        if self.train:
            
            full_labels = np.transpose(np.array(self.complete_labels))
            hybrids = np.where(self.targets == 2)
            conf_wat_sources = np.squeeze(np.where(full_labels[0] == 1))
            conf_ht_sources = np.squeeze(np.where(full_labels[0] == 2))
            unc_wat_sources = np.squeeze(np.where(full_labels[0] == 4))
            at17_bent_sources = np.squeeze(np.where(full_labels[2] == 2))
            
            updated_targets = np.array(self.targets)
            updated_targets[hybrids] = 3
            updated_targets[conf_wat_sources] = 2
            updated_targets[conf_ht_sources] = 2
            updated_targets[unc_wat_sources] = 2
            
            for i in range(len(at17_bent_sources)):
                
                if full_labels[0][at17_bent_sources[i]] == -1:
                    
                    updated_targets[at17_bent_sources[i]] = 2
            
            self.targets = updated_targets.astype(int).tolist()
            
        else:
            
            full_labels = np.transpose(np.array(self.complete_labels))
            hybrids = np.where(self.targets == 2)
            conf_wat_sources = np.squeeze(np.where(full_labels[0] == 1))
            conf_ht_sources = np.squeeze(np.where(full_labels[0] == 2))
            unc_wat_sources = np.squeeze(np.where(full_labels[0] == 4))
            at17_bent_sources = np.squeeze(np.where(full_labels[2] == 2))
            
            updated_targets = np.array(self.targets)
            updated_targets[hybrids] = 3
            updated_targets[conf_wat_sources] = 2
            updated_targets[conf_ht_sources] = 2
            updated_targets[unc_wat_sources] = 2
            
            for i in range(len(at17_bent_sources)):
                
                if full_labels[0][at17_bent_sources[i]] == -1:
                    
                    updated_targets[at17_bent_sources[i]] = 2
            
            self.targets = updated_targets.astype(int).tolist()
            
# ---------------------------------------------------------------------------------