import numpy as np
import torch
import os
import torch.utils.data as data
import shutil
from digitalpathology.adapters.batchadapter import BatchAdapter
from torchvision import transforms


def transform(transform_object, x):

    y = np.ones(len(x), dtype=np.int)
    x = np.expand_dims(x, 0)

    patches_dict = {0.5: {'patches': x.copy(), 'labels': y}}
    shapes = {0.5: (279, 279)}

    out_dict = transform_object.adapt(patches_dict, shapes=shapes, randomize=True)
    out_patch = out_dict[0.5]['patches'].squeeze(0)

    return transforms.functional.to_tensor(out_patch)


class NumpyBatchDataset(data.Dataset):
    
    def __init__(self, x_files, y_files, transform=None, order='nchw', bootstrap_seed=None, copy_path='/home/user/data', copy_flag=True, logger=None):
        super(NumpyBatchDataset, self).__init__()
 
        self.transform_object = transform        
        self.file_names = list(zip(x_files, y_files))
        self.running_idx = 0
        self.batch_length = 0
        self.transform = transform
        self.bootstrap_seed = bootstrap_seed
        self.order = order
        self.logger = logger
        
        # Copy files if neccessary.
        #
        if copy_flag:
            self._copy(copy_path)
        
        self.length = self._compute_length()
        self._load_next()

    def _copy(self, copy_path):
        
        # Make target directory if not exists.
        #
        if not os.path.exists(copy_path):
            os.makedirs(copy_path)
        
        file_names = []
        for source_path_x, source_path_y in self.file_names:
            
            target_path_x = os.path.join(copy_path, source_path_x.split('/')[-1])
            target_path_y = os.path.join(copy_path, source_path_y.split('/')[-1])
            file_names.append((target_path_x, target_path_y))
            
            if os.path.isfile(target_path_x) and os.path.isfile(target_path_y):
                self.logger.info('Exists: {} & {}'.format(target_path_x, target_path_y))
                continue
            
            if self.logger:
                self.logger.info('Currently copying files {} to {}'.format(source_path_x, target_path_x))
            
            shutil.copyfile(src=source_path_x, dst=target_path_x)
            shutil.copyfile(src=source_path_y, dst=target_path_y)

        self.file_names = file_names
            
    def _shift(self, l, n=1):
        return l[n:] + l[:n]

    def _compute_length(self):
        length = 0
        for x_filename, _ in self.file_names:
            length += np.load(x_filename, mmap_mode='r').shape[0]
        return length

    def __len__(self):
        return self.length
    
    def _load_next(self):
        '''
        Load the next buffer of data, roll the list of buffers, and initialize index.
        '''
        
        if self.logger is not None:
            self.logger.info('Opening new patch file: {}'.format(self.file_names[0][0]))
        
        # Load next batch.
        #
        self.x_buffer = np.load(self.file_names[0][0], mmap_mode='r')
        self.y_buffer = np.load(self.file_names[0][1], mmap_mode='r')
        
        # Reshape patches if needed.
        #
        if self.order == 'nwhc':
            self.x_buffer = np.transpose(self.x_buffer, axes=(0, 3, 1, 2))
        
        # Reshape labels if needed.
        #
        if len(self.y_buffer.shape) == 2 and self.y_buffer.shape[1] > 1:
            self.y_buffer = np.argmax(self.y_buffer, axis=1)
        
        # Setup filename for next batch.
        #
        self.file_names = self._shift(self.file_names)

        # Get indices for current buffer.
        #
        self.buffer_size = self.x_buffer.shape[0]
        if self.bootstrap_seed is not None:
            np.random.seed(self.bootstrap_seed)
            self.indices = np.random.choice(self.buffer_size, size=self.buffer_size, replace=True)
        else:
            self.indices = np.arange(self.buffer_size)
        
        # Reset running index.
        #
        self.running_idx = 0

    def __getitem__(self, idx):
        '''
        Load the next sample from the current buffer.
        '''
        
        # Select current index.
        #
        idx = idx % self.buffer_size 
        idx = self.indices[idx]
        
        # Load the next sample.
        #
        x, y = self.x_buffer[idx], self.y_buffer[idx]

        # Transform label to be zero indexed.
        #
        y -= 1
        
        if self.transform:
            
            # Set channels last for all transformations.
            #
            x = x.transpose((1, 2, 0))
            
            # Since target is a scalar, we only have to augment the input image.
            #
            # x = self.transform(x)
            x = transform(self.transform_object, x)
            y = torch.from_numpy(np.array(y)).squeeze().long()
            
        else:
            x = x / 255.
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(np.array(y)).squeeze().long()
        
        # Scale x from [0, 1] to [-1, 1]
        #
        x = (x - 0.5) * 2

        # self.running_idx += 1
        
        return x, y

