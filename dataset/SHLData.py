'''
Overload of torch.utils.data.Dataset for SHL Locomotion Challenge dataset.
Note the expected folder structure.
Example usage:

shl_data = SHLData(root_path="../data", flag="train", mode="singlefolder")
dataloader = DataLoader(
            shl_data,
            batch_size=32,
            shuffle=True,
            num_workers=1,
        )
for targets, labels in dataloader:
    # use data
'''

from torch.utils.data import Dataset
import posixpath as path
from os import walk
import numpy as np
from torch import unsqueeze, from_numpy, as_tensor


class SHLData(Dataset):
    '''
    | Dataset for SHL Locomotion Challenge. Please note that it expects the following folder structure
    | from root_path.
    | Data Folder structure:
    |       root_path
    |           -train
    |               ---Bag
    |               ---Hand
    |               ---Torso
    |               ---Hips
    |           -validation
    |               ---Bag
    |               ---Hand
    |               ---Torso
    |               ---Hips
    |           -test
    | Args:
    |   root_path = path to data files, see folder structure above
    |    path = one of "train", "test", "validation"; which data to load
    |    mode = one of "singlefile", "singlefolder", "all"; whether to load data from single data file all from all files
    |    sensor = data from which sensor to load, used only in singlefile mode
    |    location = data from which sensor location to load, used in singlefile and singlefolder modes

    '''

    def __init__(self, root_path, flag: str = "train", mode: str = "singlefile", sensor: str = "Acc_x", location: str = "Hips"):
        self.root_path = root_path
        self.flag = flag
        self.sensor = sensor
        self.location = location
        self.mode = mode.lower()
        if isinstance(self.flag, str):
            self.flag = self.flag.lower()  # Avoid ambiguities with Train/TRAIN/train etc
        if self.mode == "singlefile":
            self.x_data, self.y_data = self._load_from_file()
        elif self.mode == "singlefolder":
            self.x_data, self.y_data = self._load_from_folder()
        elif self.mode == "all":
            raise NotImplementedError("Mode 'all' is not yet implemented")
        else:
            raise ValueError(f"Mode not supported, received: {mode}")
        if self.flag != 'test':
            self.y_data = self.y_data - 1  # cross entropy expects 0-based indexes

    def _load_from_file(self, filepath=None, do_load_labels=True):
        '''
        | Load singular data file
        |
        | Args
        | filepath: path to file to load, if not provided, inferred from self.sensor, self.location
        | do_load_labels: whether or not to load Label.txt file
        |
        | Raises
        | FileNotFoundError
        '''
        sensor = self.sensor
        location = self.location
        if not filepath:
            if self.flag == 'test':
                filepath = path.join(
                    self.root_path, self.flag, f"{sensor}.txt")
            else:
                filepath = path.join(
                    self.root_path, self.flag, location, f"{sensor}.txt")
        try:
            if self.flag == 'test':
                np_data = np.loadtxt(filepath, dtype=np.float32, delimiter=",")        
            else:
                np_data = np.loadtxt(filepath, dtype=np.float32)
            if do_load_labels and self.flag != 'test':
                labelpath = path.join(
                    self.root_path, self.flag, location, "Label.txt")
                y_data = np.loadtxt(labelpath, dtype=int)
                y_data = np.median(y_data, axis=1).astype(int)
                return np_data, y_data
            if self.flag == 'test' and do_load_labels:
                return np_data, []
            return np_data
        except Exception as e:
            print(e)
            raise FileNotFoundError(
                f"Could not load data file: {filepath}, check whether provided sensor location, sensor type and folder structure are correct")

    def _load_from_folder(self):
        '''
        Load all sensors from folder (location)
        '''
        if self.flag == "test":
            filepath = path.join(self.root_path, self.flag)
        else:
            filepath = path.join(self.root_path, self.flag, self.location)
        labels = None
        add_data = []
        for root, dirs, files in walk(filepath):
            for file in files:
                if file.split(".")[-1] == "txt" and "Label" not in file:
                    if labels is None and self.flag != 'test':
                        data_to_append, labels = self._load_from_file(
                            path.join(root, file))
                        add_data.append(data_to_append)
                    else:
                        add_data.append(self._load_from_file(
                            path.join(root, file), do_load_labels=False))
        data = np.stack(add_data, axis=2)
        if self.flag == 'test':
            return data, []
        return data, labels
    
    def _load_all(self):
        '''
        Load all sensors from all locations
        '''
        filepath = path.join(self.root_path, self.flag)


    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, ind):
        if self.flag == "test":      
            if self.mode == "singlefile":
                # for dimensions compatibility
                return unsqueeze(from_numpy(self.x_data[ind]), 0)
            else:
                return from_numpy(self.x_data[ind])
        else:
            if self.mode == "singlefile":
                # for dimensions compatibility
                return unsqueeze(from_numpy(self.x_data[ind]), 0), as_tensor([self.y_data[ind]])
            else:
                return from_numpy(self.x_data[ind]), as_tensor([self.y_data[ind]])
