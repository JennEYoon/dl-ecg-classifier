import numpy as np, pandas as pd, sys
from torch.utils.data import Dataset
from .dataset_utils import load_data, encode_metadata
from .transforms import Compose, RandomClip, Normalize, ValClip, Retype

def get_transforms(dataset_type):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    seq_length = 4096
    normalizetype = '0-1'
    
    data_transforms = {
        
        'train': Compose([
            RandomClip(w=seq_length),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        
        'val': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0),
        
        'test': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0)
    }
    return data_transforms[dataset_type]

class ECGDataset(Dataset):
    def __init__(self, csv_path, transforms, all_features):
        ''' Class implementation of Dataset of ECG recordings
    
        :param path: The directory of the data used
        :type path: str
        :param transform: Transforms used for ECG recording
        :type transform: datasets.transforms.Compose
        '''

        # Load metadata from CSV
        df = pd.read_csv(csv_path)
        self.data = df['path'].tolist()
        labels = df.iloc[:, 4:].values
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        self.fs = df['fs'].tolist()

        self.age = df['age'].tolist()
        self.gender = df['gender'].tolist()

        self.transforms = transforms
        self.all_features = all_features

        # Compute mean and std for TOP features
        feats = all_features.select_dtypes(include='number') # Number as there's filename also in columns
        feats[np.isinf(feats)] = np.nan
        self.feats_mean = np.nanmean(feats, axis=0)
        self.feats_std =  np.nanstd(feats, axis=0)

        self.channels = 12

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        
        # Load ECG data
        file_name = self.data[item]
        ecg = load_data(file_name)
        ecg = self.transforms(ecg)

        # Get top features based on the filename
        top_feats = self.all_features[self.all_features.file_name == file_name].iloc[:, 1:].values
        # First, convert any infs to nans
        top_feats[np.isinf(top_feats)] = np.nan
        # Replace NaNs with feature means
        top_feats[np.isnan(top_feats)] = self.feats_mean[None][np.isnan(top_feats)]
        # Normalize wide features
        feats_normalized = (top_feats - self.feats_mean) / self.feats_std
        # Use zeros (normalized mean) if cannot find patient features
        if not len(feats_normalized):
            feats_normalized = np.zeros(len(self.feats_mean))[None]

        # Gather metadata
        label = self.multi_labels[item]
        age = self.age[item]
        gender = self.gender[item]
        age_gender = encode_metadata(age, gender)

        # Gather all metadata into one numpy.ndarray ([age, gender, <normalized top features>])
        feats_normalized = np.concatenate((age_gender, feats_normalized.squeeze()))

        return ecg, feats_normalized, label