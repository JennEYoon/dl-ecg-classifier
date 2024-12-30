import torch
from torch.utils.data import Dataset
import pandas as pd
from .dataset_utils import load_data, encode_metadata
from .transforms import *

def get_transforms(dataset_type, aug_type=None, precision=None, p1=None, p2=None):
    ''' Get transforms for ECG data based on the dataset type (train, validation and test).
    Transforms divided into two groups: 1) preprocessing methods and 2) rounding and augmentations. 
    Preprocessing methods are always used for all data, but augmentations are used only for the 
    training data. The rounding method can be also used for the test data.

    The rounding method is only used when the precision with a positive integer (i.e. precision is not None).

    The augmentation methods are used based on the value set in the aug_type attribute (None or a key value
    from the transforms dictionary). Now the available options are either use all the augmentation methods 
    or use them independently. The probabilities are set for the augmentations: The p1 attribute correponds 
    to the probability of an individual augmentation method, and the p2 attribute corresponds to the probability 
    of the Compose class that is used to compose several augmentations together. 

    Note that the precision and aug_type attributes are assumed to be set in the configuration yaml files.

    Four possible returns:
                            1) Only preprocessing: Crop and normalize
                            2) Preprocessing + rounding
                            3) Preprocessing + augmentations
                            4) Preprocessing + augmentations + rounding

    '''

    # We want to crop the signals approx. to 16.348 seconds
    seq_length = 4096
    normalizetype = '0-1' # 'mean-std' 

    # 1) Preprocessing transforms (done everytime)
    preprocess = {
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

    # 2) Rounding and other augmentation methods (optional)
    rounding = Round(precision) if precision is not None else None

    if dataset_type == 'train':

        # Add augmentations only to training data if the augmentation type set
        if aug_type is not None:
            
            # Probabilities for an individual augmentation (p1) and the Compose class (p2)

            transforms = {

                'all': Compose([
                    Flipx(p = p1),
                    Normalize(normalizetype),
                    Flipy(p = p1),
                    Normalize(normalizetype),
                    MultiplySine(p = p1),
                    Normalize(normalizetype),
                    MultiplyLinear(p = p1),
                    Normalize(normalizetype),
                    MultiplyTriangle(p = p1),
                    Normalize(normalizetype),
                    AddNoise(p = p1),
                    Normalize(normalizetype),
                    ResampleLinear(p = p1),
                    Normalize(normalizetype),
                    RandomStretch(p = p1),
                    Normalize(normalizetype),
                    NotchFilter(fs=250, p = p1),
                    Normalize(normalizetype),
                    Roll(p = p1),
                    Normalize(normalizetype),
                    Retype()
                ], p = p2),

                'noise': Compose([
                    AddNoise(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'roll': Compose([
                    Roll(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'flip_x': Compose([
                    Flipx(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'flip_y': Compose([
                    Flipy(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'multiply_sine': Compose([
                    MultiplySine(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'multiply_linear': Compose([
                    MultiplyLinear(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'multiply_triangle': Compose([
                    MultiplyTriangle(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'rand_stretch': Compose([
                    RandomStretch(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'resample_linear': Compose([
                    ResampleLinear(p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),

                'notch': Compose([
                    NotchFilter(fs=250, p = p1),
                    Normalize(normalizetype),
                    Retype()
                    ], p = p2),
            }

            print(f'Returning preprocessing methods {preprocess[dataset_type]}, transformers {transforms[aug_type]} and the rounding method as {rounding}')
            return preprocess[dataset_type], transforms[aug_type], rounding 
            # IF dataset_type == 'train' AND aug_type == <aug method> (training data with augmentations, possible rounding)
        else:
            print(f'Returning preprocessing methods {preprocess[dataset_type]}, transformers {aug_type} and the rounding method as {rounding}')
            return preprocess[dataset_type], aug_type,  rounding 
            # IF dataset_type == 'train' AND aug_type is None (training data without augmentations, possible rounding)
    else:
        print(f'Returning preprocessing methods {preprocess[dataset_type]}, transformers {aug_type} and the rounding method as {rounding}')
        return preprocess[dataset_type], aug_type, rounding 
        # IF dataset_type != 'train' (validation and test data)

def shorten_ecg(signal, percentage):
    _, original_length = signal.shape
    new_length = int(original_length * percentage)

    print(f'From the length {original_length} to {new_length}')

    # Randomly select a starting point
    start_index = np.random.randint(0, original_length - new_length + 1)

    # Extract the samples for each lead
    reduced_ecg_signals = signal[:, start_index:start_index + new_length]

    return reduced_ecg_signals

class ECGDataset(Dataset):
    ''' Class implementation of Dataset of ECG recordings
    
    :param path: The directory of the data used
    :type path: str
    :param preprocess: Preprocess transforms for ECG recording
    :type preprocess: datasets.transforms.Compose
    :param transform: The other transforms used for ECG recording
    :type transform: datasets.transforms.Compose
    '''

    def __init__(self, path, transforms, filter_bandwidth):
        df = pd.read_csv(path)
        self.data = df['path'].tolist()
        labels = df.iloc[:, 4:].values
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        
        self.age = df['age'].tolist()
        self.gender = df['gender'].tolist()
        self.fs = df['fs'].tolist()
        self.filter_bandwidth = filter_bandwidth
        self.preprocess, self.transforms, self.rounding = transforms
        
        if self.preprocess is not None: print('Using the following preprocessing methods:', *self.preprocess.__dict__['transforms'], sep='\n')
        if self.transforms is not None: print('Using the following augmentations:', *self.transforms.__dict__['transforms'], sep='\n')
        if self.rounding is not None: print('Using the following rounding method:', self.rounding)

        self.channels = 12
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file_name = self.data[item]
        ecg = load_data(file_name)

        # SHORTEN THE ECGS
        #ecg = shorten_ecg(ecg, 0.5)

        ####################################################
        #
        # Processing the ECGs in the following order
        #       1) Filtering
        #       2) Cropping and normalization (and retyping)
        #       3) Augmentations / Rounding
        #
        # Only cropping and normalization is used by default, 
        # every other preprocessing method needs to be set separately
        #
        ####################################################

        # - Apply filter
        if self.filter_bandwidth is not None:

            ecg_fs_orig = self.fs[item]
            lf, hf = self.filter_bandwidth
            ecg_fs_new = 500

            spline = Spline_interpolation(fs_new = ecg_fs_new, fs_old = ecg_fs_orig)
            ecg = spline(ecg)

            # If both frequencies are set, use the Bandpass filter ([lf, hf])
            # Either use the Highpass filter ([lf, None]) or the Lowpass filter ([None, hf]) based on the frequency set
            if lf and hf:
                print(f'Bandpass filter used with the cutoff frequencies of {lf} and {hf} (fs={ecg_fs})!')
                bpf = BandPassFilter(fs=ecg_fs, lf=lf, hf=hf)
                ecg = bpf(ecg)
            elif lf is None and hf:
                print(f'Lowpass filter used with the cutoff frequency of {hf}!')
                lpf = LowPassFilter(fs=ecg_fs, cf=hf)
                ecg = lpf(ecg)
            
        # - Preprocessing methods (Crop and normalize)
        ecg = self.preprocess(ecg)

        # - Use augmentations
        if self.transforms is not None:
            ecg = self.transforms(ecg)

        # - Round the ECG signals
        if self.rounding is not None:
            ecg = self.rounding(ecg)

        label = self.multi_labels[item]
        age = self.age[item]
        gender = self.gender[item]
        age_gender = encode_metadata(age, gender)
        return ecg, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()