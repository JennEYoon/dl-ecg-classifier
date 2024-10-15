import os, pandas as pd, logging
from tqdm import tqdm
from src.features.features import *
from src.dataloader.dataset_utils import load_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

""" The provided code is to extract ECG features from a given lead and store them in a dataframe. 
In https://www.cinc.org/archives/2020/pdf/CinC2020-107.pdf, Natarajan et al. extracted over 300 features from the lead II, 
including morphological features and heart rate variability features. They selected the top 20 features using Random Forest 
for the transformer network to utilize as handcrafted features.

There are three groups of features than can be used: ['full_waveform_statistics', 'heart_rate_variability_statistics', 'template_statistics'].
The codes are found in `src/features/` and initially provided by Sebastian D. Goodfellow in the PhysioNet Challenge 2017. The codes are modified by Jonathan Rubin
for the implementation of the transformer neural network (the model that is also found from this branch of the repository).
"""

# The location of the ECG data
data_path = 'data/smoke_data/'

# Where to store the feature dataframes
save_path = 'data/features/feature_test'
os.makedirs(save_path, exist_ok=True)

# Logging purposes (list the files from which the features cannot be computed)
logging.basicConfig(filename='feat_computation.log', encoding='utf-8', level=logging.DEBUG)

# Wanted features
feature_list = ['full_waveform_statistics', 'heart_rate_variability_statistics', 'template_statistics']

fs = 500  # "As recordings from separate hospitals could have different sampling rates, we first upsample or downsample each recording to 500Hz. (Natarajan et al. 2020)"
filter_bandwidth = [3, 45] # We apply an FIR (finite impulse response) bandpass filter with bandwidth between 3 - 45 Hz. (Natarajan et al. 2020)"
channel = 1 # "We initially extracted over 300 ECG features from lead II [...]. (Natarajan et al. 2020)" | PhysioNet 2021: The lead order in the hea files: I-III, aVR, avL, avF, V1-V6

# Iterate over data folders
for i, dir in enumerate(os.listdir(data_path)):
    #if i == 1: break

    full_path = os.path.join(data_path, dir)
    print(f'Computing features from {full_path}')

    # Gather all ECGs from the source (.mat OR .h5)
    all_ecg_files = [f for f in os.listdir(full_path) if '.mat' in f or '.h5' in f]

    all_features = pd.DataFrame()

    # Iterate over ECGs and extract features from the given channel
    for signal_i in tqdm(range(len(all_ecg_files))):
        signal_raw = load_data(os.path.join(full_path, all_ecg_files[signal_i]))

        feat_computator = Features(filename = all_ecg_files[signal_i], 
                                    data=signal_raw,    
                                    fs=fs, 
                                    feature_groups=feature_list)
        
        try:
            # Try to compute features
            feat_computator.calculate_features(filter_bandwidth, show=False,
                                            channel=1, normalize=True, polarity_check=True,
                                            template_before=0.25, template_after=0.4)
            all_features = pd.concat([all_features, feat_computator.get_features()], axis=0)
        
        except:
            # Get a list of ECGs from which features could not be computed
            logging.info(os.path.join(full_path, all_ecg_files[signal_i]))

        all_features.to_csv((os.path.join(save_path, dir + f'_ch{channel}.csv')), index=None)