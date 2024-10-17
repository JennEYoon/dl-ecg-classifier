import numpy as np, os, sys
import torch
import random
import pandas as pd
from utils import load_yaml
from src.modeling.predict_utils import Predicting
import logging

def read_yaml(file, model_save_dir='', multiple=False):
    ''' Read a given yaml and perform classification predictions.
    Evaluate the predictions.
    
    :param file: Absolute path for the yaml file wanted to read
    :type file: str
    :param csv_root: Absolute path for the csv file
    :type csv_root: str
    :param model_save_dir: If multiple yamls are read, the model directory is  
                           a subdirectory of the 'experiments' directory
    :type model_save_dir: str
    :param multiple: Check if multiple yamls are read
    :type multiple: boolean
    '''
    
    # Load yaml
    args = load_yaml(file)
    
    # Update paths
    feature_root = os.path.join(os.getcwd(), 'data', 'features', args.feature_path)
    csv_root = os.path.join(os.getcwd(), 'data', 'split_csvs', args.csv_path)
    args.test_path = os.path.join(csv_root, args.test_file)
    args.yaml_file_name = os.path.splitext(file)[0]
    args.yaml_file_name = os.path.basename(args.yaml_file_name)
    
    # Output directory based on if multiple yaml files are run or only one
    args.output_dir = os.path.join(os.getcwd(),'experiments', model_save_dir, args.yaml_file_name) if multiple else os.path.join(os.getcwd(),'experiments', args.yaml_file_name)
    
    # Make sure the directory for outputs exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)  

    # Find the trained model from the ´experiments´ directory as it should be saved there
    doublicate_flag = 0
    for root, _, files in os.walk(os.path.join(os.getcwd(), 'experiments')):
        if args.model in files:
            args.model_path = os.path.join(root, args.model)
            doublicate_flag += 1

    # Check if model_path never set, i.e., the trained model was found
    if not hasattr(args, 'model_path'):
        raise AttributeError('No path found for the model. Check if you have trained one.')

    if doublicate_flag > 1:
        raise Exception('There are more than one similarly named models in the experiments directory. \
                         You should not have duplicated names so that you use correct models!')

    # Load labels
    args.labels = pd.read_csv(args.test_path, nrows=0).columns.tolist()[4:]

    # ================================ #
    # ===== HANDCRAFTED FEATURES ===== #
    # ================================ #
    # Natarajan et al. computed over 300 features from lead II which they included to their deep transformer neural network.
    # Importances of the features by RandomForest are stored in the file `features_by_importance.npy`.
    # Only TOP20 features were used => let's extract the names and the computed features.
    n_features = 20 # How many top features

    # First, find the names of all the features (also exclude useless names from the list)
    feature_names = list(np.load(os.path.join('data', 'features_by_importance.npy')))
    feature_names.remove('full_waveform_duration')
    feature_names.remove('Age')
    feature_names.remove('Gender_Male')

    # Only include TOP<n_features> features
    feature_names = feature_names[:n_features]
    feature_names.append('file_name')

    # Then, load and concat the TOP<n_features> features into one dataframes from all the datasets
    args.all_features = pd.concat([pd.read_csv(os.path.join(feature_root, df), usecols=feature_names) for df in os.listdir(feature_root) if df.endswith('csv')]).reset_index(drop=True)
    new_names = [os.path.basename(name) for name in args.all_features.file_name]
    args.all_features.file_name = new_names # Cut only the file names from the full paths

    if len(args.all_features) == 0:
        raise Exception('The features weren´t loaded correctly!')
    
    if not set(feature_names).issubset(args.all_features.columns):
        raise Exception('Not all necessary features/columns are found from the feature dataframe!')

    # ================================ #
    # ================================ #
    
    # For logging purposes
    logs_path = os.path.join(args.output_dir, args.yaml_file_name + '_predict.log')
    logging.basicConfig(filename=logs_path, 
                        format='%(asctime)s %(message)s', 
                        filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S') 
    args.logger = logging.getLogger(__name__) 
    args.logger.setLevel(logging.DEBUG) 
    
    args.logger.info('Arguments:')
    args.logger.info('-'*10)
    for k, v in args.__dict__.items():
        if 'features' in k:
            args.logger.info('{} (shape={}): {}'.format(k, v.shape, v.columns.tolist()))
        else:
            args.logger.info('{}: {}'.format(k, v))
    args.logger.info('-'*10) 

    args.logger.info('Making predictions...')

    pred = Predicting(args)
    pred.setup()
    pred.predict()

    
def read_multiple_yamls(path):
    ''' Read multiple yaml files from the given directory
    
    :param directory: Absolute path for the directory
    :type path: str
    '''
    # All yaml files
    yaml_files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    
    # Save all trained models in the same subdirectory in the 'experiments' directory
    dir_name = os.path.basename(path)
    model_save_dir = os.path.join(os.getcwd(),'experiments', dir_name) 

    # Running the yaml files and training models for each
    for file in yaml_files:
        read_yaml(file, model_save_dir, True)


if __name__ == '__main__':
        
    # Seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ----- Set the path here! -----
    
    # ------------------------------

    # Load args
    given_arg = sys.argv[1]
    print('Loading arguments from', given_arg)
    arg_path = os.path.join(os.getcwd(), 'configs', 'predicting', given_arg)
    
    # Check if a yaml file or a directory given as an argument
    # Possible multiple yamls for prediction and evaluation phase!
    if os.path.exists(arg_path):

        if 'yaml' in given_arg:
            # Run one yaml
            read_yaml(arg_path)
        else:
            # Run multiple yamls from a directory
            read_multiple_yamls(arg_path)

    else:
        raise Exception('No such file nor directory exists! Check the arguments.')

    print('Done.')