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

    # Seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Load yaml
    print('Loading arguments from', os.path.basename(file))
    args = load_yaml(file)

    # Path where the needed CSV file exists
    csv_root = os.path.join(os.getcwd(), 'data', 'split_csvs', args.csv_path)

    # Update paths
    args.test_path = os.path.join(csv_root, args.test_file)
    args.yaml_file_name = os.path.splitext(file)[0]
    args.yaml_file_name = os.path.basename(args.yaml_file_name)
    
    # Output directory based on if multiple yaml files are run or only one
    args.output_dir = os.path.join(os.getcwd(),'experiments', model_save_dir, args.yaml_file_name) if multiple else os.path.join(os.getcwd(),'experiments', args.yaml_file_name)
        
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check whether the attributes considering ECG processing are set in the yaml file
    args.precision = int(args.precision) if args.precision and int(args.precision) >= 0 else None
    args.filter_bandwidth = args.filter_bandwidth if args.filter_bandwidth else None

    # If cutoff frequencies set, correct the types of each element in the list [<cf>, <cf>]
    if args.filter_bandwidth is not None:
        a, b = args.filter_bandwidth
        a = float(a) if isinstance(a, (int, float)) else None
        b = float(b) if isinstance(b, (int, float)) else None
        args.filter_bandwidth = [a, b]

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
        args.logger.info('{}: {}'.format(k, v))
    args.logger.info('-'*10) 

    args.logger.info('Making predictions...')

    pred = Predicting(args)
    pred.setup()
    pred.predict()

    
def read_multiple_yamls(path, csv_root):
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
        read_yaml(file, csv_root, model_save_dir, True)


if __name__ == '__main__':
        
    # Load args
    given_arg = sys.argv[1]
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