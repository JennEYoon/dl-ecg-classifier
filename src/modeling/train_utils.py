import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .optimizer import NoamOpt
from .models.ctn import CTN
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle

class Training(object):
    def __init__(self, args):
        self.args = args
  
    def setup(self):
        '''Initializing the device conditions, datasets, dataloaders, 
        model, loss, criterion and optimizer
        '''
        
        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = self.args.device_count
            self.args.logger.info('using {} gpu(s)'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))

        # Load the datasets       
        training_set = ECGDataset(self.args.train_path, get_transforms('train'), self.args.all_features)
        self.train_dl = DataLoader(training_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=True)
        channels = training_set.channels

        if self.args.val_path is not None:
            validation_set = ECGDataset(self.args.val_path, get_transforms('val'), self.args.all_features)
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    drop_last=True)

        self.model = CTN(in_channel=channels, out_channel=len(self.args.labels))

        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # If more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model) 
        
        # Optimizer
        self.optimizer = NoamOpt(256, 1, 4000, torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        self.model.to(self.device)
        
        
    def train(self):
        ''' PyTorch training loop
        '''
        
        self.args.logger.info('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(self.model).__name__, 
               type(self.optimizer).__name__,
               self.optimizer.optimizer.param_groups[0]['lr'], 
               self.args.epochs, 
               self.device))
        
        # Add all wanted history information
        history = {}
        history['train_loss'] = []
        history['train_micro_auroc'] = []
        history['train_micro_avg_prec'] = []  
        history['train_macro_auroc'] = []
        history['train_macro_avg_prec'] = [] 

        if self.args.val_path is not None:
            history['val_csv'] = self.args.val_path
            history['val_loss'] = []
            history['val_micro_auroc'] = []
            history['val_micro_avg_prec'] = []
            history['val_macro_auroc'] = []
            history['val_macro_avg_prec'] = []

        history['labels'] = self.args.labels
        history['epochs'] = self.args.epochs
        history['batch_size'] = self.args.batch_size
        history['lr'] = self.args.lr
        history['optimizer'] = self.optimizer
        history['criterion'] = "F.binary_cross_entropy_with_logits()"
        history['train_csv'] = self.args.train_path
        
        start_time_sec = time.time()
        
        for epoch in range(1, self.args.epochs+1):
            
            # --- TRAIN ON TRAINING SET -----------------------------
            self.model.train()            
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device) # , device=torch.device('cuda:0')
            logits_prob_all = torch.tensor((), device=self.device)
            
            batch_loss = 0.0
            batch_count = 0
            step = 0
            
            for batch_idx, (ecg, feats_normalized, labels) in enumerate(self.train_dl):

                # Train instances use only one window
                ecg = ecg.float().to(self.device) # ECGs
                feats_normalized = feats_normalized.float().to(self.device) # metadata: age, gender and top handcrafted feautres
                labels = labels.float().to(self.device) # diagnoses

                self.optimizer.optimizer.zero_grad()
                logits = self.model(ecg, feats_normalized)
                logits_prob = logits.sigmoid().data
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss_tmp = loss.item()

                labels_all = torch.cat((labels_all, labels), 0)
                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)                    
                
                train_loss += loss_tmp
                loss.backward()
                self.optimizer.step()

                # self.args.logger.infoing training information
                if step % 100 == 0:
                    batch_loss += loss_tmp
                    batch_count += ecg.size(0)
                    batch_loss = batch_loss / batch_count
                    self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                        epoch, 
                        batch_idx * len(ecg), 
                        len(self.train_dl.dataset), 
                        batch_loss
                    ))

                    batch_loss = 0.0
                    batch_count = 0
                step += 1

            train_loss = train_loss / len(self.train_dl.dataset)            
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)

            # Add information for training history
            history['train_loss'].append(train_loss)
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)

            self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train macro auroc: {:<6.2f} '.format( 
                epoch, 
                self.args.epochs, 
                train_loss, 
                train_macro_auroc))

            # --- EVALUATE ON VALIDATION SET ------------------------------------- 
            if self.args.val_path is not None:
                self.model.eval()
                val_loss = 0.0  
                labels_all = torch.tensor((), device=self.device)
                logits_prob_all = torch.tensor((), device=self.device)  
                
                for ecg, feats_normalized, labels in self.val_dl:
                    ecg = ecg.float().to(self.device) # ECGs
                    feats_normalized = feats_normalized.float().to(self.device) # handcrafted ECG features + age + sex
                    labels = labels.float().to(self.device) # diagnoses 
                    
                    with torch.no_grad():  
                        logits = self.model(ecg, feats_normalized)
                        loss = F.binary_cross_entropy_with_logits(logits, labels)
                        logits_prob = logits.sigmoid().data
                        val_loss += loss.item() * ecg.size(0)  

                        labels_all = torch.cat((labels_all, labels), 0)
                        logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                val_loss = val_loss / len(self.val_dl.dataset)
                val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
            
                history['val_loss'].append(val_loss)
                history['val_micro_auroc'].append(val_micro_auroc)
                history['val_micro_avg_prec'].append(val_micro_avg_prec)         
                history['val_macro_auroc'].append(val_macro_auroc)  
                history['val_macro_avg_prec'].append(val_macro_avg_prec)
        
                self.args.logger.info('                val loss:  {:<6.2f}   val macro auroc: {:<6.2f}'.format(
                    val_loss,
                    val_macro_auroc))


            # Create ROC Curves at the beginning, middle and end of training
            if epoch == 1 or epoch == self.args.epochs/2 or epoch == self.args.epochs:
                roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.args.roc_save_dir)

            # =====================================

             # Save a model at every 5th epoch
            if epoch in list(range(self.args.epochs)[0::5]):
                self.args.logger.info('Saved model at the epoch {}!'.format(epoch))
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    
                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '_e' + str(epoch) + '.pth')
                torch.save(model_state_dict, model_savepath)

            # Save trained model (.pth), history (.pickle) and validation logits (.csv) () after the last epoch
            if epoch == self.args.epochs:
                
                self.args.logger.info('Saving the model, training history and validation logits...')
                    
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    
                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name  + '.pth')
                torch.save(model_state_dict, model_savepath)
                
                # -- Save history
                history_savepath = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_train_history.pickle')
                with open(history_savepath, mode='wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
                # -- Save the logits from validation if used, either save the logits from the training phase
                if self.args.val_path is not None:
                    self.args.logger.info('- Validation logits and labels saved')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                               self.args.yaml_file_name + '_val_logits.csv') 
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_val_labels.csv') 
                    # Use filenames as indeces
                    filenames = [os.path.basename(file) for file in self.validation_files]

                else:
                    self.args.logger.info('- Training logits and actual labels saved (no validation set available)')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                               self.args.yaml_file_name + '_train_logits.csv') 
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_train_labels.csv') 
                    filenames = None
                
                # Save logits and corresponding labels
                labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
                labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
                labels_df.to_csv(labels_all_csv_path, sep=',')

                logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
                logits_df.to_csv(logits_csv_path, sep=',')

            del logits_prob_all
            del labels_all
            torch.cuda.empty_cache()

        # END OF TRAINING LOOP        
        
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))
        self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))