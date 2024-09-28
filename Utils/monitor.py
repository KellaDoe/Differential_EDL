from Utils.metrics import get_metric
from Utils.evaluation import metric_ood
from pathlib import Path
import os
import torch

def print_and_write(text, log_txt):
    print(text)
    with open(log_txt, 'a') as f:
        f.write(text + '\n')

class Monitor_OOD:
    def __init__(self, args, checkpoint_path):
        self.auroc_monitor = {}
        self.cp_path = {}

        self.loss_monitor = {'train': 0, 'valid': 0}
        self.acc_monitor = {'valid': 0, 'best': 0, 'epoch': 0}
        self.cp_path['acc'] = Path(os.path.join(checkpoint_path,'acc_checkpoints.pth'))


        for key in args.metric:
            self.auroc_monitor[key] = {'valid': 0, 'best': 0, 'epoch': 0}
            self.cp_path[key] = Path(os.path.join(checkpoint_path,'{}_checkpoints.pth'.format(key)))

        
    def update_monitor(self, train_loss, valid_loss, valid_acc, 
                       valid_feature_id, valid_out_id, valid_feature_ood, valid_out_ood, 
                       model, optimizer, epoch):
        self.loss_monitor['train'] = train_loss
        self.loss_monitor['valid'] = valid_loss
        self.acc_monitor['valid'] = valid_acc

        if valid_acc > self.acc_monitor['best']:
            self.acc_monitor['best'] = valid_acc
            self.acc_monitor['epoch'] = epoch
            print('Save checkpoint {} with {} accuracy in epoch {}'.format(self.cp_path['acc'],valid_acc,epoch))
            torch.save(model, self.cp_path['acc'])
        
        # process logits accroding to method
        valid_out_id = model.infer_logits(valid_feature_id, valid_out_id)
        valid_out_ood = model.infer_logits(valid_feature_ood, valid_out_ood)

        for key in self.auroc_monitor.keys():
            # msp and pent are based on probabilities and others are based on logits
            if key == 'msp' or key == 'entropy':
                valid_prob_id = model.infer_probs(valid_out_id)
                valid_prob_ood = model.infer_probs(valid_out_ood)
            else:
                valid_prob_id = valid_out_id
                valid_prob_ood = valid_out_ood
            id_metric = get_metric(key)(valid_prob_id)
            ood_metric = get_metric(key)(valid_prob_ood)

            # msp for id is higher, and others for ood should be higher
            if key == 'msp':
                epoch_metric = metric_ood(id_metric, ood_metric, verbose=False)['Bas']['AUROC']
            else:
                epoch_metric = metric_ood(ood_metric, id_metric, verbose=False)['Bas']['AUROC']
            
            self.auroc_monitor[key]['valid'] = epoch_metric
            if epoch_metric > self.auroc_monitor[key]['best']:
                self.auroc_monitor[key]['best'] = epoch_metric
                self.auroc_monitor[key]['epoch'] = epoch
                print('Save checkpoint {} with {} {} auroc in epoch {}'.format(self.cp_path[key],epoch_metric,key,epoch))
                torch.save(model, self.cp_path[key])
                
    def gen_epoch_report(self, args, log_txt, epoch):

        print_and_write('', log_txt)
        print_and_write('Valid Report in epoch {}'.format(epoch).center(90,'='), log_txt)

        print_and_write('| Dataset = {}, Method = {}, name = {}, Best acc = {:.4f}'.format(args.dataset, args.method, args.name, self.acc_monitor['best'].item()).ljust(89,' ')+'|', log_txt)
        print_and_write('Training Loss:'.center(90,'-'), log_txt)
        print_and_write('| Train_loss = {:.4f}'.format(self.loss_monitor['train']).ljust(44,' ') + '| Valid_loss = {:.4f}'.format(self.loss_monitor['valid']).ljust(45,' ')+'|', log_txt)
        
        print_and_write('ID Meric:'.center(90,'-'), log_txt)
        
        print_and_write('| valid_acc = {:.4f}'.format(self.acc_monitor['valid']).ljust(44,' ') +
                        '| best_valid_acc = {:.4f} in epoch {}'.format(self.acc_monitor['best'],self.acc_monitor['epoch']).ljust(45,)+'|', log_txt)

        print_and_write('OOD Metric:'.center(90,'-'), log_txt)

        for key in args.metric:
            print_and_write('| valid_auroc_{} = {:.4f}'.format(key,self.auroc_monitor[key]['valid']).ljust(44,' ') + 
                            '| best_auroc_{} = {:.4f} in epoch {}'.format(key,self.auroc_monitor[key]['best'],self.auroc_monitor[key]['epoch']).ljust(45,' ')+'|', log_txt)


        print_and_write(''.center(90,'='), log_txt)



        