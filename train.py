import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import time, datetime
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from fastai import *
from fastai.vision import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from Data.dataloader import data_class, dataloader
from Utils.monitor import Monitor_OOD

def print_and_write(text, log_txt, mode='a'):
    print(text)
    with open(log_txt, mode) as f:
        f.write(text + '\n')

def train_one_epoch(args, writer, model, data_loader, optimizer, epoch):
    model.train()
    size = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_output=[]
    epoch_features=[]
    epoch_targets=[]
    with tqdm(total=len(data_loader),ncols=90) as pbar:
        pbar.set_description('Training Epoch %d' % epoch)
        for images, targets in data_loader:
            pbar.update(1)
            images = images.to(args.device)
            targets = targets.to(args.device).float()
            if args.mx:
                lam = np.random.beta(1, 1)
                batch_size = images.size()[0]
                index = torch.randperm(batch_size).to(images.device.type)

                images = lam * images + (1 - lam) * images[index, :]
                # y_a, y_b = y, y[index]
                targets = lam * targets + (1 - lam) * targets[index, :]

            # Pass the inputs through the CNN model.
            features, outputs = model(images)
            epoch_output.append(outputs)
            # epoch_features.append(features)
            epoch_targets.append(targets)

            loss = model.criterion(outputs, targets, args.epochs, args.num_classes, 10, args.device)

            _, preds = torch.max(outputs, 1)
            targets = torch.argmax(targets, dim=1)

            running_loss += loss.item()*images.size(0)

            running_corrects += torch.sum(preds == targets)

            size += images.size(0)
            pbar.set_postfix(loss=running_loss / size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.debug:
                break

    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size
    epoch_output = torch.cat(epoch_output, dim=0)
    # epoch_features = torch.cat(epoch_features, dim=0)
    epoch_targets = torch.cat(epoch_targets, dim=0)

    # epoch_metrics = model.infer_metrics(epoch_features)

    if writer is not None:
        writer.add_scalars('Loss/train', {'loss': epoch_loss}, epoch)

    return epoch_loss, epoch_acc, epoch_output, epoch_features, epoch_targets

@torch.no_grad()
def evaluate(args, writer, model, data_loader, epoch, ood=False):
    model.eval()
    size = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_output=[]
    epoch_features=[]
    epoch_targets=[]
    with tqdm(total=len(data_loader),ncols=90) as pbar:
        if ood:
            pbar.set_description('Evaluate OOD: %d' % epoch)
        else:
            pbar.set_description('Evaluate ID: %d' % epoch)
        for images, targets in data_loader:
            pbar.update(1)
            images = images.to(args.device)
            targets = targets.to(args.device).float()

            # Pass the inputs through the CNN model.
            features, outputs = model(images)
            epoch_output.append(outputs)
            # epoch_features.append(features)
            epoch_targets.append(targets)

            loss = model.criterion(outputs, targets, args.epochs,args.num_classes, 10, args.device)

            _, preds = torch.max(outputs, 1)
            targets = torch.argmax(targets, dim=1)
            
            # Calculate the batch loss.
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == targets)
            size += images.size(0)

    # end epoch
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size 
    epoch_output = torch.cat(epoch_output, dim=0)
    # epoch_features = torch.cat(epoch_features, dim=0)
    epoch_targets = torch.cat(epoch_targets, dim=0)
    # epoch_metrics = model.infer_metrics(epoch_features)
    
    if writer is not None and not ood:
        writer.add_scalars('Loss/valid', {'loss': epoch_loss}, epoch)
        writer.add_scalars('Accuracy/valid', {'acc': epoch_acc}, epoch)

    return epoch_loss, epoch_acc, epoch_output, epoch_features, epoch_targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training on ISIC datasent')
    parser.add_argument('--num-workers', type=int, default=8, help='number of loader workers')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device {cuda:0, cpu}')
    parser.add_argument('--checkpoint_path',type=str, default=None, help='Checkpoint path for resuming the training')
    parser.add_argument('--debug',type=int, default=0, help='Debug mode: 1')
    parser.add_argument('--snapshot', type=int, default=0, help='save snapshot every epoch')
    parser.add_argument('--img-size', type=int, default=256, help='Training image size to be passed to the network')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--scheduler', type=int, default=1, help='momentum')
    parser.add_argument('--log-dir', default='', help='where to store results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--dataset',type=str, default='isic', help='isic/bone/retinal')
    parser.add_argument('--norm', type=int, default=0, help='use openood norm')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch for resuming the training')
    parser.add_argument('--mx', type=int, default=0, help='mixup')

    parser.add_argument('--name', type=str, default=None, help='Checkpoint path name')
    parser.add_argument('--backbone', type=str, default='ResNet18', help='ResNet18/ResNet34')
    parser.add_argument('--method', default='evidencenet',required=True)
    parser.add_argument('--reg_kl', type=float, default=1)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--metric', type=str)

    args = parser.parse_args()
    # torch.manual_seed(args.seed)

    args.log_dir = './Results/runs_openset_' + args.dataset + '/' + str(args.method) + '/' + str(args.name)
    writer = SummaryWriter(log_dir = args.log_dir)
    output_dir = Path(writer.log_dir)
    log_txt = '.' / output_dir / 'log.txt'

    
    options = vars(args)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
    args.num_classes = data_class(dataset=args.dataset)
    if args.norm:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_dl = dataloader(dataset=args.dataset,data_mode='train',batch_size=args.batch_size,num_workers=args.num_workers, 
                          image_size=args.img_size,mean=mean,std=std)
    valid_dl = dataloader(dataset=args.dataset,data_mode='valid',batch_size=args.batch_size,num_workers=args.num_workers,
                          image_size=args.img_size,mean=mean,std=std)
    ood_dl = dataloader(dataset=args.dataset,data_mode='ood',  batch_size=args.batch_size,num_workers=args.num_workers,
                        image_size=args.img_size,mean=mean,std=std)

    print_and_write('Dataset Information'.center(90,'='),log_txt,'w')
    print_and_write('Current dataset: {}'.format(args.dataset),log_txt)
    print_and_write('Number of classes: {}'.format(args.num_classes),log_txt)
    print_and_write('train: {}, valid: {}, ood: {}'.format(len(train_dl.dataset),len(valid_dl.dataset),len(ood_dl.dataset)),log_txt)

    
    # Initialize the model
    print_and_write('Building model'.center(90,'='),log_txt)
    
    # baseline
    if args.method == 'convnet':
        from Methods.ConvNet import ConvNet
        model = ConvNet(args.backbone, args.num_classes)
    elif args.method == 'evidencenet':
        from Methods.EvidenceNet import EvidenceNet
        model = EvidenceNet(args.backbone, args.num_classes, args.reg_kl)
    elif args.method == 'evidencenetwokl':
        from Methods.EvidenceNetwoKL import EvidenceNetwoKL
        model = EvidenceNetwoKL(args.backbone, args.num_classes)
    elif args.method == 'priornet':
        from Methods.PriorNet import PriorNet
        model = PriorNet(args.backbone, args.num_classes)
    elif args.method == 'repriornet':
        from Methods.PriorNet_Re import PriorNet_Re

        model = PriorNet_Re(args.backbone, args.num_classes)
    elif args.method == 'reconevidencenet':
        from Methods.ReconEvidenceNet import ReconEvidenceNet
        model = ReconEvidenceNet(args.backbone, args.num_classes)

    elif args.method == 'edlmse':
        from Methods.EvidenceNet_MSE import EvidenceNet_MSE
        model = EvidenceNet_MSE(args.backbone, args.num_classes)
    elif args.method == 'edlmsewokl':
        from Methods.EvidenceNet_MSEwoKL import EvidenceNet_MSEwoKL
        model = EvidenceNet_MSEwoKL(args.backbone, args.num_classes)
    elif args.method == 'iedl':
        from Methods.IEDL import I_EvidenceNet
        model = I_EvidenceNet(args.backbone, args.num_classes)
    elif args.method == 'redl':
        from Methods.REDL import R_EvidenceNet
        model = R_EvidenceNet(args.backbone, args.num_classes)

    elif args.method == 'rolenet':
        from Methods.ROLENet import ROLENet
        model = ROLENet(args.backbone, args.num_classes)
    else:
        raise NotImplementedError('method not found')
    
    args.metric = model.metrics
    options.update(
        {
            'use_gpu': use_gpu,
            'num_classes': args.num_classes,
            'loss': args.loss,
            'metric': args.metric,
        }
    )
    model = model.to(args.device)
    n_parameters = sum(p.numel() for p in model.parameters())
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    print_and_write('Current method: {}'.format(args.method),log_txt)
    print_and_write('Using backbone/loss: {}/{}'.format(args.backbone, args.loss),log_txt)
    print_and_write('Using metric: {}'.format(args.metric),log_txt)
    print_and_write('Batch size: {}/ Learning rate: {}/ Weight decay: {}'.format(args.batch_size,args.lr,args.weight_decay),log_txt)
    
    print_and_write("Start Training".center(90,"="),log_txt)
    print_and_write(str(args),log_txt)


    checkpoint_path = Path(os.path.join(output_dir,'checkpoints'))
    os.makedirs(checkpoint_path, exist_ok=True)
    monitor = Monitor_OOD(args, checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        #Train one epoch
        train_loss, train_acc, train_output, train_features, train_target = train_one_epoch(args, writer, model, train_dl, optimizer, epoch)
        # evaluate
        valid_loss, valid_acc, valid_id_output, valid_id_features, valid_id_targets = evaluate(args, writer, model, valid_dl, epoch, ood=False)
        _, _, valid_ood_output, valid_ood_features, _ = evaluate(args, writer, model, ood_dl, epoch, ood=True)
        
        # save checkpoint
        monitor.update_monitor(train_loss, valid_loss, valid_acc, 
                               valid_id_features, valid_id_output, valid_ood_features, valid_ood_output, 
                               model, optimizer, epoch)
        monitor.gen_epoch_report(args,log_txt,epoch)

        if (epoch > 0.2 *args.epochs) and args.scheduler:
            scheduler.step(valid_loss)
        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print_and_write('Epoch training time {}\n'.format(epoch_total_time_str),log_txt)

        if args.snapshot:
            snapshot_path = checkpoint_path / 'snapshot_{}.pth'.format(epoch)
            torch.save(model, snapshot_path)
        if args.debug:
            break

    if writer is not None: writer.close()