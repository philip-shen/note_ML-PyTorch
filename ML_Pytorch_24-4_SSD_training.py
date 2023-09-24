'''
a-PyTorch-Tutorial-to-Object-Detection/train.py

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py    
'''
import os, sys, time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
import json, argparse, pathlib

sys.path.append('./libs')

#from datasets import PascalVOCDataset
import datasets 
#from utils import *
import utils
from logger_setup import *
import lib_misc

strabspath=os.path.abspath(sys.argv[0])
strdirname=os.path.dirname(strabspath)
str_split=os.path.split(strdirname)
prevdirname=str_split[0]
dirnamelog=os.path.join(strdirname,"logs")
dirname_test_wav= os.path.join(strdirname,"test_wav")
dirnametest= os.path.join(strdirname,"test")

def est_timer(start_time):
    time_consumption, h, m, s= lib_misc.format_time(time.time() - start_time)         
    msg = 'Time Consumption: {}.'.format( time_consumption)#msg = 'Time duration: {:.2f} seconds.'
    logger.info(msg)

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, 
                                    weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        #print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        logger.info(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = datasets.PascalVOCDataset(data_folder,
                                                split='train',
                                                keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, 
                                               num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
    logger.info(f'epochs: {epochs}')
    logger.info(f'decay_lr_at: {decay_lr_at}')

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            utils.adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        utils.save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses = utils.AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            utils.clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            #print('Epoch: [{0}][{1}/{2}]\t'
            #      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
            #                                                      batch_time=batch_time,
            #                                                      data_time=data_time, loss=losses))
            
            logger.info('\nEpoch: [{0}][{1}/{2}], Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f}),Data Time: {data_time.val:.3f} ({data_time.avg:.3f}),Loss: {loss.val:.4f} ({loss.avg:.4f})'\
                        .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,loss=losses))
            
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def make_datalists(voc07_path='', voc12_path='', opt_verbose='OFF'):
    utils.create_data_lists(voc07_path,                            
                            voc12_path,
                            output_folder= './', 
                            opt_verbose=opt_verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_24-4_SSD_training')
    parser.add_argument('--conf', type=str, default='config.json', help='Config json')
    args = parser.parse_args()
    logger_set(strdirname)
    
    t0 = time.time()
    local_time = time.localtime(t0)
    msg = 'Start Time is {}/{}/{} {}:{}:{}'
    logger.info(msg.format( local_time.tm_year,local_time.tm_mon,local_time.tm_mday,\
                            local_time.tm_hour,local_time.tm_min,local_time.tm_sec))
    
    opt_verbose = 'On'
    json_file= args.conf
    json_path_file = pathlib.Path(strdirname)/json_file
    
    if (not os.path.isfile(json_file))  :
        raise Exception( f'Please check json file: {json_file}  if exist!!! ')
    json_data = json.load(json_path_file.open())

    # Data parameters
    data_folder = './'  # folder with data files
    keep_difficult = True  # use objects considered difficult to detect?

    home = os.path.expanduser("~")    
    voc12_path = f'{home}/infinicloud/{json_data["Config"][3]["path_trainval"]}'

    voc07_path=f'{home}/projects/VOCDetection/2007/trainval/VOCdevkit/VOC2007/'
                            #voc12_path,#'/media/ssd/ssd data/VOC2012'
    voc12_path = f'{home}/projects/VOCDetection/2012/trainval/VOCdevkit/VOC2012/'
    make_datalists(voc07_path=voc07_path, voc12_path=voc12_path)
        
    # Model parameters
    # Not too many here since the SSD300 has a very specific structure
    n_classes = len(utils.label_map)  # number of different types of objects
    logger.info(f'n_classes: {n_classes}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True
    
    # Learning parameters
    checkpoint = None  # path to model checkpoint, None if none
    batch_size = json_data["Config"][3]["batch_size"]#8  # batch size
    iterations = json_data["Config"][3]["iterations"]#120000  # number of iterations to train
    workers = json_data["Config"][3]["workers"]#4  # number of workers for loading data in the DataLoader
    print_freq = json_data["Config"][3]["print_freq"]#200  # print training status every __ batches
    lr = json_data["Config"][3]["lr"]#1e-3  # learning rate
    decay_lr_at = json_data["Config"][3]["decay_lr_at"]#[80000, 100000]  # decay learning rate after these many iterations
    decay_lr_to = json_data["Config"][3]["decay_lr_to"]#0.1  # decay learning rate to this fraction of the existing learning rate
    momentum = json_data["Config"][3]["momentum"]#0.9  # momentum
    weight_decay = json_data["Config"][3]["weight_decay"]#5e-4  # weight decay
    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    main()

    est_timer(t0)