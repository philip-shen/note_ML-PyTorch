'''
a-PyTorch-Tutorial-to-Object-Detection/eval.py

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py
'''
'''
File "./libs/utils.py", line 262, in calculate_mAP
original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
'''
            
from tqdm import tqdm
from pprint import PrettyPrinter
import torch
import os, sys, time
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


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = utils.calculate_mAP(device, det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    #print('\nMean Average Precision (mAP): %.3f' % mAP)
    logger.info(f'\nMean Average Precision (mAP): {mAP: %.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_24-6_SSD_evaluation')
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

    # Good formatting when printing the APs for each class and mAP
    pp = PrettyPrinter()

    # Parameters
    data_folder = './'
    keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
    batch_size = json_data["Config"][3]["batch_size"]#64
    workers = json_data["Config"][3]["workers"]#4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #checkpoint = './checkpoint_ssd300.pth.tar'
    home = os.path.expanduser("~")    
    checkpoint = f'{home}/infinicloud/{json_data["Config"][4]["weights"]}'
    if not os.path.isfile(checkpoint):
        raise Exception(f"checkpoint: {checkpoint} not found!")

    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)

    # Switch to eval mode
    model.eval()

    # Load test data
    test_dataset = datasets.PascalVOCDataset(data_folder,
                                                split='test',
                                                keep_difficult=keep_difficult)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, shuffle=False,
                                                collate_fn=test_dataset.collate_fn, 
                                                num_workers=workers, pin_memory=True)

    evaluate(test_loader, model)

    est_timer(t0)