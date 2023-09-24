'''
    -PyTorch-Tutorial-to-Object-Detection/detect.py

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py
'''
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os, sys ,time
import json, pathlib
import argparse

sys.path.append('./libs')

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

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def est_timer(start_time):
    time_consumption, h, m, s= lib_misc.format_time(time.time() - start_time)         
    msg = 'Time Consumption: {}.'.format( time_consumption)#msg = 'Time duration: {:.2f} seconds.'
    logger.info(msg)


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [utils.rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./data/calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=utils.label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=utils.label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=utils.label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_24-5_SSD_inference')
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    # ファイルパスのリストを作成
    #checkpoint = 'checkpoint_ssd300.pth.tar'    
    home = os.path.expanduser("~")    
    checkpoint = f'{home}/infinicloud/{json_data["Config"][4]["weights"]}'
    if not os.path.isfile(checkpoint):
        raise Exception(f"checkpoint: {checkpoint} not found!")

    '''
    checkpoint = torch.load(checkpoint) 
    .
    .
    .
    ModuleNotFoundError: No module named 'model'
    '''
    '''
    ModuleNotFoundError: No module named ‘models‘解决torch.load问题【天坑】
    https://blog.csdn.net/weixin_42815846/article/details/115289861
    
    '''
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    #print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    logger.info(f'\nLoaded checkpoint from epoch {start_epoch}')

    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    img_path = './images/human_robot.jpg'#'./images/cowboy-757575_640.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()

    est_timer(t0)