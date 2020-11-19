import os, sys
sys.path.append(os.getcwd())
from api import *
import argparse
from PIL import Image
from datasets.utils import *

parser = argparse.ArgumentParser(description='Pytorch Object Detection Demo')
parser.add_argument(
    '--cfg',
    help='path to config file'
)
args = parser.parse_args()
cfg_file = args.cfg

cfg = load_cfg(cfg_file)
mode = 'TEST'
demo_dir = cfg['DATASET']['ROOT_'+mode]

prepare_device(cfg, mode)
detector = prepare_detector(cfg, mode)
dataset = prepare_dataset(cfg, detector, mode)
inferencer = Inferencer(cfg, detector, dataset, mode)

for filename in os.listdir(demo_dir):
    if filename.endswith('jpg'):
        if filename[:5] == 'pred_': 
            continue
        img = Image.open(os.path.join(demo_dir, filename))
        pred = inferencer.pred(img)
        name = demo_dir + '/pred_' + filename.split('.')[0]+'.jpg'
        dataset.show(img, pred, name)
