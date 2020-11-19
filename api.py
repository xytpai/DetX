import os
import torch
import time
import yaml
import random
import logging
from datasets.utils import *


def load_log(cfg_file):
    logging.basicConfig( # cfg_file_name.log -> root
        level=logging.DEBUG, filename=os.path.split(cfg_file)[1].split('.')[0]+'.log', 
        filemode='a', format='%(asctime)s - %(levelname)s: %(message)s')


def load_cfg(cfg_file):
    cfg = None
    weight_file = None
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    weight_file = os.path.join(cfg['DETECTOR']['ROOT_WEIGHT'], 
        os.path.split(cfg_file)[1].split('.')[0] + '.pkl')
    cfg['weight_file'] = weight_file
    return cfg


def prepare_device(cfg, mode):
    if mode == 'TRAIN':
        torch.cuda.set_device(cfg['TRAIN']['DEVICES'][0])
        seed = cfg['TRAIN']['SEED']
        if seed >= 0:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.set_device(cfg[mode]['DEVICE'])


def prepare_detector(cfg, mode):
    dt = __import__('detectors.'+cfg['DETECTOR']['NAME'], 
                        fromlist=(cfg['DETECTOR']['NAME'],))
    detector = dt.Detector(cfg, mode=mode)
    if mode == 'TRAIN':
        if cfg['TRAIN']['LOAD']:
            detector.load_state_dict(torch.load(cfg['weight_file'], map_location='cpu'))
        detector = torch.nn.DataParallel(detector, device_ids=cfg['TRAIN']['DEVICES'])
        detector = detector.cuda(cfg['TRAIN']['DEVICES'][0])
        detector.train()
    else: 
        detector.load_state_dict(torch.load(cfg['weight_file'], map_location='cpu'))
        detector = detector.cuda(cfg[mode]['DEVICE'])
        detector.eval()
    return detector


def prepare_dataset(cfg, detector, mode):
    ds = __import__('datasets.'+cfg['DATASET']['NAME'], 
                        fromlist=(cfg['DATASET']['NAME'],))
    dataset = ds.Dataset(cfg, mode)
    return dataset


def prepare_loader(cfg, dataset, mode):
    assert mode == 'TRAIN'
    return dataset.make_loader()


def prepare_optimizer(cfg, detector, mode):
    assert mode == 'TRAIN'
    lr_base = cfg['TRAIN']['LR_BASE']
    params = []
    for key, value in detector.named_parameters():
        if not value.requires_grad:
            continue
        _lr = lr_base
        _weight_decay = cfg['TRAIN']['WEIGHT_DECAY']
        if "bias" in key:
            _lr = lr_base * 2
            _weight_decay = 0
        params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]
    opt = torch.optim.SGD(params, lr=_lr, momentum=cfg['TRAIN']['MOMENTUM'])
    return opt


class Trainer(object):
    def __init__(self, cfg, detector, dataset, loader, opt):
        self.cfg = cfg
        self.detector = detector
        self.detector.train()
        self.dataset = dataset
        self.loader = loader
        self.opt = opt
        if cfg['TRAIN']['LOAD_TRAINED_LOG']:
            self.step = int(self.detector.module.trained_log[0])
            self.epoch = int(self.detector.module.trained_log[1])
        else:
            self.step = 0
            self.epoch = 0
        # lr
        self.grad_clip = cfg['TRAIN']['GRAD_CLIP']
        self.lr_base = cfg['TRAIN']['LR_BASE']
        self.lr_gamma = cfg['TRAIN']['LR_GAMMA']
        self.lr_schedule = cfg['TRAIN']['LR_SCHEDULE']
        self.warmup_iters = cfg['TRAIN']['WARMUP_ITER']
        self.warmup_factor = 1.0/3.0  
        self.device = cfg['TRAIN']['DEVICES']
        self.save = cfg['TRAIN']['SAVE']
        
    def step_epoch(self, save_last=False):
        if self.epoch >= self.cfg['TRAIN']['NUM_EPOCH']: 
            if save_last:
                self.detector.module.trained_log[0] = self.step
                self.detector.module.trained_log[1] = self.epoch
                torch.save(self.detector.module.state_dict(), self.cfg['weight_file'])
            return True
        logging.info('Start epoch ' + str(self.epoch))
        self.detector.train()
        self.detector.module.backbone.freeze_stages(int(self.cfg['TRAIN']['FREEZE_STAGES']))
        if self.cfg['TRAIN']['FREEZE_BN']: self.detector.module.backbone.freeze_bn()
        # loop        
        for i, data in enumerate(self.loader):
            # lr function
            lr = self.lr_base
            if self.step < self.warmup_iters:
                alpha = float(self.step) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
                lr = lr*warmup_factor 
            else:
                for j in range(len(self.lr_schedule)):
                    if self.step < self.lr_schedule[j]:
                        break
                    lr *= self.lr_gamma
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
            # #########
            if i == 0: batch_size = int(data['imgs'].shape[0])
            torch.cuda.synchronize()
            start = time.time()
            self.opt.zero_grad()          
            loss = eval(self.dataset.loss_def)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.grad_clip)
            self.opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=\
                self.device[0]) / 1024 / 1024)
            torch.cuda.synchronize()
            totaltime = int((time.time() - start) * 1000)
            print('total_step:%d: epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                (self.step, self.epoch, i*batch_size, len(self.dataset), loss, maxmem, totaltime, lr))
            self.step += 1
        self.epoch += 1
        if self.save:
            self.detector.module.trained_log[0] = self.step
            self.detector.module.trained_log[1] = self.epoch
            torch.save(self.detector.module.state_dict(), self.cfg['weight_file'])
        return False


class Inferencer(object):
    def __init__(self, cfg, detector, dataset, mode):
        self.cfg = cfg
        self.detector = detector
        self.detector.eval()
        self.dataset = dataset
        self.mode = mode
        self.normalizer = dataset.normalizer
        assert self.mode != 'TRAIN'

    def pred(self, img_pil):
        with torch.no_grad():
            img, location = self.dataset.transform_inference_img(img_pil)
            return self.detector(img.cuda(), location)

