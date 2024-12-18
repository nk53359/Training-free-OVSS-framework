import torch
import clip
from PIL import Image
import os
import open_clip

from model_open_clip import CLIPWrapperOpenCLIP

from mmseg.datasets import CityscapesDataset
from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner import Runner
import custom_datasets
from mmseg.registry import DATASETS
from clip_segmentor_clip import CLIPForSegmentationCLIP


init_default_scope('mmseg')
import torch
import timm
import torchmetrics
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config.fromfile("/home/nkombol/Open Vocabulary Baseline/configs/cfg_city_scapes.py")


# Build dataset outside of "build_dataloader" because of linking issues
# TODO: fix linker
dataset = DATASETS.build(cfg.test_dataloader.dataset)
cfg.test_dataloader.dataset = dataset
runner = Runner.build_dataloader(cfg. test_dataloader)

# TODO: further streamline segmentor initialisation

model_wrapper = CLIPWrapperOpenCLIP(cfg.model.clip_path, pretrained=cfg.model.pretrained, device = device).to(device)

segmentor = CLIPForSegmentationCLIP(name_path = cfg.model.name_path, device= device , net = model_wrapper)

# TODO: make metric calc configurable
iou_metric = torchmetrics.JaccardIndex(task='multiclass',ignore_index = 255,  num_classes=len(dataset._metainfo["classes"])).cuda()

for img in tqdm(runner):
    with torch.no_grad(): 
        pretprocessed_img = segmentor.data_preprocessor(img)
        # TODO: move channel reordering to image_preprocess
        #MMSeg loads images in bgr, open_clip expects rgb
        img_inputs_rgb = img["inputs"][0][[2, 1, 0], :, :]*1. 

        img_tensor = segmentor.image_preprocess(img_inputs_rgb).unsqueeze(0).half().cuda()
        
        res = segmentor.predict(img_tensor, pretprocessed_img["data_samples"])
        preds =  res[0].pred_sem_seg.data.cuda()
        target = res[0].gt_sem_seg.data.cuda()

        iou_metric.update(preds, target)

        

mean_iou = iou_metric.compute()
print(mean_iou)

