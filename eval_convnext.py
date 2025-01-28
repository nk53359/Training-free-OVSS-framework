import torch
import clip
from PIL import Image
import os
import open_clip

from model_convnext import CLIPWrapperOpenCLIPConvNext

from mmseg.datasets import CityscapesDataset
from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner import Runner
import custom_datasets
from mmseg.registry import DATASETS
from clip_segmentor_convnext import CLIPForSegmentationConvNextCLIP

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


init_default_scope('mmseg')
import torch
import timm
import torchmetrics
from tqdm import tqdm
import numpy as np

from helper import save_tensor_as_image,save_tensor_as_cityscapes_image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg_path = "/home/nkombol/Open Vocabulary Baseline/configs/convnext/cfg_city_scapes.py"
cfg = Config.fromfile(cfg_path)

# Build dataset outside of "build_dataloader" because of linking issues
# TODO: fix linker
dataset = DATASETS.build(cfg.test_dataloader.dataset)
cfg.test_dataloader.dataset = dataset
runner = Runner.build_dataloader(cfg. test_dataloader)

# TODO: further streamline segmentor initialisation

model_wrapper = CLIPWrapperOpenCLIPConvNext(cfg.model.clip_path, pretrained=cfg.model.pretrained, device = device).to(device)

sam = sam_model_registry["vit_h"](checkpoint="/home/nkombol/Open Vocabulary Baseline/sam_vit_h_4b8939.pth").to(device)
mask_generator = SamAutomaticMaskGenerator(sam)


segmentor = CLIPForSegmentationConvNextCLIP(name_path = cfg.model.name_path, 
    device= device, 
    net = model_wrapper, 
    mask_gen = mask_generator, 
    extra_class = cfg.model.extra_class,
    logit_scale =  cfg.model.get("logit_scale",40),
    prob_thd =cfg.model.get("prob_thd",0.0) ,
    area_thd = cfg.model.get("area_thd",None),
    slide_stride=cfg.model.get("slide_stride", 112),
    slide_crop=cfg.model.get("slide_crop", 336 if "proxyclip" in cfg_path else 224)
)

# TODO: make metric calc configurable
iou_metric = torchmetrics.JaccardIndex(task='multiclass',  num_classes=len(dataset._metainfo["classes"]) + 1, ignore_index = 255).to(device)

for img in tqdm(runner):
    with torch.no_grad(): 
        pretprocessed_img = segmentor.data_preprocessor(img)
        # TODO: move channel reordering to image_preprocess
        #MMSeg loads images in bgr, open_clip expects rgb
        
        img_inputs_rgb = img["inputs"][0][[2, 1, 0], :, :]*1. 
        #print(img_inputs_rgb)

        img_tensor = segmentor.image_preprocess(img_inputs_rgb).unsqueeze(0).half().to(device)

        img_inputs_sam = np.array(Image.open(img["data_samples"][0].metainfo["img_path"]).convert("RGB"))
        #print(img_inputs_sam)
        #exit()

        res = segmentor.predict(img_tensor, pretprocessed_img["data_samples"], img_inputs_sam)
        preds =  res[0].pred_sem_seg.data.to(device)
        target = res[0].gt_sem_seg.data.to(device)

        iou_metric.update(preds, target)

mean_iou = iou_metric.compute()
print(mean_iou)

