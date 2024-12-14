import torch
import clip
from PIL import Image
from model import CLIPWrapper
from mmseg.datasets import CityscapesDataset
from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner import Runner
import custom_datasets
from mmseg.registry import DATASETS
from clip_segmentor import CLIPForSegmentation
init_default_scope('mmseg')
import torch
import timm
import torchmetrics

cfg = Config.fromfile("/home/nkombol/Open Vocabulary Baseline/configs/cfg_city_scapes.py")

# Build outside of "build_dataloader" because of linking issues
# TODO: fix linker
dataset = DATASETS.build(cfg.test_dataloader.dataset)
cfg.test_dataloader.dataset = dataset
runner = Runner.build_dataloader(cfg. test_dataloader)



model = timm.create_model('vit_base_patch16_clip_224', pretrained=True).cuda()
# TODO: find better alternative to taking visual and text tower from different models
text_model, _ = clip.load("ViT-B/16", device="cuda")
model_wrapper = CLIPWrapper(model, text_model).cuda()

# TODO: move visual and text model instantiation to segmentor
segmentor = CLIPForSegmentation(clip_path = "ViT-B/32", name_path = cfg.model.name_path, device=torch.device('cuda'), net = model_wrapper)


for img in runner:
    # TODO: make metric calc configurable
    img_tensor = img["inputs"][0].unsqueeze(0).cuda()
    res = segmentor.predict(img_tensor, img["data_samples"])
    preds =  res[0].pred_sem_seg.data.cuda()
    target = res[0].gt_sem_seg.data.cuda()
    iou_metric = torchmetrics.JaccardIndex(task='multiclass',ignore_index = 255,  num_classes=len(dataset._metainfo["classes"])).cuda()
    print(preds)
    print(target)
    print(iou_metric(preds, target))
    break

