import torch
import torch.nn as nn
import sys 
sys.path.append("..")
import numpy as np
from prompts.imagenet_template import openai_imagenet_template
from pycocotools import mask as mask_utils
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from helper import save_tensor_as_image,save_tensor_as_cityscapes_image
from mmseg.registry import MODELS
from pamr import PAMR

from torchvision.transforms import Compose, Normalize


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

@MODELS.register_module()
class CLIPForSegmentationConvNextCLIP(BaseSegmentor):
    def __init__(self, name_path, device=torch.device('cuda'),
                    pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40, 
                    slide_stride=112, slide_crop=224, area_thd=None, net = None, tokenizer = None, mask_gen = None,
                    extra_class = None):
        
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)

        self.device = device
        self.net = net
        self.tokenizer = self.net.tokenizer
        self.mask_gen = mask_gen
        self.extra_class = extra_class

        for i in range(len(self.net.preprocess.transforms)):
            if isinstance(self.net.preprocess.transforms[i], Normalize):
                mean = np.array(self.net.preprocess.transforms[i].mean)*255
                std = np.array(self.net.preprocess.transforms[i].std)*255
                break
        self.image_preprocess = Compose([
            Normalize(mean=mean ,  std=std),  # Normalizes the image
        ])
        
        query_words, self.query_idx = get_cls_idx(name_path)

        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.net.encode_text(query)

                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False

        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

    def forward_feature(self, img, logit_size=None, sam_img_input = None, img_path = None):
        if type(img) == list:
            img = img[0]
        image_features = self.net.encode_image(img, return_all=True, csa=True).permute(0, 3,1,2) # without permute with permute:1 640 1024 2048 
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if logit_size == None:
            image_features = nn.functional.interpolate(image_features, size=img.shape[-2:], mode='bilinear').permute(0, 2,3,1)
        else:
            image_features = nn.functional.interpolate(image_features, size=logit_size, mode='bilinear').permute(0, 2,3,1)

        
        if  isinstance(self.mask_gen, str):
            mask_json = open(self.mask_gen + img_path.strip().split("/")[-1].split(".")[0] + ".json", "r", encoding = "utf-8")
            mask_info = mask_json.read()
            data = eval(mask_info)
            
            mask_json.close()  
            masks = [torch.tensor(mask_utils.decode(mask["segmentation"])) for mask in data["annotations"]]

        else:
            img_inputs_rgb = img[0].cpu().numpy() # 1 1024 2048 640
            input_tensor = np.transpose(img_inputs_rgb, (1, 2, 0))
            masks = self.mask_gen.generate(sam_img_input)

        mask_embeddings = []
        msks = []
        for mask in masks:
            if not isinstance(self.mask_gen, str):
                mask = torch.tensor(mask["segmentation"])
            mask = mask.to(self.device)
            pooled_features = torch.sum(image_features[0]* mask.unsqueeze(-1), dim=(0, 1))
            averaged_features = pooled_features / (torch.sum(mask) * 1.)

            mask_embeddings.append(averaged_features)
            msks.append(mask)
        msks = torch.stack(msks, dim = 0)
        
        mask_embeddings = torch.stack(mask_embeddings, dim=0) # Nmask x D

        mask_logits = mask_embeddings @ self.query_features.T  # Nmask x C

        return mask_logits, msks

    def predict(self, inputs, data_samples, sam_img_input):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        mask_logits, masks  = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], sam_img_input, data_samples[0].metainfo["img_path"]) #batch_img_metas[0]['ori_shape'])

        mask_classes = mask_logits.argmax(dim = 1) # num of classes x 1
        mask_classes = self.query_idx[mask_classes]

        masks_time_logits = []

        for mask, one_mask_logits in zip(masks, mask_logits):
            mask = mask*1.
            
            mask[mask == 0] = float("-Inf")
            mask_logit = one_mask_logits.max()
            msk_tms_logit = mask * mask_logit
            masks_time_logits.append(msk_tms_logit)
        
        # Make sure pixels not part of any masks get categorized into "extra_class"
        masks_time_logits.append(torch.zeros_like(masks_time_logits[0]))
        new_value = torch.tensor([self.extra_class]).to(self.device)
        mask_classes = torch.cat((mask_classes, new_value))


        masks_time_logits = torch.stack(masks_time_logits, dim = 0)
        masks_to_pixel = torch.argmax(masks_time_logits, dim = 0) # [width x height]

        segmented_image = mask_classes[masks_to_pixel]


        return self.postprocess_result(None, data_samples, segmented_image)
    
    def postprocess_result(self, seg_logits, data_samples, segmented_image):

        for i in range(1):
            
            data_samples[i].set_data({
                'pred_sem_seg':
                PixelData(**{'data': segmented_image.unsqueeze(0)})
            })

        return data_samples
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices