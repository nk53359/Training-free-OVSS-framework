import torch
import torch.nn as nn
import sys 
sys.path.append("..")
import numpy as np
from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS
from pamr import PAMR

from torchvision.transforms import Compose, Normalize

from myutils import UnNormalize

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
class CLIPForSegmentationCLIP(BaseSegmentor):
    def __init__(self, name_path, device=torch.device('cuda'),
                    pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40, 
                    slide_stride=112, slide_crop=224, area_thd=None, net = None, tokenizer = None,
                    vfm_model = None):
        
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.net = net
        self.tokenizer = self.net.tokenizer
        self.device = device

        self.vfm_model = vfm_model
        self.vfm = None
        if vfm_model is not None:
            self.load_vmf()

        for i in range(len(self.net.preprocess.transforms)):
            if isinstance(self.net.preprocess.transforms[i], Normalize):
                mean = np.array(self.net.preprocess.transforms[i].mean)*255
                std = np.array(self.net.preprocess.transforms[i].std)*255
                break
        self.image_preprocess = Compose([
            Normalize(mean=mean ,  std=std),  # Normalizes the image
        ])

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
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

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        ex_feats = None
        if self.vfm_model is not None:
            clip_token_size = img.shape[-2] // self.net.model.visual.patch_size[0], img.shape[-1] // self.net.model.visual.patch_size[1]

            imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
            imgs_norm = torch.stack(imgs_norm, dim=0)

            imgs_norm = imgs_norm.half()
            I, J, ex_feats = self.vfm_forward_feature(imgs_norm, clip_token_size)
        
        image_features = self.net.encode_image(img, return_all=True, csa=True, ex_feats = ex_feats)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.vfm is None:
            image_features = image_features[:, 1:]

        logits = image_features @ self.query_features.T

        patch_size = self.net.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]

        if self.vfm is None:
            logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        else:
            logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)


        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        
        return logits

    def vfm_forward_feature(self, imgs_norm, clip_token_size):

        if self.vfm_model == 'sam':
            patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size
            imgs_norm = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.image_encoder(imgs_norm)

        elif self.vfm_model == 'dino':
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)
            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            
 
            nb_im = feat.shape[0]  # Batch size
            nb_tokens = feat.shape[1]  # Number of tokens
            nh = self.vfm.blocks[0].attn.num_heads  # Number of heads

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]

            patch_size = self.vfm.patch_embed.patch_size

            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size
            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)

        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]

        elif self.vfm_model == 'mae':
            patch_size = self.vfm.patch_embed.patch_size
            imgs_norm = F.interpolate(imgs_norm, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            image_feat = self.vfm.forward_features(imgs_norm)
            ex_feats = rearrange(image_feat, 'b (h w) c -> b c h w', h=I, w=J)

        else:
            I, J = clip_token_size
            ex_feats = None
        return I, J, ex_feats
        

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img)).to(torch.float16)
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        scaled_h = h_img //16
        scaled_w = w_img // 16
        attn_preds = img.new_zeros((21, 28, 21, 28))
        attn_count_mat = img.new_zeros((21, 28, 21, 28))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')
        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)
        return logits

    def predict(self, inputs, data_samples):
        
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
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], meta = batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)
    
    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(torch.float)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype) 
                seg_logits[1:] *= area_pred.transpose(0, -1)

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples
    def load_vmf(self):
        if self.vfm_model == 'sam':
                    self.vfm = sam_model_registry["vit_b"](checkpoint=checkpoint)
                    # self.vfm = sam_model_registry["vit_l"](checkpoint=checkpoint)

        elif self.vfm_model == 'dino':
                    # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
                    # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
                    # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
                    self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif self.vfm_model == 'dinov2':
                    # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
                    self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        elif self.vfm_model == 'mae':
                    self.vfm = models_vit.__dict__['vit_base_patch16'](img_size=slide_crop, num_classes=0, global_pool=False)
                    checkpoint_model = torch.load(checkpoint, map_location='cpu')['model']
                    state_dict = vfm.state_dict()
                    for k in ['head.weight', 'head.bias']:
                        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                            print(f"Removing key {k} from pretrained checkpoint")
                            del checkpoint_model[k]
                    # interpolate position embedding
                    interpolate_pos_embed(vfm, checkpoint_model)
                    # load pre-trained model
                    vfm.load_state_dict(checkpoint_model, strict=False)
        else:
                    print("vlm_model not supported")

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(self.device)

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