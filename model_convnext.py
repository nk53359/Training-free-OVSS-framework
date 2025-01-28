import torch
import torch.nn.functional as F
import open_clip

class CLIPWrapperOpenCLIPConvNext(torch.nn.Module):
    def __init__(self, model_name, pretrained, device):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device = device)

        self.model.eval()
        self.tokenizer = tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        self.pretrained = pretrained

        self.dtype = self.model.visual.trunk.stem[0].weight.dtype
        self.patch_size = 32 # features are /32 ie like patch_size is 32
    def encode_image(self, image, return_all=False, csa=False):
        return self.forward(image.type(self.dtype), return_all=return_all, csa=csa)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def forward(self, x: torch.Tensor, return_all=False, csa=True):
        B, nc, w, h = x.shape

        x = self.model.visual.trunk.stem(x)  # shape = [*, width, grid, grid]
        x = self.model.visual.trunk.stages(x)  # shape = [*, width, grid, grid]
        x = self.model.visual.trunk.head.norm(x).permute(0, 2, 3, 1)  # shape = [*, width, grid, grid]
        
        x = self.model.visual.head.proj(x)  # shape = [*, width, grid, grid]

        return x
   
