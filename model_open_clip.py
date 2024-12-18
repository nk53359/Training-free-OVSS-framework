import torch
import torch.nn.functional as F
import open_clip


class CLIPWrapperOpenCLIP(torch.nn.Module):
    def __init__(self, model_name, pretrained, device):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device = device)
        self.model.eval()
        self.tokenizer = tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        self.pretrained = pretrained

        self.dtype = self.model.visual.conv1.weight.dtype
        self.patch_size = self.model.visual.conv1.kernel_size[0]
    def encode_image(self, image, return_all=False, csa=False):
        return self.forward(image.type(self.dtype), return_all=return_all, csa=csa)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def forward(self, x: torch.Tensor, return_all=False, csa=True):
        B, nc, w, h = x.shape
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if x.shape[1] != self.model.visual.positional_embedding.shape[0]:
            x = x + self.model.visual.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)           
        
        
        if not self.model.visual.transformer.resblocks[0].attn.batch_first:  # open_clip implemented with batch_first dim in attn layers
                x = x.permute(1, 0, 2)  # NLD -> LND, if underlying model doesn't expect batch as first dimension
        for blk in self.model.visual.transformer.resblocks[:-1]:
            x = blk(x)
                
        for blk in self.model.visual.transformer.resblocks[-1:]:
            if self.model.visual.transformer.resblocks[0].attn.batch_first:  # must be in sequence_length first format for custom_attn
                x = x.permute(1, 0, 2)  # NLD -> LND 
            
            x = x + self.custom_attn(blk.attn, blk.ln_1(x), csa=csa)
            x = x + blk.mlp(blk.ln_2(x))
        x = x.permute(1, 0, 2)  # LND -> NLD
            
        if return_all:
            return self.model.visual.ln_post(x) @ self.model.visual.proj

        x = self.model.visual.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.visual.proj
        
        return x
    def custom_attn(self, attn_layer, x, return_attn=False, with_attn=False, csa=False):
        
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if csa:
            q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            k_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
        else:
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)

        if return_attn:
            return attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output

