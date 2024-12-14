from PIL import Image
import requests
import timm
import clip
import torch
import torch.nn.functional as F


class CLIPWrapper(torch.nn.Module):
    def __init__(self, clip_net, text_model):
        super().__init__()
        
        self.model = clip_net
        self.dtype = next(clip_net.parameters()).dtype
        self.text_model = text_model
        #inace je float16
        self.text_model.visual.proj = torch.nn.Parameter(self.text_model.visual.proj.to(torch.float32))
        #self.width = 
    def encode_image(self, image, return_all=False, csa=False):
        return self.forward(image.type(self.dtype), return_all=return_all, csa=csa)

    def encode_text(self, text):

        return self.text_model.encode_text(text)

    def forward(self, x: torch.Tensor, return_all=False, csa=True):
        B, nc, w, h = x.shape

        x = self.model.patch_embed.proj(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        #x = torch.cat([self.model.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        """
        if x.shape[1] != self.model.pos_embed.shape[0]:
            x = x + self.model.visual.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.model.pos_embed.to(x.dtype)
        """
        x = self.model._pos_embed(x) #ovo samo po sebi dodaje cls, i pos embeddings

        x = self.model.norm_pre(x)           

        x = x.permute(1, 0, 2)  # NLD -> LND
        for blk in self.model.blocks[:-1]:
            x = blk(x)
        for blk in self.model.blocks[-1:]:
            x = x + self.custom_attn(blk.attn, blk.norm1(x), csa=csa)
            x = x + blk.mlp(blk.norm2(x))
        x = x.permute(1, 0, 2)  # LND -> NLD
            
        

        if return_all:
            return self.model.norm(x) @ self.text_model.visual.proj #timm model nema proj od 768 -> 512, pa se krade iz CLIP-a

        x = self.model.norm(x[:, 0, :])
        if self.model.proj is not None:
            x = x @ self.text_model.visual.proj

        return x
    def custom_attn(self, attn_layer, x, return_attn=False, with_attn=False, csa=False):
        
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.qkv.weight, attn_layer.qkv.bias ).chunk(3, dim=-1)
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
        attn_output = attn_layer.proj(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output


