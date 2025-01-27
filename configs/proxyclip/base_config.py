# base configurations
model = dict(
    type='CLIPForSegmentation',
    wrapper = 'CLIPWrapperOpenCLIP',
    pretrained = 'openai', #openai laion2b_s34b_b88k
    clip_path='ViT-B-16',
    vfm_model='dino',   # sam, mae, dino, dinov2
)

