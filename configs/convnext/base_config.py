# base configurations
model = dict(
    type='CLIPForSegmentation',
    wrapper = 'CLIPWrapperOpenCLIP',
    pretrained = 'laion_aesthetic_s13b_b82k_augreg', #openai laion2b_s34b_b88k
    clip_path='convnext_base_w_320'
)