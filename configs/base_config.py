# base configurations
model = dict(
    type='CLIPForSegmentation',
    wrapper = 'CLIPWrapperOpenCLIP',
    pretrained = 'laion2b_s34b_b88k',
    clip_path='ViT-B-16'
)
