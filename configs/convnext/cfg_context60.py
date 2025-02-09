_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/sclip/cls_context60.txt',
    logit_scale=50,
    prob_thd=0.1,
    extra_class = 0
)

# dataset settings
dataset_type = 'PascalContext60Dataset'
data_root = '/mnt/sata_ssd1/nkombol/datasets/VOCdevkit/VOC2010/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))