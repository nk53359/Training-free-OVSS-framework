_base_ = '/home/nkombol/Open Vocabulary Baseline/configs/base_config.py'

# model settings
model = dict(
    name_path='./configs/sclip/cls_ade20k.txt',
    extra_class = 150
)


# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/mnt/sata_ssd1/nkombol/datasets/ade/ADEChallengeData2016/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))