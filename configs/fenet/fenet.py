_base_ = ['./dataset_setting.py']


# optimizer
lr = 0.01  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01, betas=(0.95, 0.99))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='OneCycle', max_lr=lr, pct_start=0.4, div_factor=10)  # lr_updater.py


# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=81)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=100,  # 分清epoch batch_size and interval
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None  # 类似预训练 读取后还是从头开始训练
resume_from = './work_dirs/fenet/epoch_80.pth'  # 从epoch位置开始 按照新传入的cfg文件继续训练到max epoch（可与之前的不同）lr什么的应该也会依据读入的epoch变化
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

