import time
import sys
import os
import pynvml
import torch

t_s = time.time()
hour = 2.0
minute = 0
second = 0
while 1:
    if time.time() - t_s > (hour*60+minute)*60+second:
        break

gpu_nums = torch.cuda.device_count()
flag = 0
while(1):
    for i in range(gpu_nums):
        pynvml.nvmlInit()
        # 这里的0是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used/1024/1024
        if used < 1024:
            flag = 1
            break
    if flag:
        break

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone seresnext50_32x4d --lr 1e-4 --max_epochs 10 --T_max 10 --train_batch_size 14 --weight_decay 1e-6 --onehot --version v52 --ricap 0 --mixup 0 --cutmix 0 --fmix 0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone seresnext50_32x4d --lr 1e-4 --max_epochs 10 --T_max 10 --train_batch_size 14 --weight_decay 1e-6 --onehot --version v52-cutmix --ricap 0 --mixup 0 --cutmix 0.5 --cutmix_beta 1.0 --fmix 0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone seresnext50_32x4d --lr 1e-4 --max_epochs 10 --T_max 10 --train_batch_size 14 --weight_decay 1e-6 --onehot --version v52-mixup --ricap 0 --mixup 0.5 --mixup_beta 0.4 --cutmix 0 --fmix 0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone seresnext50_32x4d --lr 1e-4 --max_epochs 10 --T_max 10 --train_batch_size 14 --weight_decay 1e-6 --onehot --version v52-cutmix-cutmix --ricap 0 --mixup 0.3 --mixup_beta 0.4 --cutmix 0.3 --cutmix_beta 1.0 --fmix 0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone seresnext50_32x4d --lr 1e-4 --max_epochs 15 --T_max 15 --train_batch_size 14 --weight_decay 1e-6 --onehot --version v52-cutmix-cutmix-15ep --ricap 0 --mixup 0.3 --mixup_beta 0.4 --cutmix 0.3 --cutmix_beta 1.0 --fmix 0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

# t_s = time.time()
# hour = 4
# minute = 0
# second = 0
# while 1:
#     if time.time() - t_s > (hour*60+minute)*60+second:
#         break
# os.system('python train.py --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --batch-size 32 --img-size 768 768 --device 0 --single-cls --noautoanchor')
