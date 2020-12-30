import time
import sys
import os
import pynvml
import torch

t_s = time.time()
hour = 0
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

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --weight_decay 1e-6 --onehot --version v47-beta2 --ricap 0 --mixup 0 --cutmix 0 --fmix 0.5 --fmix_beta 2.0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --weight_decay 1e-6 --onehot --version v47-beta3 --ricap 0 --mixup 0 --cutmix 0 --fmix 0.5 --fmix_beta 3.0 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --weight_decay 1e-6 --onehot --version v47-delta2 --ricap 0 --mixup 0 --cutmix 0 --fmix 0.5 --fmix_beta 1.0 --fmix_delta 2 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --weight_decay 1e-6 --onehot --version v47-delta3 --ricap 0 --mixup 0 --cutmix 0 --fmix 0.5 --fmix_beta 1.0 --fmix_delta 3 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --weight_decay 1e-6 --onehot --version v47-delta4 --ricap 0 --mixup 0 --cutmix 0 --fmix 0.5 --fmix_beta 1.0 --fmix_delta 4 --snapmix 0 --smooth 0.7 --sampler common'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --weight_decay 1e-6 --onehot --version v47-delta5 --ricap 0 --mixup 0 --cutmix 0 --fmix 0.5 --fmix_beta 1.0 --fmix_delta 5 --snapmix 0 --smooth 0.7 --sampler common'
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
