import time
import sys
import os
import pynvml
import torch

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
# cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b4_ns --lr 1e-4 --max_epochs 10 --train_batch_size 16 --precision 16 --weight_decay 1e-6 --version v14'
# print('还原efficientnet：\n', cmd)
# os.system(cmd)

# cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b4_ns --lr 1e-4 --max_epochs 10 --train_batch_size 8 --precision 32 --weight_decay 1e-6 --version v15'
# print('单精度训练：\n', cmd)
# os.system(cmd)

# cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone efficientnet_b0 --lr 1e-4 --max_epochs 10 --train_batch_size 20 --precision 32 --weight_decay 8e-4 --version v16'
# print(cmd)
# os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --precision 32 --weight_decay 1e-6 --version v18'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --precision 32 --weight_decay 1e-5 --version v19'
print(cmd)
os.system(cmd)

cmd = f'CUDA_VISIBLE_DEVICES={i} python train.py --backbone tf_efficientnet_b0_ns --lr 1e-4 --max_epochs 10 --train_batch_size 20 --precision 32 --weight_decay 1e-4 --version v19'
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
