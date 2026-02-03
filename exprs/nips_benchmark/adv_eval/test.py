import torch

ckpt = torch.load("/mnt/afs_1/huangyushi/RobustART/models/adv/ConvNext-B-CvSt/convnext_b_cvst_robust.pt")
print(ckpt.keys())