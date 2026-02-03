import torch
from transformers import AutoModel

repo = "facebook/convnext-base-224"   # 或者本地路径 "./convnext-base-224"
model = AutoModel.from_pretrained(repo)

# 保存为 .pth（state_dict）
torch.save(model.state_dict(), "/mnt/afs_1/huangyushi/RobustART/models/convnext_base_224.pth")
