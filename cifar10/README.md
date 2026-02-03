This is the code for our NeurIPS 2022 paper "Exploring the Relationship between Architecture and Adversarially Robust Generalization".

## Prerequisites
Create anaconda environment.
```
conda create -n archadv python=3.6
conda activate archadv
```
Install requirements.
```
pip install -r requirements.txt
```


## Training

Conduct vanilla training for the selected architecture (e.g., ViT):
```
python train.py --net vit --lr 0.00025 --wd 0.05
```


Conduct standard PGD-$\ell_{\infty}$ adversarial training and adversarial evaluation for the selected architecture (e.g., ViT):
```
python train.py --net vit --lr 0.00025 --wd 0.05 --advtrain
```
