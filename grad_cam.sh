export name=vit_base_patch16_224_augmentation
export SLURM_PROCID=0
export SLURM_NTASKS=1
export SLURM_NODELIST=localhost
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export PYTHONPATH=/data/RobustART/:$PYTHONPATH
mkdir -p visualization/atten_rollout/${name}
python prototype/prototype/tools/inference.py \
    --config exprs/nips_benchmark/augmentation/vit_base_patch16_224/config.yaml \
    -i datasets/images/val \
    -o visualization/atten_rollout/${name} \
    --attn_rollout \
    --meta_file datasets/images/meta/val.txt

