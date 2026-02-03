export PYTHONPATH=/mnt/afs_1/huangyushi/RobustART
export SKIP_DIST=1
export GLOG_vmodule=MemcachedClient=-1
export CUDA_VISIBLE_DEVICES=0


# attack_names=('fgsm' 'fgsm' 'pgd_linf' 'pgd_linf' 'autoattack_linf' 'autoattack_linf' 'mim_linf' 'mim_linf' 'pgd_l2' 'pgd_l2' 'pgd_l1' 'pgd_l1')
# eps=('2/255' '0.5/255' '2/255' '0.5/255' '2/255' '0.5/255' '2/255' '0.5/255' '2.0' '0.5' '400.0' '100.0')
attack_names=('none' 'fgsm' 'fgsm' 'fgsm' 'pgd_linf' 'pgd_linf' 'autoattack_linf' 'autoattack_linf' 'mim_linf' 'mim_linf' 'pgd_l2' 'pgd_l2' 'pgd_l1' 'pgd_l1')
eps=('0' '8/255' '2/255' '0.5/255' '2/255' '0.5/255' '2/255' '0.5/255' '2/255' '0.5/255' '2.0' '0.5' '400.0' '100.0')
eps_small=('0' '0.5/255' '0.5' '0.5/255' '0.5/255' '0.5/255' '100.0' '0.5')
eps_mid=('0' '2/255' '2.0' '2/255' '2/255' '2/255' '400.0' '2.0')
eps_large=('0' '8/255' '8.0' '8/255' '8/255' '8/255' '1600.0' '8.0')

model_names=(
    'convnext_base'
    'convnextv2_base'
    'convnext_base_cvst'
    'vit_base'
    'vit_base_cvst'
    "clip_vit_l_14"
    "clip_vit_l_14_fare2_clip"
    "clip_vit_l_14_tecoa2_clip"
)
# corresponding paths
model_paths=(
    '/mnt/afs_1/huangyushi/RobustART/models/convnext_base_224.pth'
    '/mnt/afs_1/huangyushi/RobustART/models/convnextv2_base_1k_224_ema.pt'
    '/mnt/afs_1/huangyushi/RobustART/models/convnext_b_cvst_clean.pt'
    '/mnt/afs_1/huangyushi/RobustART/models/jx_vit_base_p16_224-80ecf9dd.pth'
    '/mnt/afs_1/huangyushi/RobustART/models/vit_b_cvst_clean.pt'
    '/mnt/afs_1/huangyushi/RobustART/models/ViT-L-14.pt'
    '/mnt/afs_1/huangyushi/RobustART/models/fare2-clip'
    '/mnt/afs_1/huangyushi/RobustART/models/tecoa2-clip'
)

i=0
j=3



for ((j=0;j<${#attack_names[@]};j++)) do
    python -u -m prototype.prototype.solver.benchmark_eval_adv --config config.yaml --src_name ${model_names[${i}]} --src_path ${model_paths[${i}]} --tgt_name ${model_names[${i}]} --tgt_path ${model_paths[${i}]} --attack ${attack_names[${j}]} --eps ${eps[${j}]}
done
# python -u -m prototype.prototype.solver.benchmark_eval_adv --config config.yaml --src_name ${model_names[${i}]} --src_path ${model_paths[${i}]} --tgt_name ${model_names[${i}]} --tgt_path ${model_paths[${i}]} --attack none --eps 0