#!/bin/bash
# model names and paths for those to be eval
model_names=(
    'my_model_A--base'
    'my_model_B--ema'
)
# corresponding paths
model_paths=(
    '/xxx/xxx/my_model_A.pth.tar'
    '/xxx/xxx/my_model_B.pth.tar'
)
# attack setting
attack_names=('none' 'pgd_linf' 'pgd_l2' 'fgsm' 'autoattack_linf' 'mim_linf' 'pgd_l1' 'ddn_l2')
eps_small=('0' '0.5/255' '0.5' '0.5/255' '0.5/255' '0.5/255' '100.0' '0.5')
eps_mid=('0' '2/255' '2.0' '2/255' '2/255' '2/255' '400.0' '2.0')
eps_large=('0' '8/255' '8.0' '8/255' '8/255' '8/255' '1600.0' '8.0')

# start testing
for ((i=0;i<${#model_names[@]};i++)) do
    mkdir "${model_names[i]}"
    cd "${model_names[i]}"
    # test clean and each attack
    for ((j=0;j<${#attack_names[@]};j++)) do
        PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 spring.submit arun -s -r --gpu -n16 "python -u -m prototype.solver.benchmark_eval_adv --config ../config.yaml --src_name ${model_names[${i}]} --src_path ${model_paths[${i}]} --tgt_name ${model_names[${i}]} --tgt_path ${model_paths[${i}]} --attack ${attack_names[${j}]} --eps ${eps_small[${j}]}"
        if [ ${j} != 0 ]
        then
             PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 spring.submit arun -s -r --gpu -n16 "python -u -m prototype.solver.benchmark_eval_adv --config ../config.yaml --src_name ${model_names[${i}]} --src_path ${model_paths[${i}]} --tgt_name ${model_names[${i}]} --tgt_path ${model_paths[${i}]} --attack ${attack_names[${j}]} --eps ${eps_mid[${j}]}"
             PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 spring.submit arun -s -r --gpu -n16 "python -u -m prototype.solver.benchmark_eval_adv --config ../config.yaml --src_name ${model_names[${i}]} --src_path ${model_paths[${i}]} --tgt_name ${model_names[${i}]} --tgt_path ${model_paths[${i}]} --attack ${attack_names[${j}]} --eps ${eps_large[${j}]}"
        fi
    done
    cd ..
done
