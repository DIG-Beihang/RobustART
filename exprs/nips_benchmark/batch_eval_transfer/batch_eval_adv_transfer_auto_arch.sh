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
attack_names=('fgsm')
eps=('8/255' '2/255' '0.5/255')
dir_name=('fgsm_0.031' 'fgsm_0.007' 'fgsm_0.001')
task_cnt=0

# start testing
for ((i=0;i<${#eps[@]};i++)) do
    for ((j=0;j<${#model_names[@]};j++)) do
        for ((k=0;k<${#model_names[@]};k++)) do
            if [ ${task_cnt} -gt 400 ]; then
                exit 0
            fi
            cd "${model_names[j]}_To_${model_names[k]}"
            if [ ! -d "./${dir_name[${i}]}" ]; then
                PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 spring.submit arun -s -r --gpu -n16 "python -u -m prototype.solver.benchmark_eval_adv --config ../config.yaml --src_name ${model_names[${j}]} --src_path ${model_paths[${j}]} --tgt_name ${model_names[${k}]} --tgt_path ${model_paths[${k}]} --attack fgsm --eps ${eps[${i}]}"
                task_cnt=$(($task_cnt+1))
            else
                if [ ! -f "./${dir_name[${i}]}/results.txt.all" ]; then
                    rm -r "./${dir_name[${i}]}"
                    PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 spring.submit arun -s -r --gpu -n16 "python -u -m prototype.solver.benchmark_eval_adv --config ../config.yaml --src_name ${model_names[${j}]} --src_path ${model_paths[${j}]} --tgt_name ${model_names[${k}]} --tgt_path ${model_paths[${k}]} --attack fgsm --eps ${eps[${i}]}"
                    task_cnt=$(($task_cnt+1))
                fi
            fi
            cd ..
        done
    done
done
