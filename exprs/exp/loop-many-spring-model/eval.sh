PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit arun -r -n32 --job-type urgent --gpu "python -u -m prototype.prototype.solver.multi_eval_solver --config config.yaml --evaluate"

# PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
# srun --mpi=pmi2 -p Test -n16 --gres=gpu:8 --ntasks-per-node=8 "python -u -m prototype.solver.cls_solver --config config.yaml --evaluate"

