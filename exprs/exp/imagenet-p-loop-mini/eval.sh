PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
# Use srun if you have
#spring.submit arun -r -n8 --gpu -x SH-IDC1-10-5-37-21 \
python -u -m prototype.prototype.solver.imgnet_p_eval_solver --config config.yaml --evaluate

# PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
# srun --mpi=pmi2 -p Test -n16 --gres=gpu:8 --ntasks-per-node=8 "python -u -m prototype.solver.cls_solver --config config.yaml --evaluate"

