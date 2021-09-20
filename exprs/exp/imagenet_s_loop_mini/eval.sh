T=`date +%m%d%H%M`
PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r -n2 -x BJ-IDC1-10-10-30-236 --gpu --job-type=urgent "python -u -m prototype.prototype.solver.multi_eval_decoder_resize_solver --config config.yaml --evaluate 2>&1 | tee log.train.$T"

#PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
#srun --mpi=pmi2 -p Test -n16 --gres=gpu:8 --ntasks-per-node=8 "python -u -m prototype.solver.multi_eval_decoder_resize_solver --config config.yaml --evaluate"

