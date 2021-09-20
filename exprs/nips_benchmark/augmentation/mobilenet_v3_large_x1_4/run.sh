PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
# Use srun if you have
#spring.submit run -r -n16 --gpu
python -u -m prototype.prototype.solver.cls_solver --config config.yaml
# --recover=checkpoints/ckpt.pth.tar 

