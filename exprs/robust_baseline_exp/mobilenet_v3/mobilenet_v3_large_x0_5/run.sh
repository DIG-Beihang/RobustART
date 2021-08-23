PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r -n16 --gpu "python -u -m prototype.solver.cls_solver --config config.yaml"
# --recover=checkpoints/ckpt.pth.tar 
