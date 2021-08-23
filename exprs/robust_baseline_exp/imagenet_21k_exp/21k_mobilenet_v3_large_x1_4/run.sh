PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r -n32 --gpu --job-name=21k_mobilenet "python -u -m prototype.solver.cls_solver --config config.yaml"
# --recover=checkpoints/ckpt.pth.tar 
