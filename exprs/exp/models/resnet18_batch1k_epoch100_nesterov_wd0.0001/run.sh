PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n16 --cpus-per-task=5 \
"python -u -m prototype.solver.cls_solver --config config.yaml"  # --evaluate
