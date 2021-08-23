PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r -n16 --gpu "python -u -m prototype.solver.adv_cls_solver_train_pgd --config config.yaml"
# --recover=checkpoints/ckpt.pth.tar

