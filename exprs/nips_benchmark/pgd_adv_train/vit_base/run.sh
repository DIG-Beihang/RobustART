export PYTHONPATH=/mnt/afs_1/huangyushi/RobustART
# export SKIP_DIST=1
export GLOG_vmodule=MemcachedClient=-1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29601

# torchrun \
#   --nproc_per_node=8 \
#   -u -m prototype.prototype.solver.adv_cls_solver_train_pgd \
#   --config config.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29601  -m prototype.prototype.solver.adv_cls_solver_train_pgd_new --config config.yaml 