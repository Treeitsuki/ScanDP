# Examples:

#   bash scripts/deploy_random.sh idp3 cam_pcd 0312_idp

dataset_path=/home/cvl/cvl/ScanDP/scandp/data/gridmap


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


cd scandp


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python deploy_random.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 
