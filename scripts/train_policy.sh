# Examples:

#   bash scripts/train_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/train_policy.sh dp_224x224_r3m gr1_dex-image 0913_example
#   bash scripts/train_policy.sh scandp_img_r3m cam_img 0219_img
#   bash scripts/train_policy.sh scandp_conv3d cam_gridmap 0204
#   bash scripts/train_policy.sh idp3 cam_pcd 0312_idp
#   bash scripts/train_policy.sh scandp_spconv cam_gridmap 1107

# dataset_path=/home/user/workspace/dataset/gridmap_sum_weighted
# dataset_path=/home/user/workspace/dataset/gridmap
dataset_path=/home/user/workspace/dataset/gridmap_real
# dataset_path=/home/cvl/cvl/ScanDP/scandp/data/dataset/pc_1024


DEBUG=False
# wandb_mode=offline
wandb_mode=online


alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    save_ckpt=False
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd scandp

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 
