# Examples:

#   bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/deploy_genesis.sh scandp_img_r3m cam_img 0130_img
#   bash scripts/deploy_genesis.sh scandp_conv3d cam_gridmap 0128_1000
#   bash scripts/deploy_genesis.sh scandp_conv3d cam_gridmap 0227_wobatch
#   bash scripts/deploy_genesis.sh scandp_spconv cam_gridmap 0310_spconv_50
#   bash scripts/deploy_genesis.sh scandp_spconv cam_gridmap 0409_08m

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

python deploy_gridmap.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 
