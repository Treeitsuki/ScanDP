# bash scripts/vis_dataset.sh

# change your own dataset path
dataset_path=/home/cvl/cvl/ScanDP/scandp/data/test_long

vis_cloud=0
cd scandp
python vis_dataset.py --dataset_path $dataset_path \
                    --use_img 1 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 0 \
                    --downsample 1 \