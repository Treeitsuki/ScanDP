#!/bin/bash

# ===== 設定 =====
# demo_dir: HDF5ファイルが格納されているディレクトリ
# script_path: 上記のPythonコードを保存したファイル（例: teleop_process.py）
demo_dir="/home/user/workspace/data/hdf5/"
script_path="/home/user/workspace/scandp/create_dataset.py"

# 出力ディレクトリを設定（任意）
output_dir="/home/user/workspace/save_dir"
mkdir -p "$output_dir"
source .venv/bin/activate

# ===== 実行ログ出力=====
echo "=== Start processing all HDF5 files in: $demo_dir ==="

# ===== ループ処理 =====
for file in "$demo_dir"/*.h5; do
    if [ -f "$file" ]; then
    echo "Processing file: $file"
        # Pythonスクリプトを実行 (output_dir を渡す)
    python3 "$script_path" "$file" --output-dir "$output_dir"

        # 実行結果の確認
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $file"
        else
            echo "❌ Error processing: $file"
        fi
        echo "-------------------------------------------"
    fi
done

python3 convert_demos.py --demo_dir "$output_dir" --save_dir ./gridmap --save_img 1 --save_depth 1

echo "=== All processing completed! ==="
