import torch
import time

from model import PointTransformerV3, Point

# def main():
#     bs = 128
#     npts = 4096
#     len_xyz = 3
#     feat_dims = 1
#     coord = torch.rand(bs, npts, len_xyz).cuda().reshape(bs * npts, len_xyz)
#     feat = torch.rand(bs, npts, feat_dims).cuda().reshape(bs * npts, feat_dims)
#     offset = torch.tensor([i * npts for i in range(1, bs + 1)]).cuda()
#     grid_size = torch.tensor(0.2).cuda()

#     pointTransformer = PointTransformerV3(in_channels=feat_dims,
#                                           enable_flash=False
#                                         ).to(offset.device)
#     # print("model parameters:", sum(param.numel() for param in pointTransformer.parameters()))

#     data_dict = dict(
#         coord=coord,
#         feat=feat,
#         offset=offset,
#         grid_size=grid_size
#     )
#     print(data_dict.keys())

#     start_time = time.time()
#     result = pointTransformer(data_dict)
#     end_time = time.time()

#     # print(result.keys())
#     print("test result", result['feat'].shape)
#     # print(result['feat'])
#     print(f"Inference time: {end_time - start_time:.4f} seconds")

def PTv3():
    num_feat = 3
    patch_size = 100
    batch_size = 64
    batch_vals = torch.arange(0, batch_size, step=1)
    repeat_vals = torch.tensor([patch_size for i in range(batch_size)])
    batch_vals = torch.repeat_interleave(batch_vals, repeat_vals).cuda()
    feats = torch.rand((patch_size*batch_size, num_feat)).cuda()
    sample_data = {"feat": feats, "batch": batch_vals,
                "coord": feats.cuda(), "grid_size": 0.01}
    sample_dict = Point(sample_data)

    extractor = PointTransformerV3(cls_mode=True, 
                               in_channels=num_feat, 
                               enc_channels=(16, 32, 64, 128, 256),
                               enable_flash=False,
                               ).cuda()
    
    start_time = time.time()
    output = extractor(sample_dict)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    print("test result", output['feat'].shape)
    
    return output

def avg_pool_feats(point):
    # https://github.com/Pointcept/PointTransformerV3/issues/58
    feat = point.feat
    offsets = point.offset
    num_batch = offsets.shape[0]
    d = feat.shape[-1]
    avg_feats = torch.zeros((num_batch, d), dtype=feat.dtype, device=feat.device)
    for point_num in range(num_batch):
        if point_num == 0:
            lb = 0
            ub = offsets[point_num]+1
        else:
            lb = ub-1
            ub = offsets[point_num]+1
        range_ = ub-lb
        if range_ > 1:
            avg_feats[point_num] = torch.mean(feat[lb:ub])
    return avg_feats

def test():
    num_feat = 1  # 特徴量の次元（占有確率）
    batch_size = 2  # バッチサイズ

    # OGM (Occupancy Grid Map) をロード
    ogm = torch.load("/home/cvl/cvl/ScanDP/scandp/prob_map.pth")  # torch.Size([25, 25, 25])
    
    # 占有確率が 0.5 より大きいボクセルの座標 (x, y, z) を取得
    points = (ogm >= 0.5).nonzero(as_tuple=False).float().cuda()  # (N, 3) の座標データ
    
    # 各ボクセルの占有確率を特徴量 (feat) として取得
    occupancy_probs = ogm[ogm >= 0.5].view(-1, 1).float().cuda()  # (N, 1)

    num_points = points.shape[0]  # 有効な点の数
    
    # バッチサイズ分複製
    batched_points = points.repeat(batch_size, 1)  # (batch_size * N, 3)
    batched_feats = occupancy_probs.repeat(batch_size, 1)  # (batch_size * N, 1)

    # バッチ情報を作成
    batch_vals = torch.arange(0, batch_size, step=1).repeat_interleave(num_points).cuda()
    
    sample_data = {
        "feat": batched_feats,  # 占有確率
        "batch": batch_vals,
        "coord": batched_points,  # 点群座標 (x, y, z)
        "grid_size": 0.01
    }
    
    sample_dict = Point(sample_data)

    extractor = PointTransformerV3(cls_mode=True, 
                                   in_channels=num_feat, 
                                   enc_channels=(16, 32, 64, 128, 256),
                                   enable_flash=False,
                                   ).cuda()
    
    start_time = time.time()
    output = extractor(sample_dict)
    end_time = time.time()
    
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    print("test result", output['feat'].shape)
    
    return output
if __name__ == "__main__":
    # main()

    # output = PTv3()
    output = test()
    # print(output.keys())
    avg = avg_pool_feats(output)
    print(avg.shape)