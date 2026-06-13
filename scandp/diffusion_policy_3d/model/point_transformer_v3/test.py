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
    model = PointTransformerV3(cls_mode=True, 
                               in_channels=num_feat, 
                               enc_patch_size=(512, 512, 512, 512, 512),
                               enable_flash=False,
                               ).cuda()
    patch_size = 400
    batch_size = 32
    batch_vals = torch.arange(0, batch_size, step=1)
    repeat_vals = torch.tensor([patch_size for i in range(batch_size)])
    batch_vals = torch.repeat_interleave(batch_vals, repeat_vals).cuda()
    feats = torch.rand((patch_size*batch_size, num_feat)).cuda()
    sample_data = {"feat": feats, "batch": batch_vals,
                "coord": feats.cuda(), "grid_size": 0.01}
    sample_dict = Point(sample_data)
    
    start_time = time.time()
    output = model(sample_dict)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    print("test result", output['feat'].shape)
    
    return output

def ave_pool_feats(point):
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

if __name__ == "__main__":
    # main()

    output = PTv3()
    print(output.keys())
    print(output["sparse_conv_feat"])
    avg = ave_pool_feats(output)
    print(avg.shape)
    print(avg)