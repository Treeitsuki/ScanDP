# Troubleshooting (Q&A)

**Q: The alignment between the partial scan and the reference model is clearly wrong. What should I check first?**  
A: Make sure the voxel downsampling resolution is identical for both point clouds (partial scan and reference). Using different voxel sizes changes point distribution and feature consistency, and is the most common cause of misalignment. Set the same `voxel_size` (or `downsample_voxel`) for both before feature extraction and ICP.

**Q: ICP alone keeps converging to a wrong pose. How can I improve robustness?**  
A: Do not rely on ICP as the first step. Use a global alignment stage such as FPFH feature matching + RANSAC (or equivalent) to provide a coarse initial transform, then refine with ICP. This is especially important when the initial pose is far from the reference.

**Q: What is the recommended alignment flow for large initial misalignment?**  
A: Use a two-stage pipeline:  
1) Voxel downsampling with the same resolution for both clouds  
2) FPFH feature extraction  
3) RANSAC (or similar) for coarse alignment  
4) ICP for fine alignment
