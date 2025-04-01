# Scene Graph Generation Pipeline

## Steps to Generate the Scene Graph

### 1️⃣ Create a Segmented Mesh
Run `labelmaker` as described in the main [README](../../README.md) to generate `point_lifted_mesh.ply`. This will serve as the segmented mesh for further processing.

### 2️⃣ Separate Object Instances
Run the **instance separation** script on the segmented mesh:

```bash
python scripts/scene_graph/instance_separation.py --workspace path/to/point_lifted_mesh.ply --save_path path/to/mesh_instances.ply
```

- This script will separate all instances of the same objects, treating them as **nodes** in the scene graph.
- The output will be saved at the location specified by `--save_path`.

📌 **Tuning Tip**: Modify the **HDBSCAN parameters** in [`instance_separation.py`](./instance_separation.py#L60-L64) to improve instance separation based on your data.

### 3️⃣ (Optional) Generate Mesh from Mask3D Predictions
You can also visualize the resulted mesh using **Mask3D prediction masks** for comparison. Run:

```bash
python scripts/scene_graph/mask3d_mesh.py --scene_mesh path/to/mesh.ply --mask3d path/to/intermediate/scannet200_mask3d_1 --output path/to/mesh_mask3d.ply
```

Ensure that you pass the correct paths as arguments.

### 4️⃣ Create the Scene Graph
Run the **scene graph generation** script:

```bash
python scripts/scene_graph/scene_graph.py --workspace path/to/mesh_instances.ply 
```

- If running the scene graph on a **Mask3D-generated mesh**, set the `--mapping_column` argument to `"id"`.
- You can **customize the scene graph** by modifying:
  - [`get_spatial_relationship()`](./scene_graph.py#L14) parameters.
  - [`statistical outlier removal`](./scene_graph.py#L193) settings in Open3D.
<br><br>
<p align="center">
  <img src="./labelmaker_scene_graph.gif" alt="Scene Graph Video" width="900">
</p>