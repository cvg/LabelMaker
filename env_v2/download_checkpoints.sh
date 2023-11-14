env_name=labelmaker
dir_name="$(pwd)/$(dirname "$0")"
eval "$(conda shell.bash hook)"
conda activate $env_name

echo $dir_name
mkdir -p $dir_name/../checkpoints

# ovseg https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?pli=1
gdown "1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy" -O $dir_name/../checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth

# recognize-anything https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth
gdown "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth" -O $dir_name/../checkpoints/ram_swin_large_14m.pth

# grounding dino https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/GroundingDINO#checkpoints
gdown "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" -O $dir_name/../checkpoints/groundingdino_swint_ogc.pth

# sam-hq https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
gdown 1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8 -O $dir_name/../checkpoints/sam_hq_vit_h.pth

# cmx https://drive.google.com/file/d/1hlyglGnEB0pnWXfHPtBtCGGlKMDh2K--/view
gdown 1hlyglGnEB0pnWXfHPtBtCGGlKMDh2K-- -O $dir_name/../checkpoints/NYUDV2_CMX+Segformer-B2.pth

# InternImage https://huggingface.co/OpenGVLab/InternImage/blob/main/upernet_internimage_h_896_160k_ade20k.pth
gdown https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth -O $dir_name/../checkpoints/upernet_internimage_h_896_160k_ade20k.pth

# Mask3D https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_val.ckpt
gdown "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt" -O $dir_name/../checkpoints/mask3d_scannet200_benchmark.ckpt

# omnidata https://drive.google.com/file/d/1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI/view
# too many download
gdown "1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI" -O $dir_name/../checkpoints/omnidata_dpt_depth_v2.ckpt

# omnidata normal model https://drive.google.com/file/d/1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR/view
gdown "1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t" -O $dir_name/../checkpoints/omnidata_dpt_normal_v2.ckpt
