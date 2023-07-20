import numpy as np
from PIL import Image
import argparse
import os
from pathlib import Path
import logging
import shutil
from tqdm import tqdm
import cv2

log = logging.getLogger(__name__)


def process_arkit(scene_dir, keys):
    scene_dir = Path(scene_dir)
    assert scene_dir.exists()
    imgdir = scene_dir / "color"
    shutil.rmtree(imgdir, ignore_errors=True)
    imgdir.mkdir(exist_ok=False)
    depthdir = scene_dir / "depth"
    shutil.rmtree(depthdir, ignore_errors=True)
    depthdir.mkdir(exist_ok=False)

    rgb_keys = sorted(
        x.name.split('.png')[0] for x in (scene_dir / 'vga_wide').iterdir())

    for i, k in enumerate(tqdm(keys)):
        # get the closest depth key
        for j, ck in enumerate(rgb_keys):
            if ck >= k:
                key_before = rgb_keys[j - 1]
                key_after = ck
                break
        # now decide whether to take before or after
        # based on which one is closer
        rgb_key = min(
            key_before,
            key_after,
            key=lambda x: abs(float(x.split('_')[1]) - float(k.split('_')[1])))
        rgb = Image.open(scene_dir / "vga_wide" / f"{rgb_key}.png")
        # rotate
        rgb = rgb.transpose(Image.ROTATE_270)
        rgb.save(imgdir / f"{i}.jpg")
        depth = cv2.imread(str(scene_dir / "highres_depth" / f"{k}.png"),
                           cv2.IMREAD_UNCHANGED)
        # rotate
        depth = depth.T[:, ::-1]
        cv2.imwrite(str(scene_dir / 'depth' / f"{i}.png"), depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=str)
    parser.add_argument("--sample", type=int, default=1)
    flags = parser.parse_args()

    scene_dir = Path(str(flags.scene_dir)).resolve()
    assert scene_dir.exists()
    keys = sorted(
        x.name.split('.png')[0]
        for x in (scene_dir / 'highres_depth').iterdir())
    # now subsample
    keys = keys[::int(flags.sample)]
    print(f"Processing {len(keys)} frames")

    process_arkit(scene_dir, keys)
