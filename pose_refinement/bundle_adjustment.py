import pycolmap
import gin
import os

import numpy as np

from pathlib import Path

from hloc import (extract_features, match_features, reconstruction,
                  pairs_from_exhaustive, pairs_from_retrieval, match_dense, triangulation)
from pixsfm.bundle_adjustment import GeometricBundleAdjuster, BundleAdjuster

@gin.configurable
class BundleAdjustment():

    def __init__(self, colmap_dir,
                 n_retrieval=2):
        
        self.colmap_dir = colmap_dir
        self.n_retrieval = n_retrieval
        self.dense_conf = match_dense.confs['loftr'] # don't use Aachen, 
        self.dense_conf['model']['weights'] = 'indoor'

    def init(self):        
        pass

    def run(self):
        self.scene_path = Path(self.colmap_dir)
        
        self.retrieval_conf = extract_features.confs['netvlad']


        self.tmp_dir = Path(self.colmap_dir)
        self.sfm_pairs = self.tmp_dir / 'sfm-pairs.txt'
        image_dir = self.scene_path / 'images'
        image_list = []
        image_paths = list(image_dir.iterdir())
        image_paths = [f for f in image_paths if f.name.split('.')[0] != 'query']
        image_paths = sorted(image_paths, key=lambda x: int(x.name.split('.')[0]))
        
        image_paths = image_paths

        image_list_path = []
        indices = np.arange(len(image_paths))

        for index in indices:
            image_list.append(image_paths[index])
            image_list_path.append(
                str(Path(image_paths[index]).relative_to(image_dir)))

        self.retrieval_path = extract_features.main(self.retrieval_conf,
                                                    image_dir,
                                                    self.tmp_dir,
                                                    image_list=image_list_path)
        
        pairs_from_retrieval.main(self.retrieval_path,
                                    self.sfm_pairs,
                                    num_matched=self.n_retrieval)
        
        feature_path, match_path = match_dense.main(conf=self.dense_conf,
                                                    pairs=self.sfm_pairs,
                                                    image_dir=image_dir,
                                                    export_dir=self.tmp_dir)
        
        image_reader_options = pycolmap.ImageReaderOptions()
        image_reader_options.camera_model = "PINHOLE"

        self.model = triangulation.main(self.tmp_dir,
                                        self.tmp_dir / 'sparse',
                                        image_dir,
                                        self.sfm_pairs,
                                        feature_path,
                                        match_path,
                                        skip_geometric_verification=True)
        
        bundle_adjuster_conf = BundleAdjuster.default_conf
        bundle_adjuster_conf['optimizer']['refine_focal_length'] = False
        bundle_adjuster_conf['optimizer']['refine_principal_point'] = False
        bundle_adjuster_conf['optimizer']['refine_extra_params'] = False
        self.bundle_adjuster = GeometricBundleAdjuster(bundle_adjuster_conf)
        self.bundle_adjuster.refine(self.model)
        

    def save(self):
        colmap_output_dir = os.path.join(self.colmap_dir, 'loftr_ba')
        os.makedirs(colmap_output_dir, exist_ok=True)
        self.model.write_text(colmap_output_dir)

if __name__ == '__main__':
    refinement = BundleAdjustment('output/point_to_plane_5_filter_retrieval_uncertain_threshold/scene0575_00/colmap')
    refinement.init()
    refinement.run()
    refinement.save()