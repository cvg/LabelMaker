import pycolmap
import gin
import os

import numpy as np

from pathlib import Path
from copy import deepcopy
from hloc import (extract_features, match_features, reconstruction,
                  pairs_from_exhaustive, pairs_from_retrieval, match_dense, triangulation, 
                  pairs_from_sequential)
from pixsfm.bundle_adjustment import GeometricBundleAdjuster, BundleAdjuster
from pixsfm._pixsfm import _bundle_adjustment as ba

import pixsfm
pixsfm.set_debug()
@gin.configurable
class BundleAdjustment():

    def __init__(self, 
                 colmap_dir,
                 matching='retrieval',
                 n_retrieval=2,
                 mode='loftr',):
        
        self.colmap_dir = colmap_dir
        self.mode = mode
        self.matching = matching
        self.n_retrieval = n_retrieval

        if self.mode == 'loftr':
            self.dense_conf = match_dense.confs['loftr'] # don't use Aachen, 
            self.dense_conf['model']['weights'] = 'indoor'
        
        elif self.mode == 'superpoint + superglue':
            self.feature_conf = extract_features.confs['superpoint_aachen']  
            self.matcher_conf = match_features.confs['superglue']
            self.matcher_conf['model']['weights'] = 'indoor'

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
        feature_path = self.tmp_dir / 'features.h5'
        match_path = self.tmp_dir / 'matches.h5'
        
        image_paths = image_paths

        image_list_path = []
        indices = np.arange(len(image_paths))

        for index in indices:
            image_list.append(image_paths[index])
            image_list_path.append(
                str(Path(image_paths[index]).relative_to(image_dir)))


        if self.matching == 'retrieval':
            self.retrieval_path = extract_features.main(self.retrieval_conf,
                                                        image_dir,
                                                        self.tmp_dir,
                                                        image_list=image_list_path)
            
            pairs_from_retrieval.main(self.retrieval_path,
                                        self.sfm_pairs,
                                        num_matched=self.n_retrieval)
            
        elif self.matching == 'exhaustive':
            pairs_from_exhaustive.main(self.sfm_pairs, image_list=image_list_path)

        elif self.matching == 'sequential':
            pairs_from_sequential.main(self.sfm_pairs, image_list=image_list_path, window_size=10)
        
        if self.mode == 'loftr':
            print('Dense matching...')

            feature_path, match_path = match_dense.main(conf=self.dense_conf,
                                                        pairs=self.sfm_pairs,
                                                        image_dir=image_dir,
                                                        export_dir=self.tmp_dir)
        elif self.mode == 'superpoint + superglue':
            extract_features.main(self.feature_conf, image_dir, image_list=image_list_path, feature_path=feature_path)
            match_features.main(self.matcher_conf, self.sfm_pairs, features=feature_path, matches=match_path)

        
        image_reader_options = pycolmap.ImageReaderOptions()
        image_reader_options.camera_model = "PINHOLE"


        print('Triangulation...')
        self.model = triangulation.main(self.tmp_dir,
                                        self.tmp_dir / 'sparse',
                                        image_dir,
                                        self.sfm_pairs,
                                        feature_path,
                                        match_path,
                                        skip_geometric_verification=False)
        
        colmap_output_dir = os.path.join(self.colmap_dir, 'triangulation')
        os.makedirs(colmap_output_dir, exist_ok=True)
        self.model.write_text(colmap_output_dir)
        
        print('Bundle adjustment...')
        bundle_adjuster_conf = BundleAdjuster.default_conf
        bundle_adjuster_conf['optimizer']['refine_focal_length'] = False
        bundle_adjuster_conf['optimizer']['refine_principal_point'] = False
        bundle_adjuster_conf['optimizer']['refine_extra_params'] = False
        bundle_adjuster_conf['optimizer']['solver']['minimizer_progress_to_stdout'] = True
        bundle_adjuster_conf['optimizer']['loss']['name'] = 'huber'
        bundle_adjuster_conf['optimizer']['loss']['params'] = [1.]
        bundle_adjuster_conf['references']['loss']['name'] = 'huber'
        bundle_adjuster_conf['references']['loss']['params'] = [1.]

        ba_model = deepcopy(self.model)

        reg_image_ids = ba_model.reg_image_ids()
        ba_setup = ba.BundleAdjustmentSetup()
        ba_setup.add_images(set(reg_image_ids))
        ba_setup.set_constant_pose(reg_image_ids[20])
        ba_setup.set_constant_tvec(reg_image_ids[21], [0])


        self.bundle_adjuster = GeometricBundleAdjuster(bundle_adjuster_conf)
        self.bundle_adjuster.refine(ba_model, problem_setup=ba_setup)

        ba_model.align_poses(self.model)

        self.ba_model = ba_model


    def save(self):
        colmap_output_dir = os.path.join(self.colmap_dir, 'triangulation_ba')
        os.makedirs(colmap_output_dir, exist_ok=True)
        self.ba_model.write_text(colmap_output_dir)

if __name__ == '__main__':
    refinement = BundleAdjustment('output/scannet_pose_refinement_downsample_1_sequential_loftr/scene0575_00/colmap', matching='sequential', mode='loftr')
    refinement.init()
    refinement.run()
    refinement.save()