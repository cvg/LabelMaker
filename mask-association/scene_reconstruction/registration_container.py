import os
import shutil
import glob
from pathlib import Path
from typing import Dict, Optional, Literal

import torch
import numpy as np
import deeplake
import pycolmap
from torchtyping import TensorType as TorchTensor
from hloc import extract_features, match_features, match_dense
from hloc.pairs_from_poses import *
from hloc.pairs_from_retrieval import *
from hloc import reconstruction, triangulation, localize_sfm
from natsort import natsorted

from utils.config_utils import populate_dict
from utils.deeplake_utils import dataset_name_from_path
from scene_reconstruction.colmap_utils import *
from scene_reconstruction.utils import *
from utils.camera_poses import *


class RegistrationContainer:
    """ 
    Interface for constructing sfm reconstructions and/or registering datasets via hloc/pycolmap.
    """
    def __init__(
        self,
        ds: deeplake.Dataset,
        name: str = None,
        descriptors_extractor_global: Optional[str] = None, # netvlad
        descriptors_extractor_local : Optional[str] = None, # superpoint_inloc
        matcher                     : Optional[str] = None, # superglue
        overwrite                   : bool = False,
    ):
        """
        :param id: Name of the container.
        :param ds: Dataset to use for reconstruction/registration.
        :param name: Name to append to id as suffix. If None, don't include suffix (default reconstruction).
        :param descriptors_extractor_global: Global descriptors extractor to use.
        :param descriptors_extractor_local : Local  descriptors extractor to use.
        :param matcher                     : Matcher to use e.g. superglue for sparse, loftr for dense matching
        :param overwrite                   : If True, overwrite existing container.

        Warning: if you change hyperparams but the container already exists, you will need to call 
        with overwrite=True to recompute the container. This is because if a file already exists, 
        below methods ensure that it is loaded instead of recomputed. 
        """
        self.id = dataset_name_from_path(ds.path).replace('/', '#') + (f'_{name}' if name is not None else '')
        self.ds = ds
        self.path = Path(ds.path)/'registration'/self.id
        self.matcher_dense = matcher is not None and matcher in {'loftr'}
        self.model_configs = {
            'descriptors_extractor_global': extract_features.confs[descriptors_extractor_global] \
                if descriptors_extractor_global else None,
            'descriptors_extractor_local' : extract_features.confs[descriptors_extractor_local] \
                if descriptors_extractor_local else None,
            'descriptors_extractor_local_dense': match_dense.confs[matcher] \
                if self.matcher_dense else None,
            'matcher': (match_features.confs[matcher] if matcher else None) \
                if not self.matcher_dense else match_dense.confs[matcher],
        }

        def load_descriptors(key: str) -> Optional[Path]:
            config = self.model_configs[f'descriptors_extractor_{key}']
            if config is None:
                return None
            # features computed in matcher for local dense
            if (self.path/config['output']).exists() or key == 'local_dense':
                return self.path/config['output']
            return extract_features.main(config, self.path/'images', export_dir=self.path)

        def load_file_dict(regex: str) -> Dict:
            filenames = natsorted(glob.glob(regex))
            fid = lambda filename: filename.split('@')[-1].split('.')[0]
            return {fid(filename): filename for filename in filenames}

        def load_reconstruction(path: Path) -> Optional[pycolmap.Reconstruction]:
            if path.exists():
                return pycolmap.Reconstruction(path)
            return None

        if self.path.exists() and overwrite:
            shutil.rmtree(self.path)
        self.metadata = pycolmap_unpack_dataset(self.path, ds, self.id)
        self.descriptors = {k: load_descriptors(k) for k in ['global', 'local', 'local_dense']}
        self.descriptors_local_key = 'local_dense' if self.matcher_dense else 'local'
        self.pairs   = load_file_dict(str(self.path/'pairs@*.txt'))
        self.matches = load_file_dict(str(self.path/'matches@*.h5'))
        self.reconstruction       = load_reconstruction(self.path/'reconstruction')
        self.reconstruction_empty = load_reconstruction(self.path/'reconstruction_empty')

    def retrieve_pairs(self, method: Literal['knn', 'poses'] = 'knn', reference_container=None, post_process=None, **kwargs) -> Path:
        """
        Generate image similarity pairs via `method`.
        :param reference_container: Container to retrieve similiar images from. If None, use self.
        :param num_matched (knn, poses): Number of most similiar images to retrieve per image. If None, retrieve all.
        :param min_score (knn): Minimum score to consider a pair.
        :param min_num_matched (knn): Min number of frames to match regardless of score
        :param post_process (knn): Post processing function to apply to pairs.
            Should take input of sorted List[(i, j)] where i < j and return subset list in same formatted.
        """
        if reference_container is None:
            reference_container = self
        instance = reference_container.id
        self.pairs[instance] = self.path/f'pairs@{instance}.txt'
        if self.pairs[instance].exists():
            return self.pairs[instance]

        if method == 'poses':
            raise NotImplementedError

        elif method == 'knn':
            if not 'min_score' in kwargs or 'num_matched' in kwargs:
                print('Warning: no min_score or num_matched specified, will retrieve all pairs.')
                
            filename_descriptors           = self.descriptors['global']
            filename_descriptors_reference = reference_container.descriptors['global']
            names           = list_h5_names(filename_descriptors)
            names_reference = list_h5_names(filename_descriptors_reference)
            descriptors           = get_descriptors(names          , filename_descriptors)
            descriptors_reference = get_descriptors(names_reference, filename_descriptors_reference)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            scores = torch.einsum('id,jd->ij', descriptors.to(device), descriptors_reference.to(device))
            if reference_container is self:
                invalid = torch.arange(len(names          )).view(-1,  1) == \
                          torch.arange(len(names_reference)).view( 1, -1)
                invalid = invalid.to(device)
                scores.masked_fill_(invalid, float('-inf'))

            if 'min_score' in kwargs:
                # loftr fails to write keypoints for unmatched frames
                topk = torch.topk(scores, kwargs.get('min_num_matched', 1), dim=1)
                topk_mask = torch.zeros_like(scores).scatter_(1, topk.indices, 1).bool()
                fill_mask = (scores < kwargs['min_score']) & (~topk_mask)
                scores.masked_fill_(fill_mask, float('-inf'))

            elif 'num_matched' in kwargs:
                topk = torch.topk(scores, kwargs['num_matched'], dim=1)
                indices = topk.indices
                valid_mask = topk.values.isfinite().cpu().numpy()

            if 'num_matched' not in kwargs:
                indices = torch.repeat_interleave(
                    torch.arange(len(names))[None], len(names_reference), dim=0
                )
                valid_mask = scores.isfinite().cpu().numpy()
            
            pairs = [(i, indices[i, j].item()) for i, j in zip(*np.where(valid_mask))]
            if reference_container is self:
                # deduplicate pairs
                pairs = {(i, j) if i < j else (j, i) for i, j in pairs}
            pairs = [(
                name2frame_num(names[i]),
                name2frame_num(names_reference[j]), scores[i, j].item()
            ) for i, j in pairs]
            pairs = sorted([(p1, p2, s) if p1 < p2 else (p2, p1, s) for p1, p2, s in pairs])
            if post_process:
                pairs = post_process(pairs)
            root = name2root(names[0])
            pairs = [(
                f'{root}-{p1:04}.png', 
                f'{root}-{p2:04}.png'
            ) for p1, p2, _ in pairs]
        
        else:
            raise NotImplementedError(f'Pairs retrieval method {method} not supported.')
        
        save_pairs(self.pairs[instance], pairs)
        return self.pairs[instance]

    def match(self, reference_container=None, pairs=None) -> Path:
        """
        Match local descriptors using pairs generated by retrieve_pairs(`reference_container`).
        """
        if reference_container is None:
            reference_container = self
        instance = reference_container.id
        self.matches[instance] = self.path/f'matches@{instance}.h5'
        if self.matches[instance].exists():
            return self.matches[instance]
        
        if not self.matcher_dense:
            descriptors = self.descriptors['local']
            descriptors_reference = reference_container.descriptors['local']
            match_features.main(
                self.model_configs['matcher'], 
                self.pairs[instance] if pairs is None else pairs,
                features=descriptors, features_ref=descriptors_reference, matches=self.matches[instance]
            )
        else:
            match_dense.main(
                self.model_configs['matcher'],
                self.pairs[instance] if pairs is None else pairs,
                self.path/'images',
                features=self.descriptors['local_dense'], matches=self.matches[instance], export_dir=self.path
            )
        return self.matches[instance]

    def build_reconstruction_empty(self) -> pycolmap.Reconstruction:
        """
        Create empty sfm reconstruction to guide triangulation at `path`.
        """
        path = self.path/'reconstruction_empty'
        if self.reconstruction_empty is not None:
            return self.reconstruction_empty
        else:
            os.makedirs(path)

        poses = self.ds['pose'].numpy()
        poses = nerfstudio2colmap_poses(torch.from_numpy(poses))

        cameras, image_filenames = self.metadata['cameras'], self.metadata['image_filenames']
        reconstruction = pycolmap.Reconstruction()
        for camera in cameras:
            reconstruction.add_camera(camera)
        for i, (filename, camera, pose) in enumerate(zip(image_filenames, cameras, poses)):
            qvec, tvec = matrix4x4_to_qt(pose.reshape(1, 4, 4))
            qvec = qvec.numpy().reshape(4, 1).astype(np.float64)
            tvec = tvec.numpy().reshape(3, 1).astype(np.float64)
            image = pycolmap.Image(str(filename.name), [], tvec, qvec, camera_id=i, id=i) # [] denotes points3D
            reconstruction.add_image(image), reconstruction.register_image(image.image_id) 
        reconstruction.write(path)
        return reconstruction

    def build_reconstruction(self, num_matched=5, known_poses=True) -> pycolmap.Reconstruction:
        """ 
        Create sfm reconstruction using pairs and matches generated by `retrieve_pairs` and `match`. If poses are not,
        known, colmap will estimate them using their own coordinate system.
        """
        path       = self.path/'reconstruction'
        path_empty = self.path/'reconstruction_empty'
        if self.reconstruction is not None:
            return self.reconstruction
        else:
            os.makedirs(path)

        instance = self.id
        if instance not in self.pairs:
            self.retrieve_pairs(num_matched=num_matched)
        if instance not in self.matches:
            self.match(self)

        descriptors = self.descriptors[self.descriptors_local_key]
        if not known_poses:
            # pycolmap estimates camera poses
            self.reconstruction = reconstruction.main(
                path, 
                self.path/'images', self.pairs[instance], descriptors, self.matches[instance]
            )
        else:
            self.reconstruction_empty = self.build_reconstruction_empty()
            self.reconstruction = triangulation.main(
                path, 
                path_empty,
                self.path/'images', self.pairs[instance], descriptors, self.matches[instance],
                mapper_options={
                    'ba_refine_focal_length'   : False,
                    'ba_refine_extra_params'   : False,
                    'ba_refine_principal_point': False,
                }
            )
        return self.reconstruction

    def localize(self, reference_container=None, num_matched=5) -> np.ndarray:
        """
        Localize own dataset using `reference_container`'s sfm reconstruction.
        """
        assert (self.path/'queries.txt').exists(), \
            'Queries file not found. Use `generate_queries=True` to enable localization.'
        
        if reference_container is None:
            reference_container = self
        assert reference_container.reconstruction is not None, \
            'Reference reconstruction not found. Call reference_container.build_reconstruction() first.'

        instance = reference_container.id
        if instance not in self.pairs:
            self.retrieve_pairs(reference_container, num_matched=num_matched)
        if instance not in self.matches:
            self.match(reference_container)
        
        poses_filename = self.path/f'poses@{instance}.txt'
        localize_sfm.main(
            reference_container.reconstruction, 
            self.path/'queries.txt', 
            self.pairs[instance], self.descriptors[self.descriptors_local_key], self.matches[instance],
            poses_filename,
            covisibility_clustering=False, # not required with superpoint & superglue
        )
        return read_poses(poses_filename)
    
    def get_poses(self) -> HomogeneousTransform:
        """
        Get poses from the sfm reconstruction.
        """
        assert self.reconstruction is not None, 'Reconstruction not found. Call self.build_reconstruction() first.'
        return poses_from_reconstruction(self.reconstruction)

    def get_pairs(self, reference_container=None, as_dict=False):
        """
        Get retrieved pairs (image id1, image id2) from respective pair file. 
        """
        _id = self.id if not reference_container else reference_container.id
        filename = self.pairs.get(_id, None)
        assert filename, 'Pairs file not found. Call self.retrieve_pairs() first.'
        return read_pairs(filename, as_dict=as_dict)

    def get_keypoints(self, return_uncertainty=False):
        """
        Get keypoints (x, y) for all frames from the respective local descriptors file. 
        """
        filename = self.descriptors.get(self.descriptors_local_key, None)
        assert filename, 'Local descriptors missing. Please regenerate the container.'
        return read_keypoints(
            filename, self.metadata['image_names'], return_uncertainty=return_uncertainty
        )

    def get_matches(self, reference_container=None, return_scores=False, return_reverse=False):
        """
        Get matches for all pairs.
        :param return_scores: whether to return scores
        :param return_reverse: whether to add (name2, name1) given (name1, name2) in matches (default not added)
        """
        _id = self.id if not reference_container else reference_container.id
        filename = self.matches.get(_id, None)
        assert filename, 'Matches file not found. Call self.match() first.'
        return read_matches(
            filename, self.get_pairs(as_dict=False), return_scores=return_scores, return_reverse=return_reverse
        )
    
