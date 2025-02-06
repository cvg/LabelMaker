import random
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, Type

import deeplake
import igraph
import deeplake
import torch
import torch.nn as nn
from torchtyping import TensorType as TorchTensor
from tqdm import tqdm

from nerfstudio.configs.base_config import InstantiateConfig

from utils.config_utils import *
from utils.deeplake_utils import load_tensor_at
from scene_reconstruction.registration_container import RegistrationContainer, Keypoints, Matches
from scene_reconstruction.utils import *
from utils.segmentation import *
from utils.camera_poses import *
from optimized.union_find import UnionFind


@dataclass
class SequenceSegmentationFrame:
    """ Frame name and index in deeplake dataset """
    name : str
    index: int

    """ Frame image """
    image: TorchTensor["H", "W", 3]

    """ Binary segmentation masks and bounding boxes """
    bmasks: Mask
    bboxes: BBox

    """ Sfm local descriptors """
    keypoints: Keypoints

    """ Sfm matches wrt to other images in container. """
    matches: Matches

    """ Area of each bmask used for determining label order """
    bmask_areas: TorchTensor["num_masks"] = None

    """ Auxiliary maps between region indices and keypoints indices"""
    keypoints2bmask: Dict[int, int] = None
    bmask2keypoints: Dict[int, int] = None

    """ Auxiliary information that can be computed given sequence_label_base """
    sequence_label_base: int = None
    sequence_labels: TorchTensor["num_masks"] = None

    """ Instance label for each mask """
    instance_labels: TorchTensor["num_masks"] = None

    def __post_init__(self):
        """
        Compute auxiliary maps and information.
        """
        self.num_labels = len(self.bmasks) # excluding background = 0
        self.bmask_areas = torch.tensor([bmask.sum().item() for bmask in self.bmasks]) 

        # Used during sequence segmentation matching process
        self.bmask2keypoints = {}
        self.keypoints2bmask = defaultdict(list)
        for i, bmask in enumerate(self.bmasks):
            subkpnts = points_in_bmask(self.keypoints, bmask, return_indices=True)
            self.bmask2keypoints[i] = subkpnts
            for j in subkpnts:
                self.keypoints2bmask[j.item()].append(i)

        # Convert keypoints2mask to tensors
        max_overlap = max([len(v) for v in self.keypoints2bmask.values()] + [0])
        self.keypoints2bmask_tensor = torch.full((len(self.keypoints), max_overlap), -1, dtype=torch.int64)
        for k, v in self.keypoints2bmask.items():
            self.keypoints2bmask_tensor[k, :len(v)] = torch.tensor(v)

        # Used during sequence segmentation region unification
        if self.sequence_label_base is None:
            return
        self.sequence_labels = torch.arange(self.num_labels) + self.sequence_label_base

        # free up memory
        self.bmasks = None
        self.bboxes = None

    def match_regions(self, other_frame: "SequenceSegmentationFrame", min_keypoint_count=10):
        """
        Given `other_frame`, return dict mapping each bmask -> other bmask: score, defined as the ratio of matched 
        keypoints to total keypoints in the other bmask:
        :param other_frame: other frame to match against
        :param sequence_label_count: number of sequence labels in the entire sequence
        :param min_keypoint_count: min number of matches to regions in `other_frame` required for comparison
        """
        matches = self.matches[other_frame.name]
        matches_lookup = -torch.ones((len(self.keypoints)), dtype=torch.int64)
        matches_lookup[matches[:, 0]] = matches[:, 1]
        scores = {}
        for i in range(self.num_labels):
            if len(self.bmask2keypoints[i]) < min_keypoint_count:
                continue
            match_kpnts = matches_lookup[self.bmask2keypoints[i]]
            match_kpnts = match_kpnts[match_kpnts != -1]
            if len(match_kpnts) < min_keypoint_count:
                continue
            # tally raw matching counts
            labels = other_frame.keypoints2bmask_tensor[match_kpnts].flatten()
            labels = labels[labels != -1]
            if len(labels) == 0:
                continue
            counter = torch.bincount(labels, minlength=labels.max() + 1)
            counter[counter < min_keypoint_count] = 0
            # compute scores
            indices = torch.nonzero(counter).reshape(-1)
            scores[i] = {j.item(): counter[j].item() / len(match_kpnts) for j in indices}
        return scores
    
    def rasterize_bmasks(self, bmasks=None, descending=False) -> Mask:
        """
        Fuse bmasks w/ instance labels into unified instance mask. Rasterization order determined by instance label,
        which are ordered by mean area over frames containing the instance.
        """
        sorted, indices = torch.sort(self.instance_labels, descending=descending)
        if bmasks is None:
            bmasks = self.bmasks
        combined_mask = torch.zeros(bmasks[0].shape, dtype=torch.int32)
        for instance, mask in zip(sorted, bmasks[indices]):
            if instance == 0:
                continue
            combined_mask[mask] = instance.item()
        return combined_mask


@dataclass
class SequenceSegmentationConfig(InstantiateConfig):

    """ Target SequenceSegmentation to instantiate """
    _target: Type = field(default_factory=lambda: SequenceSegmentation)

    """ Parameters for RegistrationContainer and pair generation """
    container_config: Dict = field(default_factory=dict)

    """ 
    Parameters for sequence segmentation
    min_match_score    : min score for a match to be considered valid
    min_match_keypoints: min number of matched keypoints for a match to be considered valid
    min_track_len      : min number of frames a region must be tracked for it to be considered valid
    """
    min_match_score: float = 0.75
    min_match_keypoints: int = 4
    min_track_len: int = 10
    min_track_len_disjoint: int = 3

    """ Used for community detection to merge linked regions and split repeat regions in same frame """
    communities_threshold: int = 3
    communities_repeat_depth: int = 1

    """ Config for VPR and keypoint dense correspondence matchers """
    matcher_vpr_pairs_config: Dict = field(default_factory=dict)
    matcher_keypoints_config: Dict = field(default_factory=dict)

    """ Load/store tensor names/group from deeplake ds """
    tensor_name_mask  : str = 'sam/mask'
    tensor_name_bbox  : str = 'sam/bbox'
    tensor_name_output: str = 'sam'

    """ Degree thresholds for filtering bad VPR pairs based on camera view """
    Rdmat_max = 90
    Rdmat_min = 0

    def __post_init__(self):
        populate_dict(self.container_config        , {'descriptors_extractor_global': 'netvlad', 'matcher': 'loftr'})
        populate_dict(self.matcher_vpr_pairs_config, {'min_score': 0.25, 'min_num_matched': 10})
        populate_dict(self.matcher_keypoints_config, {'model': {'weights': 'indoor'}, 'max_error': 1, 'cell_size': 1}) # 5 for replica, 1 for record3d
        

class SequenceSegmentation(nn.Module):
    """
    Unify segmentation masks across a sequence of images via sfm keypoints and matches.
    """
    
    def __init__(self, config: InstantiateConfig, ds: Optional[deeplake.Dataset]=None):
        super().__init__()
        if ds is None:
            print("Warning: No dataset provided. Some features such as create_storage_tensors() may not be available.")
        self.ds = ds
        self.config = config
        
        
    def create_container(self):
        """
        Create reconstruction container and generate pairs and matches.
        """
        assert self.config.tensor_name_mask in self.ds, f'{self.config.tensor_name_mask} tensor not found in ds.'
        assert self.config.tensor_name_bbox in self.ds, f'{self.config.tensor_name_bbox} tensor not found in ds.'

        self.container = RegistrationContainer(
            self.ds, name='sequence_segmentation', overwrite=False,  **self.config.container_config
        )
        override_dict(   self.container.model_configs['matcher'],      self.config.matcher_keypoints_config)
        pairs_filename = self.container.retrieve_pairs(method='knn', **self.config.matcher_vpr_pairs_config)
        self.container.match(pairs=pairs_filename)
        print(self.config)

    def create_frames(self):
        """
        Build frames for each image according to keypoints and matches generated by the container.
        """
        names     = self.container.metadata['image_names']
        keypoints = self.container.get_keypoints()
        matches   = self.container.get_matches(return_reverse=True)

        self.sequence_label_counter = 0 # number of distinct labels seen in previous frames
        self.sequence_label2frame_index = {}
        self.frames = {}
        for index, (name, image, bmasks, bboxes) in tqdm(
            enumerate(zip(
                names, 
                self.ds['image'], 
                self.ds[self.config.tensor_name_mask],
                self.ds[self.config.tensor_name_bbox],
            )), desc='Generating frames'
        ):
            frame = SequenceSegmentationFrame(
                name, index,
                torch.from_numpy(image .numpy()),
                torch.from_numpy(bmasks.numpy()).permute(2, 0, 1), # (H, W, num_masks) -> (num_masks, H, W)
                torch.from_numpy(bboxes.numpy()),
                keypoints[name],
                matches  [name],
                sequence_label_base=self.sequence_label_counter
            )
            self.sequence_label2frame_index.update({label: index for label in range(
                self.sequence_label_counter, 
                self.sequence_label_counter + frame.num_labels
            )})
            self.sequence_label_counter += frame.num_labels
            self.frames[name] = frame

    def unite(self):
        """
        Generate connection graph denoting which regions are potentially connected across frames.
        """
        self.connection_edges = []
        self.connection_edges_weights = []

        poses = torch.from_numpy(self.ds['pose'].numpy())
        Rdmat, Tdmat = Rt_dists(poses, deg=True)
        Tdmat_max = Tdmat.max()
        camera_threshold = lambda t, lb, ub: (ub - lb) * (1 - t / Tdmat_max) + lb

        def unite_attempt(name1, name2):
            frame1 = self.frames[name1]
            frame2 = self.frames[name2]
            
            # Remove bad pairs based on camera view
            threshold = camera_threshold(Tdmat[frame1.index, frame2.index],
                lb=self.config.Rdmat_min,
                ub=self.config.Rdmat_max,
            )
            if Rdmat[frame1.index, frame2.index] > threshold:
                return

            # Compute match scores and build edges
            scores12 = frame1.match_regions(frame2, self.config.min_match_keypoints)
            scores21 = frame2.match_regions(frame1, self.config.min_match_keypoints)
            for ibmask1, matches in scores12.items():
                for ibmask2 in matches.keys():
                    # best buddy pairs above threshold
                    if (
                        scores12[ibmask1][ibmask2] > self.config.min_match_score and 
                        scores21[ibmask2][ibmask1] > self.config.min_match_score
                    ):
                        self.connection_edges.append((
                            frame1.sequence_labels[ibmask1].item(),
                            frame2.sequence_labels[ibmask2].item()
                        ))
                        self.connection_edges_weights.append(
                            scores12[ibmask1][ibmask2] + \
                            scores21[ibmask2][ibmask1]
                        )

        for name1, name2 in tqdm(self.container.get_pairs(), desc='Merging regions'):
            unite_attempt(name1, name2)

        self.connection_graph = igraph.Graph(edges=self.connection_edges, directed=False)
        self.connection_graph.simplify()
        print('Graph stats:')
        print('Nodes: ', self.connection_graph.vcount())
        print('Edges: ', self.connection_graph.ecount())

    def unite_postprocess(self):
        """
        Postprocess the connection graph to merge regions into instances.
        """
        self.union_find = UnionFind(self.sequence_label_counter)
        #for edge in tqdm(self.connection_graph.es, desc='Merging regions'):
        #    self.union_find.union(edge.source, edge.target)

        def union_community(community):
            check_valid_node = lambda n: 0 <= n < (self.sequence_label_counter)
            for i in range(1, len(community)):
                assert check_valid_node(community[0])
                assert check_valid_node(community[i])
                self.union_find.union(community[0], community[i])

        def detect_distinct_nodes(community):
            frame2nodes = defaultdict(list)
            for n in community:
                frame2nodes[self.sequence_label2frame_index[n]].append(n)
            nodes_distinct = max(frame2nodes.values(), key=lambda x: len(x))
            return nodes_distinct
        
        def detect_distinct_communities(community, depth):
            nodes_distinct = detect_distinct_nodes(community)
            if len(nodes_distinct) == 1:
                return [community]
            nodes2index = {n: i for i, n in enumerate(community)}
            index2nodes = {i: n for i, n in enumerate(community)}
            subgraph = self.connection_graph.subgraph(community)
            # WARNING: igraph community label propagation segfaults when called many times
            subcommunities = subgraph.community_label_propagation(
                initial=[nodes2index[n] if n in nodes_distinct else -1 for n in community], # negative entries denote unlabeled nodes, 
                fixed  =[                  n in nodes_distinct         for n in community]
            )
            subcommunities = [
                [index2nodes[i] for i in subcommunity] for subcommunity in subcommunities
            ]
            if depth == 0:
                return subcommunities
            ret = []
            for subcommunity in subcommunities:
                ret.extend(detect_distinct_communities(subcommunity, depth - 1))
            return ret

        communities = self.connection_graph.community_leiden(resolution_parameter=0, weights=self.connection_edges_weights)#.as_clustering()
        #communities = self.connection_graph.community_multilevel(resolution=0, weights=self.connection_edges_weights)
        count = 0
        for c in communities:
            if len(c) > self.config.communities_threshold:
                communities_refined = detect_distinct_communities(c, depth=self.config.communities_repeat_depth)
                for sc in communities_refined:
                    union_community(sc)
                count += len(communities_refined)
        print(f'Found {count} communities with more than {self.config.communities_threshold} members.')

    def compute_instance_labels_stats(self):
        """
        Compute statistics for instance labels over the entire sequence
        :returns: dict mapping label to num frames present in, total area, and mean area over containing frames
        """
        track_count = Counter()
        areas_count = Counter()
        for _, frame in self.frames.items():
            for label in frame.instance_labels.unique():
                track_count[label.item()] += 1
                areas_count[label.item()] += frame.bmask_areas[frame.instance_labels == label.item()].sum().item()        
        areas_mean = {k: v / track_count[k] for k, v in areas_count.items()}
        return track_count, areas_count, areas_mean

    def create_instance_labels(self):
        """
        Post process instance labels for each frame in the sequence.
        """
        # assign labels
        for _, frame in self.frames.items():
            frame.instance_labels = torch.zeros_like(frame.sequence_labels)
            for i, label in enumerate(frame.sequence_labels):
                instance = self.union_find.find(label.item())
                frame.instance_labels[i] = instance + 1 # 0 is background

        # compute instance stats
        frame_count, _, areas_mean = self.compute_instance_labels_stats()

        # remove instances that are too short
        for _, frame in self.frames.items():
            for i, label in enumerate(frame.instance_labels):
                if label.item() == 0: continue
                if frame_count[label.item()] < self.config.min_track_len:
                    frame.instance_labels[i] = 0
        '''
        # remove noise tracks
        track_indices = defaultdict(list)
        for _, frame in self.frames.items():
            for label in frame.instance_labels.unique():
                if label.item() == 0: continue
                track_indices[label.item()].append(frame.index)

        # sort tracks by length and keep longest track_percent fraction of tracks
        for label, indices in track_indices.items():
            indices.sort()
            chain, track = [], []
            for i in indices:
                if len(track) == 0 or i - track[-1] == 1:
                    track.append(i)
                    continue
                chain.append(track)
                track = []
            chain = sorted(chain, key=lambda x: len(x), reverse=True)
            chain = filter(lambda x: len(x) >= self.config.min_track_len_disjoint, chain)
            track_indices[label] = {i for track in chain for i in track}

        # remove filtered tracks
        for _, frame in self.frames.items():
            for i, label in enumerate(frame.instance_labels):
                if label.item() == 0: continue
                if frame.index not in track_indices[label.item()]:
                    frame.instance_labels[i] = 0
        '''
        
        # relabel instance labels to be consecutive ordered via area mean
        def enumerate_labels():
            labels = set()
            for frame in self.frames.values():
                labels.update(frame.instance_labels.unique().tolist())
            return labels
        
        self.labels = enumerate_labels()
        print(self.labels)
        if 0 in self.labels:
            self.labels.remove(0)
        self.labels = [0] + sorted(list(self.labels), key=lambda x: areas_mean[x], reverse=True)
        for frame in self.frames.values():
            instance_labels_reference = frame.instance_labels.clone()
            for i, label in enumerate(self.labels):
                frame.instance_labels[instance_labels_reference == label] = i
        self.labels = enumerate_labels()
        print(f'Found {len(self.labels)} instance labels, including background.')
        
    def forward(self) -> List[Dict]:
        """
        Returns per frame global instance labels.
        """
        self.create_container()
        self.create_frames()
        self.unite()
        self.unite_postprocess()
        self.create_instance_labels()

        self.ds.info[f'{self.config.tensor_name_output}/sequence_segmentation/num_classes'] = max(self.labels) + 1
        appended = []
        for frame in tqdm(self.frames.values()):
            bmasks = load_tensor_at(self.ds, self.config.tensor_name_mask, frame.index)
            bmasks = bmasks.permute(2, 0, 1) # (H, W, N) -> (N, H, W)
            appended.append({f'{self.config.tensor_name_output}/sequence_segmentation/mask': frame.rasterize_bmasks(bmasks)})
        return appended

    
    def append_features(self, ds: deeplake.Dataset) -> None:
        """
        Process sequence and append results to the deeplake dataset.
        """
        outputs = self()
        for output in outputs:
            output = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in output.items()
            }
            ds.append(output, skip_ok=True)

    
    def create_storage_tensors(self):
        """
        Create tensor in `ds` for storing sequence segmentation information.
        """
        self.ds.create_tensor(f'{self.config.tensor_name_output}/sequence_segmentation/mask', htype='generic', dtype='int32')