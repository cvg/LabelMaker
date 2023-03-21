import gin
import os

import numpy as np
import open3d as o3d

@gin.configurable
class AbsoluteRegistration:
    def __init__(self):
        pass
    
    def init(self, nodes):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

        for node in nodes:
            self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(node.pose))
            
            for edge in node.edges:
                print(node.idx, edge.idx)
                self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(node.idx,
                                                                                 edge.idx,
                                                                                 edge.transformation,
                                                                                 edge.information,
                                                                                 uncertain=edge.uncertain))

        print('Built pose graph with {} nodes and {} edges'.format(len(self.pose_graph.nodes), len(self.pose_graph.edges)))
        print(self.pose_graph)


    def run(self):
        voxel_size=0.007
        max_correspondence_distance_fine = voxel_size * 10.

        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(self.pose_graph,
                                                           o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                           o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                           option)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, node in enumerate(self.pose_graph.nodes):
            pose_file = os.path.join(save_dir, f'{str(i).zfill(5)}.txt')
            np.savetxt(pose_file, node.pose)
