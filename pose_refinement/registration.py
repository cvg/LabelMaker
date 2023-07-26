import os
import cv2

import numpy as np
import open3d as o3d

from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_frame(path, name):
    image_file = os.path.join(path, 'images', name)
    depth_file = os.path.join(path, 'depth', name.replace('jpg', 'png'))

    image = np.asarray(Image.open(image_file))
    depth = np.asarray(Image.open(depth_file)).astype(np.float32) / 1000.0

    return image, depth

def get_rgbd(image, depth, depth_trunc=3.):
    
    depth = o3d.geometry.Image(depth)
    image = o3d.geometry.Image(image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image, 
                                                              depth,
                                                              depth_scale=1.0,
                                                              depth_trunc=depth_trunc,
                                                              convert_rgb_to_intensity=False)

    return rgbd



def lift_points(kps, image, depth, intrinsics, depth_trunc=3.):

    kps = np.floor(kps)
    kps = kps.astype(int)
    xx = kps[:, 0]
    yy = kps[:, 1]
    zz = depth[yy, xx]

    mask = (zz > 0.) & (zz < depth_trunc)    

    # resizing image to depth shape
    colors = image[yy, xx]

    xx = xx * zz
    yy = yy * zz
    pp = np.stack((xx, yy, zz))
    points = np.linalg.inv(intrinsics.intrinsic_matrix) @ pp
    points = points.T

    assert points.shape[0] == mask.shape[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=40))

    image = o3d.geometry.Image(image)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image, 
                                                                     depth, 
                                                                     depth_scale=1, 
                                                                     depth_trunc=depth_trunc, 
                                                                     convert_rgb_to_intensity=False)



    pcd_full = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, np.eye(4))
    pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=40))
    mask = mask.copy()

    return pcd, pcd_full, mask

def lift_points_full(image, depth, intrinsics, depth_trunc=3.):

  

    image = o3d.geometry.Image(image)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image, 
                                                                     depth, 
                                                                     depth_scale=1, 
                                                                     depth_trunc=depth_trunc, 
                                                                     convert_rgb_to_intensity=False)



    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, np.eye(4))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=40))

    return pcd

if __name__ == '__main__':
    from relative_registration import RelativeRegistration


    root_dir = '/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans'
    scene = 'scene0575_00'

    relative_registration = RelativeRegistration(root_dir, 
                                                scene, 
                                                matching='overlap', 
                                                downsample=1,
                                                stop_frame=-1,
                                                icp_type='point_to_plane',
                                                depth_threshold=3.,
                                                overlap_threshold=0.5,
                                                normal_estimation_radius=0.5,
                                                normal_estimation_max_nn=40,
                                                icp_max_iteration=100,
                                                init_with_relative=True)
    relative_registration.init()

    # setting up paths
    experiment_path = Path('output/registration_5')
    os.makedirs(experiment_path, exist_ok=True)
    image_path = experiment_path / 'images'
    depth_path = experiment_path / 'depth'
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    # downsampling images
    if os.listdir(image_path) !=  len(relative_registration.frames):
        print('Loading all images and copying them into the registration workspace')
        for i, image_name in tqdm(enumerate(relative_registration.frames), total=len(relative_registration.frames)):
            # image_name = f'{image_name}.jpg'
            image_file = os.path.join(relative_registration.root_dir, relative_registration.scene, 'data', 'color', image_name)
            depth_file = os.path.join(relative_registration.root_dir, relative_registration.scene, 'data', 'depth', image_name.replace('jpg', 'png'))
        
            image = np.asarray(Image.open(image_file)).astype(np.uint8)
            depth = np.asarray(Image.open(depth_file))

            # resizing image to depth shape
            h, w = depth.shape
            image = cv2.resize(image, (w, h))
            image = Image.fromarray(image)
            image.save(image_path / image_name)
            depth = Image.fromarray(depth)
            depth.save(depth_path / image_name.replace('.jpg', '.png'))

    fragment_size = 50
    fragment_overlap = 4

    # setup fragments

    fragment_end = fragment_size
    fragment_start = 0

    fragments = []

    while fragment_end < len(relative_registration.frames) - 1:
        fragments.append((fragment_start, fragment_end))
        fragment_start = fragment_end - fragment_overlap
        fragment_end = fragment_start + fragment_size

    if fragments[-1][1] < len(relative_registration.frames) - 1:
        fragments[-1] = (fragments[-1][0], len(relative_registration.frames) - 1)
        
    print(f'Generated {len(fragments)} fragments with last fragment {fragments[-1]}')

    print('Setup dense matching configs and workspace')

    dense_conf = match_dense.confs['loftr'] # don't use Aachen, 
    dense_conf['model']['weights'] = 'indoor'

    tmp_dir = experiment_path / 'hloc'
    os.makedirs(tmp_dir, exist_ok=True)
    sfm_pairs = tmp_dir / 'sfm-pairs.txt'
    sfm_pairs_loop = tmp_dir / 'sfm-pairs-loop.txt'
    sfm_pairs_loop_filtered = tmp_dir / 'sfm-pairs-loop-filtered.txt'

    image_dir = image_path
    image_list = []

    image_paths = image_dir.iterdir()
    image_paths = [f for f in image_paths if f.name.split('.')[0] != 'query']
    image_paths = sorted(image_paths, key=lambda x: int(x.name.split('.')[0]))

    feature_path = tmp_dir / 'features.h5'
    match_path = tmp_dir / 'matches.h5'

    image_paths = image_paths

    image_list_path = []
    indices = np.arange(len(image_paths))

    for index in indices:
        image_list.append(image_paths[index])
        image_list_path.append(
            str(Path(image_paths[index]).relative_to(image_dir)))
        

    # retrieval setup

    retrieval_conf = extract_features.confs['netvlad']

    retrieval_path = extract_features.main(retrieval_conf,
                                        image_dir,
                                        tmp_dir,
                                        image_list=image_list_path)

    pairs_from_retrieval.main(retrieval_path,
                            sfm_pairs_loop,
                            num_matched=50)


    # filtering matches according to fragments
    loop_pairs = []
    with open(sfm_pairs_loop, 'r') as file:
        for line in file:
            loop_pairs.append(line.rstrip().split(' '))

    loop_pairs_filtered = []
    for pair in loop_pairs:
        idx0, idx1 = int(pair[0].replace('.jpg', '')), int(pair[1].replace('.jpg', ''))
        if abs(idx0 - idx1) > 200:
            loop_pairs_filtered.append(pair)

    print(f'Recuded match set to {len(loop_pairs_filtered)} pairs')
    with open(sfm_pairs_loop_filtered, 'w') as file:
        for pair in loop_pairs_filtered:
            file.write(f'{pair[0]} {pair[1]}\n')
            
    match_graph = np.zeros((len(relative_registration.frames), len(relative_registration.frames)))

    mode = 'sequential'

    # setup matches for all fragments
    with open(sfm_pairs, 'w') as file:
        for frag in fragments:
            for i in range(frag[0], frag[1]):
                if mode == 'exhaustive':
                    for j in range(frag[0], frag[1]):
                        if i != j and match_graph[i, j] == 0:
                            file.write(f'{i}.jpg {j}.jpg\n')
                            match_graph[i, j] = 1
                            match_graph[j, i] = 1
                if mode == 'sequential':
                    file.write(f'{i}.jpg {i + 1}.jpg\n')
    
    feature_path, match_path = match_dense.main(conf=dense_conf,
                                            pairs=sfm_pairs,
                                            image_dir=image_dir,
                                            export_dir=tmp_dir)
    
    _, match_loop_path = match_dense.main(conf=dense_conf,
                                      pairs=sfm_pairs_loop_filtered,
                                            image_dir=image_dir,
                                            export_dir=tmp_dir)
    
    kp_database = {}

    for image_name in range(0, len(relative_registration.frames)):
        image_name = f'{image_name}.jpg'
        try:
            keypoints = get_keypoints(feature_path, image_name)
            kp_database[image_name] = keypoints
        except Exception as e:
            print(e)

    # registration settings and options
    mu, sigma = 0, 0.05
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    estimation_full = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    max_correspondence_distance = 0.1
    max_correspondence_distance_full = 0.1
    max_correspondence_distance_full_small = 0.01
    depth_trunc = 2.5
    voxel_size = 0.02   

    odometry_option = o3d.pipelines.odometry.OdometryOption()

    # load all pairs
    pairs = []
    with open(sfm_pairs, 'r') as file:
        for line in file:
            line = line.rstrip().split(' ')
            pairs.append((line[0], line[1]))

    fragments_reconstruction = []

    for frag_idx, frag in enumerate(fragments):
        # iterate over all loop closure pairs and load files
        
        odometry = np.eye(4)
        pcd = None
        
        edges = []
        intrinsics = relative_registration.intrinsics
        for idx, (name0, name1) in tqdm.tqdm(enumerate(pairs), total=len(pairs)):
            idx0, idx1 = int(name0.replace('.jpg', '')), int(name1.replace('.jpg', ''))
                    
            if idx0 < frag[0] or idx0 > frag[1] - 1 or idx1 < frag[0] or idx1 > frag[1] - 1:
                continue
                    
            image0, depth0 = load_frame(experiment_path, name0)
            image1, depth1 = load_frame(experiment_path, name1)            

            rgbd0 = get_rgbd(image0, depth0, depth_trunc)
            rgbd1 = get_rgbd(image1, depth1, depth_trunc)
            
            kps0 = kp_database[name0]
            kps1 = kp_database[name1]

            # lift the two images
            pcd0, pcd0_full, mask0 = lift_points(kps0, image0, depth0, intrinsics, depth_trunc=depth_trunc)
            pcd1, pcd1_full, mask1 = lift_points(kps1, image1, depth1, intrinsics, depth_trunc=depth_trunc)

            matches, scores = get_matches(match_path, name0, name1)

            # extract keypoints and and only use them where mask is set
            kps0_m = kps0[mask0]
            kps1_m = kps1[mask1]

            # compute index correction
            match_index_correction_0 = np.cumsum(mask0 == 0)
            match_index_correction_1 = np.cumsum(mask1 == 0)

            valid_matches = (mask0[matches[:, 0]]) & (mask1[matches[:, 1]])

            matches[:, 0] = matches[:, 0] - match_index_correction_0[matches[:, 0]]
            matches[:, 1] = matches[:, 1] - match_index_correction_1[matches[:, 1]]

            # only use keypoints that are matched
            kps0_m_f = kps0_m[matches[:, 0]]
            kps1_m_f = kps1_m[matches[:, 1]]

            matches_m = matches[valid_matches, :]

            pcd0_m = o3d.geometry.PointCloud()
            pcd0_m.points = o3d.utility.Vector3dVector(np.asarray(pcd0.points)[mask0, :])
            pcd0_m.colors = o3d.utility.Vector3dVector(np.asarray(pcd0.colors)[mask0, :] / 255.)

            pcd1_m = o3d.geometry.PointCloud()
            pcd1_m.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[mask1, :])
            pcd1_m.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors)[mask1, :] / 255.)

            correspondences = o3d.utility.Vector2iVector(matches_m)

            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcd0_m, 
                                                                                            pcd1_m, 
                                                                                            correspondences, 
                                                                                            max_correspondence_distance,
                                                                                            ransac_n=5)

            transformation_sparse = result.transformation
            transformation_sparse = estimation.compute_transformation(pcd0_m, pcd1_m, result.correspondence_set)
            result = o3d.pipelines.registration.evaluate_registration(pcd0_m, pcd1_m, max_correspondence_distance, transformation_sparse)

            # pcd0_full = pcd0_full.voxel_down_sample(voxel_size=voxel_size)
            # pcd1_full = pcd1_full.voxel_down_sample(voxel_size=voxel_size)
            
            # result_full = o3d.pipelines.registration.registration_icp(pcd0_full, 
            #                                                           pcd1_full,
            #                                                           max_correspondence_distance_full, 
            #                                                           init=transformation_sparse, 
            #                                                           estimation_method=estimation_full)
    # 
            # result_full = o3d.pipelines.registration.registration_icp(pcd0_full, 
            #                                                           pcd1_full,
            #                                                           max_correspondence_distance_full_small, 
            #                                                           init=result_full.transformation, 
            #                                                           estimation_method=estimation_full)
    # 


            transformation_sparse = np.copy(transformation_sparse)
            transformation = transformation_sparse

            
            information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(pcd0_full,
                                                                                            pcd1_full,
                                                                                            max_correspondence_distance_full, 
                                                                                            transformation)

            
            odo_init = np.eye(4)
            
            [success_color_term, trans_color_term,
            info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd0, rgbd1,
                intrinsics, transformation_sparse,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), odometry_option)
            
            [success_hybrid_term, trans_hybrid_term,
            info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd0, rgbd1,
                intrinsics, transformation_sparse,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), odometry_option)
            
            
            # if success_color_term:
                # print("Using RGB-D Odometry")
                # print(trans_color_term)
                # pcd_ = pcd0_full.transform(trans_color_term)
                # pcd_color = pcd_  + pcd1_full
                # pcd_color = pcd_color.voxel_down_sample(0.02)
                
                # o3d.visualization.draw_plotly([pcd_color])

            # if success_hybrid_term:
                # print("Using Hybrid RGB-D Odometry")
                # print(trans_hybrid_term)
                # pcd_ = pcd0_full.transform(trans_hybrid_term)
                # pcd_color = pcd_  + pcd1_full
                # pcd_color = pcd_color.voxel_down_sample(0.02)
                # o3d.visualization.draw_plotly([pcd_color])
        
            transformation = trans_hybrid_term if success_hybrid_term else trans_color_term
        
            transformation = trans_color_term
        
            

            n_full_correspondences = np.asarray(result.correspondence_set).shape[0]
            n0_full = np.asarray(pcd0_full.points).shape[0]
            n1_full = np.asarray(pcd1_full.points).shape[0]


            max_ratio = max(n_full_correspondences / n0_full, n_full_correspondences / n1_full)


            edges.append({'source': name0, 
                        'target': name1, 
                        'transformation': transformation.tolist(),
                        'transformation_sparse': transformation_sparse.tolist(),
                        'information': info.tolist(),
                        'max_ratio': max_ratio,
                        'n_full_correspondences': n_full_correspondences})
            
            if pcd is None:
                pcd = pcd0_full
                
            odometry = np.dot(transformation, odometry)
            pcd1_full = pcd1_full.transform(np.linalg.inv(odometry))
            pcd = pcd + pcd1_full
            pcd = pcd.voxel_down_sample(voxel_size)
            
            
        # o3d.visualization.draw_plotly([pcd])
        fragments_reconstruction.append({'edges': deepcopy(edges)})


    fragments_updated = []

    for frag_idx, frag in enumerate(fragments_reconstruction):
            
        odometry = [np.eye(4)]
        last_idx = 0
        
        edges = deepcopy(frag['edges'])
        
        idx_offset = None
        
        for e in edges:
            name0 = e['source']
            name1 = e['target']

            idx0 = int(name0.replace('.jpg', ''))
            idx1 = int(name1.replace('.jpg', ''))
            
            if idx_offset is None:
                idx_offset = idx0
            
            if idx1 == idx0 + 1:
                if last_idx < idx1 - 1:
                    print('Problem', idx1, last_idx)
                odometry.append(np.dot(e['transformation'], odometry[-1]))
                last_idx = idx1
                
        pose_graph = o3d.pipelines.registration.PoseGraph()
        
        frag['odometry'] = deepcopy(odometry)

        for idx, p in enumerate(odometry):
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(p)))
            
        for e in edges:
            name0, name1 = e['source'], e['target']
            idx0, idx1 = int(name0.replace('.jpg', '')), int(name1.replace('.jpg', ''))

            uncertain = False if idx0 + 1 == idx1 else True
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(idx0 - idx_offset,
                                                                            idx1 - idx_offset,
                                                                            np.asarray(e['transformation']),
                                                                            np.asarray(e['information']),
                                                                            uncertain=uncertain))
            
        print("Optimizing PoseGraph ...")
        edge_prune_threshold=0.25
        icp_threshold_fine = 0.01
        option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=icp_threshold_fine,
                                                                    edge_prune_threshold=edge_prune_threshold,
                                                                    reference_node=0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(pose_graph,
                                                        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                        option)
            
        
        frag['poses'] = [np.linalg.inv(n.pose) for n in pose_graph.nodes]
        
        tsdf = o3d.pipelines.integration.ScalableTSDFVolume(sdf_trunc=0.06, 
                                                            voxel_length=0.02, 
                                                            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        intrinsics = relative_registration.intrinsics
        path = os.path.join('../output/registration_test')
        mesh_path = 'meshes_pgo'
        os.makedirs(mesh_path, exist_ok=True)

        

        for idx, n in enumerate(pose_graph.nodes):
            # print(idx, '/', len(pose_graph.nodes))
            
            if idx + idx_offset > len(relative_registration.frames):
                continue
            
            try:
                name = idx + idx_offset
                name = f'{name}.jpg'
                pose = np.linalg.inv(n.pose)


                image, depth = load_frame(path, name)
                rgbd = get_rgbd(image, depth, depth_trunc=3.)
                tsdf.integrate(rgbd, intrinsics, odometry[idx])
            except Exception as e:
                print(e)
        
        frag['tsdf'] = deepcopy(tsdf)
        
        fragments_updated.append(deepcopy(frag))

    fragment_matches = [(i, i+1) for i in range(0, len(fragments) - 1)]

    fragment_edges = []

    meshes = []

    odometry = np.eye(4)

    for i, f_m in enumerate(fragment_matches):
        print(f_m)
        pcd0 = fragments_updated[f_m[0]]['tsdf'].extract_point_cloud()
        pcd1 = fragments_updated[f_m[1]]['tsdf'].extract_point_cloud()
        
        init = fragments_updated[f_m[0]]['odometry'][-1]
        
        result = o3d.pipelines.registration.registration_icp(pcd0, 
                                                            pcd1,
                                                            0.1, 
                                                            init=init, 
                                                            estimation_method=estimation_full)
        
        result_small = o3d.pipelines.registration.registration_icp(pcd0, 
                                                                pcd1,
                                                                0.01, 
                                                                init=result.transformation, 
                                                                estimation_method=estimation_full)
        
        transformation = result_small.transformation
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(pcd0,
                                                                                        pcd1,
                                                                                        0.01, 
                                                                                        transformation)

        
        pcd0_ = pcd0.transform(transformation)
        
        
        odometry = np.dot(transformation, odometry)
        
        
        # o3d.visualization.draw_plotly([mesh0, mesh1])
    #     o3d.visualization.draw_plotly([pcd0_, pcd1])

        fragment_edges.append({'source': f_m[0],
                            'target': f_m[1],
                            'transformation': transformation,
                            'information': information})
        

    # fragment loop closures

    matched_fragments = np.zeros((len(fragments), len(fragments)))

    fragment_loop_edges = []

    def frame_to_fragment(idx, size=50, overlap=4):
        return idx // (size - overlap)
        
    depth_trunc = 3.

    for (name0, name1) in tqdm.tqdm(loop_pairs_filtered, total=len(loop_pairs_filtered)):
        idx0, idx1 = int(name0.replace('.jpg', '')), int(name1.replace('.jpg', ''))
        
        
        frag0_idx = min(frame_to_fragment(idx0, size=fragment_size, overlap=fragment_overlap), len(fragments) - 1)
        frag1_idx = min(frame_to_fragment(idx1, size=fragment_size, overlap=fragment_overlap), len(fragments) - 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    
        
        if matched_fragments[frag0_idx, frag1_idx] == 1:
            continue
        
        # skip match if continuous fragments
        if abs(frag0_idx - frag1_idx) < 2:
            continue
            
        image0, depth0 = load_frame(experiment_path, name0)
        image1, depth1 = load_frame(experiment_path, name1)            
        
        kps0 = kp_database[name0]
        kps1 = kp_database[name1]

        # lift the two images
        pcd0, pcd0_full, mask0 = lift_points(kps0, image0, depth0, intrinsics, depth_trunc=depth_trunc)
        pcd1, pcd1_full, mask1 = lift_points(kps1, image1, depth1, intrinsics, depth_trunc=depth_trunc)

        matches, scores = get_matches(match_loop_path, name0, name1)

        # extract keypoints and and only use them where mask is set
        kps0_m = kps0[mask0]
        kps1_m = kps1[mask1]

        # compute index correction
        match_index_correction_0 = np.cumsum(mask0 == 0)
        match_index_correction_1 = np.cumsum(mask1 == 0)

        valid_matches = (mask0[matches[:, 0]]) & (mask1[matches[:, 1]])

        matches[:, 0] = matches[:, 0] - match_index_correction_0[matches[:, 0]]
        matches[:, 1] = matches[:, 1] - match_index_correction_1[matches[:, 1]]

        # only use keypoints that are matched
        kps0_m_f = kps0_m[matches[:, 0]]
        kps1_m_f = kps1_m[matches[:, 1]]

        matches_m = matches[valid_matches, :]

        pcd0_m = o3d.geometry.PointCloud()
        pcd0_m.points = o3d.utility.Vector3dVector(np.asarray(pcd0.points)[mask0, :])
        pcd0_m.colors = o3d.utility.Vector3dVector(np.asarray(pcd0.colors)[mask0, :] / 255.)

        pcd1_m = o3d.geometry.PointCloud()
        pcd1_m.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[mask1, :])
        pcd1_m.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors)[mask1, :] / 255.)


    #     if matches_m.shape[0] < 400:
    #         continue
        
        correspondences = o3d.utility.Vector2iVector(matches_m)
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcd0_m, 
                                                                                        pcd1_m, 
                                                                                        correspondences, 
                                                                                        max_correspondence_distance,
                                                                                        ransac_n=5)

    
        transform_init = result.transformation
        
        frag0_start = fragments[frag0_idx][0]
        frag1_start = fragments[frag1_idx][0]
        
        local_pose0 = fragments_updated[frag0_idx]['odometry'][idx0 - frag0_start]
        local_pose1 = fragments_updated[frag1_idx]['odometry'][idx1 - frag1_start]

        
        pcd0_full = pcd0_full.transform(transform_init)
        pdc0_full = pcd0_full.voxel_down_sample(0.02)
        pcd1_full = pcd1_full.voxel_down_sample(0.02)
        
    #     o3d.visualization.draw_plotly([pcd0_full, pcd1_full])
        
        transform_init = np.dot(np.linalg.inv(local_pose1), np.dot(transform_init, local_pose0))

        
        
        pcd0_full = fragments_updated[frag0_idx]['tsdf'].extract_point_cloud()
        pcd1_full = fragments_updated[frag1_idx]['tsdf'].extract_point_cloud()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
        result = o3d.pipelines.registration.registration_icp(pcd0_full, 
                                                            pcd1_full,
                                                            0.1, 
                                                            init=transform_init, 
                                                            estimation_method=estimation_full)
        
        result_small = o3d.pipelines.registration.registration_icp(pcd0_full, 
                                                            pcd1_full,
                                                            0.01, 
                                                            init=result.transformation, 
                                                            estimation_method=estimation_full)
        
        
        transformation = result_small.transformation
        
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(pcd0,
                                                                                        pcd1,
                                                                                        0.01, 
                                                                                        transformation)

        
        pcd0_ = pcd0_full.transform(transformation)
        
        mesh0 = fragments_updated[frag0_idx]['tsdf'].extract_triangle_mesh().transform(transformation)
        mesh1 = fragments_updated[frag1_idx]['tsdf'].extract_triangle_mesh()

    # o3d.visualization.draw_plotly([mesh0, mesh1])
    #     o3d.visualization.draw_plotly([pcd0_, pcd1_full])

        fragment_loop_edges.append({'source': frag0_idx,
                                    'target': frag1_idx,
                                    'transformation': transformation,
                                    'information': information})
        matched_fragments[frag0_idx, frag1_idx] = 1
        matched_fragments[frag1_idx, frag0_idx] = 1

    
    
        

            
            
        
            
                    