import argparse
import os

from absolute_registration import AbsoluteRegistration
from relative_registration import RelativeRegistration
from bundle_adjustment import BundleAdjustment

from config import load_config
from utils import save_to_colmap, load_from_colmap, save_to_scannet

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans/')
    parser.add_argument('--scene', type=str, default='scene0000_00')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--config')
    parser.add_argument('--gin_params', default=[], nargs='+')
    args = parser.parse_args()
    return args

def main(args):
    config = load_config(args)

    # setup output directory
    output_path = os.path.join(args.output_dir, config.name, args.scene)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    relative_registration = RelativeRegistration(args.root_dir, args.scene)
    relative_registration.init()
    relative_registration.run()
    relative_registration.save(output_path + '/relative')
    relative_registration.save_nodes(output_path + '/relative_pose_graph')

    absolute_registration = AbsoluteRegistration()
    absolute_registration.init(relative_registration.nodes)
    absolute_registration.run()
    absolute_registration.save(output_path + '/absolute')

    save_to_colmap(pose_graph=absolute_registration.pose_graph, 
                   nodes=absolute_registration.raw_nodes, 
                   intrinsics=relative_registration.intrinsics.intrinsic_matrix,
                   save_dir=output_path + '/colmap',
                   image_dir=args.root_dir + '/' + args.scene + '/data/color',
                   invert_pose=True) # invert if pose graph poses no inversion if scannet poses
    

    refinement = BundleAdjustment(output_path + '/colmap')
    refinement.init()
    refinement.run()
    refinement.save()

    nodes = load_from_colmap(output_path + '/colmap/triangulation_ba')
    save_to_scannet(nodes, output_path + '/scannet')


if __name__ == '__main__':
    args = arg_parser()
    main(args)