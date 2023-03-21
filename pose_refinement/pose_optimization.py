import argparse
import os

from absolute_registration import AbsoluteRegistration
from relative_registration import RelativeRegistration

from config import load_config

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

    absolute_registration = AbsoluteRegistration()
    absolute_registration.init(relative_registration.nodes)
    absolute_registration.run()
    absolute_registration.save(output_path + '/absolute')

    
if __name__ == '__main__':
    args = arg_parser()
    main(args)