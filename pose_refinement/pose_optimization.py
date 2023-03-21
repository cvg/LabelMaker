import argparse

from absolute_registration import AbsoluteRegistration
from relative_registration import RelativeRegistration

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans/')
    parser.add_argument('--scene', type=str, default='scene0000_00')
    args = parser.parse_args()
    return args

def main(args):
    relative_registration = RelativeRegistration(args.root_dir, args.scene)
    relative_registration.init()
    relative_registration.run()

    absolute_registration = AbsoluteRegistration()
    absolute_registration.init(relative_registration.nodes)
    absolute_registration.run()
    
if __name__ == '__main__':
    args = arg_parser()
    main(args)