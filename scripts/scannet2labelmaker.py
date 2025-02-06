import argparse
import os, sys
import os, struct
import numpy as np
import shutil
import zlib
import imageio.v2 as imageio
import cv2
import png

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():
  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)
    

  def load(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      # self.sensor_name = ''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.sensorname = f.read(strlen)
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)


  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    # if not os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      #imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
      with open(os.path.join(output_path, str(f).zfill(6) + '.png'), 'wb') as f: # write 16-bit
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
        depth = depth.reshape(-1, depth.shape[1]).tolist()
        writer.write(f, depth)

  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    # if not os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f).zfill(6) + '.jpg'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    # if not os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f).zfill(6) + '.txt'))


  def export_intrinsics(self, output_path, original_intrisic = None, resize = None):
    # if not os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)
    print('exporting camera intrinsics to', output_path)
    if resize == None:
      self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
      self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
      self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
      self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))
    else:
      # if not os.path.exists(original_intrisic):
      shutil.rmtree(original_intrisic, ignore_errors=True)
      os.makedirs(original_intrisic)
      self.save_mat_to_file(self.intrinsic_color, os.path.join(original_intrisic, 'intrinsic_color.txt'))
      self.save_mat_to_file(self.extrinsic_color, os.path.join(original_intrisic, 'extrinsic_color.txt'))
      self.save_mat_to_file(self.intrinsic_depth, os.path.join(original_intrisic, 'intrinsic_depth.txt'))
      self.save_mat_to_file(self.extrinsic_depth, os.path.join(original_intrisic, 'extrinsic_depth.txt'))
      w = resize[1]/1296
      h = resize[0]/968
      intrinsic_color = self.intrinsic_color[:3, :3]
      scaled_intrinsic_color = np.diag([w,h,1])@intrinsic_color
      for i in range(0, len(self.frames)):
        target = os.path.join(output_path,str(i).zfill(6)+'.txt')
        # print(target)
        self.save_mat_to_file(scaled_intrinsic_color,target)
      
      
def arg_parser():
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--scan_dir', required=True, help='path to scan to read')
    parser.add_argument('--target_dir', required=True, help='path to output folder')
    parser.add_argument('--export_depth_images', dest='export_depth_images')
    parser.add_argument('--export_color_images', dest='export_color_images')
    parser.add_argument('--export_poses', dest='export_poses')
    parser.add_argument('--export_intrinsics', dest='export_intrinsics')
    parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True)
    return parser.parse_args()


def main():
    
    args = arg_parser()
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    # load the data
    filename = os.path.join(args.scan_dir,str(os.path.basename(args.scan_dir.rstrip(os.sep)))+'.sens')
    sys.stdout.write('loading %s...' % filename )
    sd = SensorData(filename)
    sys.stdout.write('loaded!\n')
    
    #  copy RGB mesh file
    source_mesh = os.path.join(args.scan_dir,os.path.basename(args.scan_dir.rstrip(os.sep))+'_vh_clean.ply')
    destination_mesh = os.path.join(args.target_dir, 'mesh.ply')
    try:
        with open(source_mesh, 'rb') as src:
            with open(destination_mesh, 'wb') as dst:
                dst.write(src.read())
        print(f"file {source_mesh} was copied to {destination_mesh}")
    except FileNotFoundError:
            print(f"{source_mesh} not found")
            
    # !!!resize image for labelmaker usage
    resize = [480,640]    
    if args.export_depth_images:
        sd.export_depth_images(os.path.join(args.target_dir, 'depth'))
    if args.export_color_images:
        sd.export_color_images(os.path.join(args.target_dir, 'color'), image_size = resize)
    if args.export_poses:
        sd.export_poses(os.path.join(args.target_dir, 'pose'))
    if args.export_intrinsics:
        sd.export_intrinsics(os.path.join(args.target_dir, 'intrinsic'), os.path.join(args.target_dir, 'original_intrinsic'),resize=resize)
    
    
if __name__ == '__main__':
    main()
    
    
    

