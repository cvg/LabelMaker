import os  
  
def rename(file_folder,suffix,rename2longer=False):
    if rename2longer:
        #  rename the filename from "1.suffix" to "000001.suffix"
        for filename in os.listdir(file_folder):     
            if filename.endswith(suffix):         
                base, ext = os.path.splitext(filename)         
                number = int(base)  
                formatted_number = format(number, '06d')       
                new_filename = formatted_number + ext           
                old_file_path = os.path.join(file_folder, filename)  
                new_file_path = os.path.join(file_folder, new_filename)    
                os.rename(old_file_path, new_file_path)  
                print(f'rename {old_file_path} as {new_file_path}')  
    else:
        #  rename the filename from "000001.suffix" to "1.suffix"
        for filename in os.listdir(file_folder):     
            if filename.endswith(suffix):         
                base, ext = os.path.splitext(filename)         
                new_base = str(int(base))          
                new_filename = new_base + ext  
                old_file_path = os.path.join(file_folder, filename)  
                new_file_path = os.path.join(file_folder, new_filename)    
                os.rename(old_file_path, new_file_path)  
                print(f'rename {old_file_path} as {new_file_path}')  
   

if __name__ == "__main__":
    scenes = ['./scene0164_02/','./scene0474_01/','./scene0518_00/','./scene0458_00/','./scene0000_00/']   
    folders_suffix = {'color':'.jpg', 'depth':'.png','intrinsic':'.txt','pose':'.txt','resized-instance':'.png'}
    # folders_suffix = {'neus_lifted_instance_2':'.png','neus_lifted_semantic_2':'.png','nerfacto_lifted_instance-2':'.png','nerfacto_lifted_semantic-2':'.png',
                    #   'resized_manual_label':'.png','resized-instance':'.png'}

    for scene in scenes:
        for key in folders_suffix.keys():
            path = scene + key + '/'
            print(path,folders_suffix[key])
            rename(path,folders_suffix[key],rename2longer=False)