# if not specify downlaod directory, use current directory
if [ -z "$1" ]; then
  download_dir='.'
else
  download_dir=$1
fi

cd $download_dir

wget -O replica_semantic_nerf.zip "https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip replica_semantic_nerf.zip && rm replica_semantic_nerf.zip
mv Replica_Dataset Replica_Dataset_Semantic_Nerf
cd Replica_Dataset_Semantic_Nerf
unzip \*.zip && rm -rf *.zip

cd --
cd --
