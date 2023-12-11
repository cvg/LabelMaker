# need wget unzip, this will download the dataset to the directory you are using
wget -O replica_semantic_nerf.zip "https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip replica_semantic_nerf.zip && rm replica_semantic_nerf.zip
mv Replica_Dataset Replica_Dataset_Semantic_Nerf
cd Replica_Dataset_Semantic_Nerf
unzip \*.zip && rm -rf *.zip


