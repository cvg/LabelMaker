from dask_jobqueue import SLURMCluster

if __name__ == "__main__":

  cluster = SLURMCluster(
    n_workers=1,
    memory="34GiB", # total memory, the memory limit is "memory" / "cores"
    processes=3, # dask worker number
    job_cpu=7, # cpus-per-task
    # job_mem='48GiB',# no use
    cores=5, # dask worker number, not useful when process is set
    # account='guanji',
    # memory_limit='34GiB',
    interface='access',
    walltime="00:10:00",
    job_extra_directives=[
                    "--gpus=rtx_3090:1",
                    "--mem-per-cpu=12G",
                     "--output=/cluster/home/guanji/LabelMaker/job%j.out",
                ],
                job_script_prologue=[
                   "module load eth_proxy",
                'export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"',
                'env_name=labelmaker',
                'eval "$(conda shell.bash hook)"',
                'conda activate $env_name',
                'conda_home="$(conda info | grep "active env location : " | ',
                'export AM_I_DOCKER=1',
                'export CUDA_HOST_COMPILER="$conda_home/bin/gcc"',
                'export CUDA_PATH="$conda_home"',
                'export CUDA_HOME=$CUDA_PATH',
                'export NLTK_DATA="${ENV_FOLDER}/../3rdparty/nltk_data"',
            ],
            job_directives_skip=["--mem"],
          # python=' '.join([
          #           "singularity exec --nv",
          #           "--bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints",
          #           "--bind $LABELMAKER_REPO/env_v2:/LabelMaker/env_v2",
          #           "--bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker",
          #           "--bind $LABELMAKER_REPO/testing:/LabelMaker/testing",
          #           "--bind $LABELMAKER_REPO/models:/LabelMaker/models",
          #           "--bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts"
          #           "--bind $LABELMAKER_REPO/.gitmodules:/LabelMaker/.gitmodules",
          #           "--bind $TMPDIR/.cache:$HOME/.cache",
          #           "--bind $source_dir:/source",
          #           "--bind $target_dir:/target",
          #           "/cluster/project/cvg/labelmaker/labelmaker_20231227.simg",
          #           "/miniconda3/envs/labelmaker/bin/python",
          #       ]),

  )

  print(cluster.job_script())
