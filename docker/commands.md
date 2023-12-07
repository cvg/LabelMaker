## Docker image based on Ubuntu 16.04
```sh
# Build
docker build --tag labelmaker-env-16.04 -f docker/ubuntu16.04+miniconda.dockerfile .

# Run
docker run \
  --gpus all \
  -i --rm \
  -v ./env_v2:/LabelMaker/env_v2 \
  -v ./models:/LabelMaker/models \
  -v ./labelmaker:/LabelMaker/labelmaker \
  -v ./checkpoints:/LabelMaker/checkpoints \
  -v ./testing:/LabelMaker/testing \
  -v ./.gitmodules:/LabelMaker/.gitmodules \
  -t labelmaker-env-16.04 /bin/bash
```

## Docker image based on Ubuntu 20.04

```sh
# Build
docker build --tag labelmaker-env-20.04 -f docker/ubuntu20.04+miniconda.dockerfile .

# Run
docker run \
  --gpus all \
  -i --rm \
  -v ./env_v2:/LabelMaker/env_v2 \
  -v ./models:/LabelMaker/models \
  -v ./labelmaker:/LabelMaker/labelmaker \
  -v ./checkpoints:/LabelMaker/checkpoints \
  -v ./testing:/LabelMaker/testing \
  -v ./.gitmodules:/LabelMaker/.gitmodules \
  -t labelmaker-env-20.04 /bin/bash
```
