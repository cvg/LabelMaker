```sh
docker build --tag labelmaker-env-test-16.04 -f docker/ubuntu16.04+miniconda.dockerfile .

docker run --gpus all -i --rm -v ./env_v2:/LabelMaker/env_v2 -v ./models:/LabelMaker/models -v ./labelmaker:/LabelMaker/labelmaker -v ./checkpoints:/LabelMaker/checkpoints -v ./testing:/LabelMaker/testing  -v ./.gitmodules:/LabelMaker/.gitmodules -t labelmaker-env-test-16.04 /bin/bash

docker build --tag labelmaker-env-test-20.04 -f docker/ubuntu20.04+miniconda.dockerfile .

docker run --gpus all -i --rm -v ./env_v2:/LabelMaker/env_v2 -v ./models:/LabelMaker/models -v ./labelmaker:/LabelMaker/labelmaker -v ./checkpoints:/LabelMaker/checkpoints  -v ./testing:/LabelMaker/testing -v ./.gitmodules:/LabelMaker/.gitmodules -t labelmaker-env-test-20.04 /bin/bash
```
