```sh
docker build --tag labelmaker-env-test-16.04 -f docker/ubuntu16.04+miniconda.dockerfile .
docker run --gpus all -i --rm -t labelmaker-env-test-16.04 /bin/bash
docker build --tag labelmaker-env-test-20.04 -f docker/ubuntu20.04+miniconda.dockerfile .
docker run --gpus all -i --rm -t labelmaker-env-test-20.04 /bin/bash
```
