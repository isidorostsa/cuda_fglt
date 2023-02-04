# Commands make the project

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## Commands to start the container

```bash
sudo docker run --rm --gpus all -it -e CUDBG_USE_LEGACY_DEBUGGER=1 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $PWD:/cuda_project -w /cuda_project nvidia/cuda:12.0.1-devel-ubuntu20.04
```

- -e means export CUDBG_USE_LEGACY_DEBUGGER=1
- --cap-add=SYS_PTRACE --security-opt seccomp=unconfined is for __gdb__
- -v $PWD:/cuda_project -w /cuda_project is for mounting current directory to /cuda_project in container
- nvidia/cuda:12.0.1-devel-ubuntu20.04 is the image name (Works in Pop!_OS 22.04)
- add __-it__ before __nvidia/cuda:12.0.1-devel-ubuntu20.04__ to run the container in interactive mode

## Commands to compile

### Compile

```bash
make {ARGS}
```

with ARGS as:

- BUILD_TYPE=debug or BUILD_TYPE=release
- BUILD_ENV=container or BUILD_ENV=host

### Run

```bash
./bin/fglt
```

or

```bash
./bin_debug/fglt
```