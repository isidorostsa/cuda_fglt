Command to run container:

```bash
sudo docker run --rm --gpus all -it -e CUDBG_USE_LEGACY_DEBUGGER=1 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $PWD:/cuda_project -w /cuda_project nvidia/cuda:11.6.2-devel-ubuntu20.04
```

- -e means export CUDBG_USE_LEGACY_DEBUGGER=1
- --cap-add=SYS_PTRACE --security-opt seccomp=unconfined is for __gdb__
- -v $PWD:/cuda_project -w /cuda_project is for mounting current directory to /cuda_project in container
- nvidia/cuda:11.6.2-devel-ubuntu20.04 is the image name (Works in Pop!_OS 22.04)