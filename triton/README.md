## 1. Start Triton with Custom CUDA Device and Selection model

```bash
CUDA_VISIBLE_DEVICES=5 tritonserver --model-repository=/workspace/models/ --model-control-mode=explicit --load-model=car_brands_accuracy
````

## 2. Create a Symbolic Link to Rename the Triton Process

After launching the Triton Docker container, run the following command inside the container:

```bash
ln -s /opt/tritonserver/bin/tritonserver /opt/tritonserver/bin/tritonserver_new
```

Instead of using:

```bash
tritonserver --model-repository=...
```

You can now use:

```bash
tritonserver_new --model-repository=...
```

The process will now appear as `tritonserver_new` in `nvidia-smi`, making it easier to identify when running multiple instances.
