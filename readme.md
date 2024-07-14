# Using OpenClip with Triton

Using Docker compose to start up Triton with the OpenClip model, to encode text in to vectors

* Download the model from: https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin
* Place the file at model_cache/laion--CLIP-ViT-L-14-laion2B-s32B-b82K/model.bin"



### Install Nvidia toolkit

Follow the steps here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

###

```
docker-compose up --build
```


### Test health

```
curl -v http://localhost:8000/v2/health/ready
```

### Test encoding a string

```
curl -X POST "http://localhost:8000/v2/models/openclip_text_encoder/infer" -d '{
  "inputs": [
    {
      "name": "text_encoder_input",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["hello world"]
    }
  ]
}'
```

### Stats

```
curl -v http://localhost:8000/v2/models/stats
```