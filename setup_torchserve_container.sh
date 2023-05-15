torch-model-archiver --model-name pi_vit --version 0.1 --serialized-file models/pi_vit.pt --handler handler:apply_fn --requirements-file mar_requirements.txt

torchserve --start \
    --ncs \
    --model-store model_store \
    --models pi=pi_vit.mar
           

./build_image.sh -t torchserve:1.0
docker run --rm -it --platform linux/amd64 \
    -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
    --mount type=bind,source=$(pwd)/model_store,target=/tmp/models \
    pytorch/torchserve:latest \
    torchserve --model-store /tmp/models --models pi=pi_vit.mar
