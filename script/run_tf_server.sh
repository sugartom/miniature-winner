sudo docker run --runtime=nvidia -p 8500:8500 \
	--mount type=bind,source=/home/oscar/filesys/tf_servable/,target=/models/tf_servable \
	-t --entrypoint=tensorflow_model_server tensorflow/serving:latest-gpu --port=8500 \
	--platform_config_file=/models/tf_servable/platform.conf --model_config_file=/models/tf_servable/models.conf