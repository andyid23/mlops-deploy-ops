FROM tensorflow/serving:latest

# 1. Copy model dan config ke dalam container
COPY ./serving_model /models/
COPY ./config /model_config

# 2. Set Environment Variables
ENV MODEL_NAME=andyid-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

# 3. Custom Entrypoint untuk menjalankan TF Serving
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} "$@"' > /usr/bin/tf_serving_entrypoint.sh && chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]