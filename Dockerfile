FROM tensorflow/serving:latest

# Copy model
COPY ./serving_model /models/

# Copy config monitoring
COPY ./config /model_config

# Set environment variables
ENV MODEL_NAME=andyid-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/v1/models/${MODEL_NAME} || exit 1

# Entrypoint untuk menjalankan TF Serving dengan monitoring
CMD tensorflow_model_server \
    --port=8500 \
    --rest_api_port=${PORT} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    --monitoring_config_file=${MONITORING_CONFIG}