FROM tensorflow/serving:latest

# Install Python 3 + pip (TF Serving image is Debian-based)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Flask and requests for the wrapper script
RUN pip3 install --no-cache-dir flask==2.3.3 requests==2.31.0 werkzeug==2.3.7

# Copy model and config
COPY ./serving_model /models/
COPY ./config /model_config

# Copy the Flask wrapper
COPY app.py /app.py

# Set environment variables
ENV MODEL_NAME=andyid-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
# PORT=8080 matches Railway's service domain configuration
ENV PORT=8080

# Start the Flask wrapper, which launches TF Serving internally and
# proxies requests from port 8080 → TF Serving REST on localhost:8501
CMD ["python3", "/app.py"]