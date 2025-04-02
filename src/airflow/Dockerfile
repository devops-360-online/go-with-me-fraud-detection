FROM apache/airflow:2.10.3

USER root

# Create directory with correct permissions
RUN mkdir -p /app/models && \
    chown -R airflow:root /app/models && \
    chmod -R 775 /app/models

WORKDIR /app

VOLUME /app/models

USER airflow

COPY requirements.txt /tmp/requirements.txt
# Add verbose output and break into smaller chunks if needed
RUN pip install -v --no-cache-dir -r /tmp/requirements.txt
