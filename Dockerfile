# Reproducible container packaging Rabi Crop Doctor (Python ML/ANN CLI + offline models + Node parity runtime)
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Rabi Crop Doctor" \
      org.opencontainers.image.description="Free offline ML & ANN crop disease classifier and field incidence tool for Indian Rabi farmers" \
      org.opencontainers.image.source="https://github.com/lunganlungpg/rabi-crop-doctor" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install Python scientific stack cleanly with no build cache bloat
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project modules, offline model weights, bilingual data, and sample leaves
COPY pyproject.toml README.md LICENSE ./
COPY python ./python
COPY data ./data
COPY models ./models
COPY dataset_samples ./dataset_samples
COPY tests ./tests
COPY web ./web

# Install package in editable/standard mode without needing external network tools
RUN pip install --no-cache-dir --no-deps .

# Default entrypoint runs CLI image classification or lets users invoke Python scripts
ENTRYPOINT ["rabi-doctor"]
CMD ["dataset_samples/wheat_yellow_rust_sample_1.png", "--crop", "wheat"]
