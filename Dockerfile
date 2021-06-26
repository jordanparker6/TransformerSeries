FROM nvcr.io/nvidia/pytorch:21.06-py3

# Instal basic utilities
RUN apt-get update && \
  apt-get install -y --no-install-recommends git wget unzip bzip2 sudo make build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/

WORKDIR /app
COPY . .


CMD ["python3", "src/main.py"]