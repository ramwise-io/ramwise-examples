# syntax=docker/dockerfile:1.10

ARG CUDA_IMAGE="nvidia/cuda:13.1.1-runtime-ubuntu24.04@sha256:7ced05425d3bdea47c018551ee001f28081dd409efe1283872cd968ad3c1e957"
FROM ${CUDA_IMAGE}

ARG SPARK_VERSION="3.5.8"
ARG SPARK_SHA512="5d5d2e6e111182ee1210d4a46a17ae27107c7ccc70433da0ded3d8197e027f949b00445bb5678ed0c1d5c59f12993d9b9ed12e43686d5aad89a9ec570e93a083"
ARG RAPIDS_VERSION="26.06.1"
ARG RAPIDS_SHA1="8a8264eb5ca1dbcd3d21c8e1032f983803941e17"
ARG USER_UID="1001"
ARG USER_GID="1001"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates curl git openjdk-17-jre-headless python3 python3-pip tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/spark.tgz \
      "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
    && echo "${SPARK_SHA512}  /tmp/spark.tgz" | sha512sum --check --strict \
    && tar -xzf /tmp/spark.tgz -C /opt \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop3" /opt/spark \
    && rm /tmp/spark.tgz

RUN curl -fsSLo /opt/spark/jars/rapids-4-spark_2.12-${RAPIDS_VERSION}-cuda13.jar \
      "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${RAPIDS_VERSION}/rapids-4-spark_2.12-${RAPIDS_VERSION}-cuda13.jar" \
    && echo "${RAPIDS_SHA1}  /opt/spark/jars/rapids-4-spark_2.12-${RAPIDS_VERSION}-cuda13.jar" | sha1sum --check --strict

COPY requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

RUN groupadd --gid "${USER_GID}" codexlab \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash codexlab \
    && mkdir -p /workspace/gpu-lab /data/cache/spark \
    && chown -R codexlab:codexlab /workspace /data/cache

ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64" \
    PATH="/opt/spark/bin:/opt/spark/sbin:${PATH}" \
    PYSPARK_PYTHON="/usr/bin/python3" \
    PYTHONPATH="/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip" \
    SPARK_HOME="/opt/spark" \
    PYTHONUNBUFFERED="1"

USER codexlab
WORKDIR /workspace/gpu-lab
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "pytest", "tests/test_spark_rapids_fallback.py", "-q"]
