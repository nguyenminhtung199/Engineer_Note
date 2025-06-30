FROM registry.icomm.vn/kh-cahanam/hawkice.modules.cache_frame:20250523-02

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# RUN apt update && apt install git -y

RUN pip install isodate

WORKDIR /app

COPY . /app