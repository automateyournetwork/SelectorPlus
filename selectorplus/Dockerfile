FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
  && apt-get update && apt-get install -y --fix-missing python3 python3-pip

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
  && pip install --break-system-packages -U --quiet langchain_experimental langchain langchain-community langchain_google_genai

RUN echo "==? Install langgraph and required components" \
  && pip install --break-system-packages --upgrade "langgraph-cli[inmem]" \
  && pip install --break-system-packages langsmith \
  && pip install --break-system-packages fastapi

RUN echo "==> Adding dotenv ..." \
  && pip install --break-system-packages python-dotenv

COPY /selectorplus /selectorplus/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]