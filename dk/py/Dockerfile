# dkr -p=1234:1234 --name summarizer --rm summarizer:2021-03-04

FROM python:3.8.0-slim as builder
RUN apt-get update \
  && apt-get install gcc=4:8.3.0-1 git=1:2.20.1-2+deb10u3 -y \
&& apt-get clean
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --user -r requirements.txt
COPY . /app

FROM python:3.8.0-slim as app
COPY --from=builder /root/.local /root/.local
# COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/app /app/app
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
EXPOSE 1234
ENTRYPOINT uvicorn app.main:app --reload --host 0.0.0.0 --port 1234
