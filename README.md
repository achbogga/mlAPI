# Pest and Disease Detection System API

This repository explains how to create PDDS Microservice API
using FastAPI and Nvidia Triton Inference Server

## Getting started

```bash
conda create --clone base --name pdds_api
conda activate pdds_api
python3 -m pip install -r requirements.txt
python3 main.py | tee pdds_api.log
```

## to run the service in the background

```bash
conda activate pdds_api
python3 main.py | tee pdds_api.log &
```

## API Usage and Documentation

- Type the following link in your web browser
- [http://localhost:8080/docs](http://localhost:8080/docs)
