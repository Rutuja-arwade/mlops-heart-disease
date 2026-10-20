# heart disease mlops
MLOps assignment 1 - heart disease prediction on UCI cleveland data. report is in report.pdf (10 pages) , screenshots folder has the screenshots

# Submitted by:
- Name: ARWADE RUTUJA NITIN
- ID: 2025ae05375

drive links for assignment submissions:
- video - https://drive.google.com/file/d/1BIK8L5S573MXqQtXSKFjAlnojAxp2jKt/view?usp=sharing
  ( same video in this repo - video/pipeline_walkthrough.mp4 )
- report - https://drive.google.com/file/d/1II87jhtBAMi7yg58JTIyyrFL6TG-nNia/view?usp=sharing

setup (python 3.11)
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
run
```
python -m src.data
python -m src.eda
python -m src.train
pytest
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

docker:
```
docker build -t heart-disease-api:1.0.0 .
docker run -d -p 8000:8000 heart-disease-api:1.0.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
swagger at localhost:8000/docs. monitoring stack - `docker compose up -d` (grafana localhost:3000 admin/heart-admin, prometheus :9090)

## k8s (minikube)
```
minikube start --driver=docker
minikube addons enable ingress
minikube addons enable metrics-server
minikube image load heart-disease-api:1.0.0
helm upgrade --install heart deploy/helm/heart-disease-api -n heart-disease --create-namespace --wait
kubectl apply -k monitoring
minikube tunnel
```
then api is at http://127.0.0.1:8080 (loadbalancer). ingress host is heart-api.local
or without helm - `kubectl apply -k deploy/k8s`

ci/cd is in .github/workflows/ci.yml (lint, test, train, docker, deploy on kind, push to ghcr)
