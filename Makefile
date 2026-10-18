IMAGE=heart-disease-api:1.0.0

data:
	python -m src.data
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest -v --cov=src
lint:
	ruff check .
docker-build:
	docker build -t $(IMAGE) .
docker-run:
	docker run -d --name heart-api -p 8000:8000 $(IMAGE)
helm-deploy:
	helm upgrade --install heart deploy/helm/heart-disease-api -n heart-disease --create-namespace --wait
