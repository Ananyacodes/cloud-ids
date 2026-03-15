#!/usr/bin/env bash
set -euo pipefail

CONFIG="${IDS_CONFIG:-configs/config.yaml}"
PROJECT=$(python3 -c "from configs.loader import get_config; c=get_config('$CONFIG'); print(c.gcp['project_id'])")
REGION=$(python3  -c "from configs.loader import get_config; c=get_config('$CONFIG'); print(c.gcp['region'])")
CLUSTER="ids-cluster"
IMAGE="gcr.io/${PROJECT}/cloud-ids:latest"

echo "==> Building Docker image: $IMAGE"
docker build -f deployment/docker/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"

echo "==> Ensuring GKE cluster exists"
gcloud container clusters describe "$CLUSTER" \
  --region "$REGION" --project "$PROJECT" 2>/dev/null || \
gcloud container clusters create "$CLUSTER" \
  --region "$REGION" \
  --project "$PROJECT" \
  --num-nodes 3 \
  --machine-type e2-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10 \
  --workload-pool "${PROJECT}.svc.id.goog"

echo "==> Getting credentials"
gcloud container clusters get-credentials "$CLUSTER" \
  --region "$REGION" --project "$PROJECT"

echo "==> Applying K8s manifests"
kubectl apply -f deployment/k8s/

echo "==> Waiting for rollout"
kubectl rollout status deployment/ids-inference -n ids

echo "==> Done. Service endpoint:"
kubectl get svc ids-inference-svc -n ids
