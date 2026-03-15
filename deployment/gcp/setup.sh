#!/usr/bin/env bash
# One-time GCP project setup for Cloud IDS
set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCS_BUCKET:-${PROJECT}-ids}"
SA_NAME="ids-inference"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "==> Enabling APIs"
gcloud services enable \
  container.googleapis.com \
  pubsub.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  cloudscheduler.googleapis.com \
  --project "$PROJECT"

echo "==> Creating GCS bucket"
gsutil mb -p "$PROJECT" -l "$REGION" "gs://${BUCKET}" 2>/dev/null || echo "(bucket exists)"

echo "==> Creating Pub/Sub topics & subscriptions"
for topic in ids-raw-traffic ids-alerts ids-analyst-queue; do
  gcloud pubsub topics create "$topic" --project "$PROJECT" 2>/dev/null || true
done
gcloud pubsub subscriptions create ids-raw-traffic-sub \
  --topic ids-raw-traffic --project "$PROJECT" \
  --ack-deadline 60 2>/dev/null || true

echo "==> Creating service account"
gcloud iam service-accounts create "$SA_NAME" \
  --display-name "IDS Inference SA" --project "$PROJECT" 2>/dev/null || true

for role in \
  roles/storage.objectAdmin \
  roles/pubsub.publisher \
  roles/pubsub.subscriber \
  roles/bigquery.dataEditor \
  roles/cloudscheduler.jobRunner; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:${SA_EMAIL}" --role "$role" --quiet
done

echo "==> Binding Workload Identity"
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT}.svc.id.goog[ids/ids-sa]" \
  --project "$PROJECT"

echo "==> Creating Cloud Scheduler retrain job"
gcloud scheduler jobs create http ids-retrain \
  --location "$REGION" \
  --schedule "0 2 * * 0" \
  --uri "https://REPLACE_WITH_CLOUD_RUN_URL/retrain" \
  --http-method POST \
  --oidc-service-account-email "$SA_EMAIL" \
  --project "$PROJECT" 2>/dev/null || echo "(job exists)"

echo "==> Setup complete!"
