AWS deploy quickstart
=====================

This project includes helpers to make deployment to AWS easier.

Prerequisites
- An AWS account and IAM role with permissions to push to ECR and (optionally) update ECS.
- AWS CLI configured locally for manual steps, or GitHub Secrets configured for Actions.

Quick manual steps (ECR + run on ECS Fargate):

1. Create ECR repository (example):

```bash
aws ecr create-repository --repository-name my-parkinsons-api --region us-east-1
```

2. Build and push locally (example):

```bash
docker build -t <account>.dkr.ecr.<region>.amazonaws.com/my-parkinsons-api:latest .
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/my-parkinsons-api:latest
```

3. Deploy to ECS Fargate: create a Task Definition and Service (use Console or AWS CLI). Ensure the task has:
- ENV `DATABASE_URL` set to your RDS connection string
- IAM role allowing access to S3 (if `MODEL_S3_BUCKET` is used)
- `MODEL_S3_BUCKET` and optional `MODEL_S3_PREFIX` env vars if you want the container to download models at startup

CI/CD
- See `.github/workflows/deploy-to-ecr.yml` which builds the image and pushes it to ECR when you push to `main`.

Notes
- By default the container will try to download model artifacts from S3 when `MODEL_S3_BUCKET` is set. If you prefer baking models into the image, remove `models/` from `.dockerignore`.
- Use AWS Secrets Manager to provide `DATABASE_URL` and other secrets to ECS tasks.
