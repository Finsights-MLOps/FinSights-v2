# ECS Deployment Workflow - Setup Guide

## Overview

This GitHub Actions workflow (`deploy-ecs.yml`) automatically deploys your FinSights application to AWS ECS whenever you push changes to the `main` branch or manually trigger it.

## Features

✅ **Automated Deployment**: Pulls images from ECR and deploys to ECS
✅ **Service Discovery**: Backend automatically registered at `backend.finsights.local:8000`
✅ **Smart Service Updates**: Creates new services if they don't exist, updates if they do
✅ **Network Configuration**: Uses existing security groups and subnets
✅ **Zero-Downtime**: Waits for services to stabilize before completing

## Prerequisites

### 1. AWS Secrets

Add these secrets to your GitHub repository:
- Go to **Settings** → **Secrets and variables** → **Actions**
- Add the following secrets:

| Secret Name | Description |
|------------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret access key |

### 2. ECR Images

Make sure you have pushed your Docker images to ECR:
```bash
# Backend
docker tag backend_finsights:latest 729472661729.dkr.ecr.us-east-1.amazonaws.com/finsights-backend:latest
docker push 729472661729.dkr.ecr.us-east-1.amazonaws.com/finsights-backend:latest

# Frontend  
docker tag frontend_finsights:latest 729472661729.dkr.ecr.us-east-1.amazonaws.com/finsights-frontend:latest
docker push 729472661729.dkr.ecr.us-east-1.amazonaws.com/finsights-frontend:latest
```

### 3. AWS Resources

The workflow uses these existing resources:
- **Cluster**: `finsights-cluster`
- **Security Group**: `sg-0e52e067d61bee6d1`
- **Service Discovery Namespace**: `ns-6dk6zufiwllnypkv` (`finsights.local`)
- **Service Discovery Service**: `srv-wjeqhqnrae5d76ra` (`backend`)
- **Subnets**: 6 subnets across availability zones
- **Task Execution Role**: `ecsTaskExecutionRole`
- **Task Role**: `aws_ecs_task_rules` (with Bedrock + S3 + S3 Vectors permissions)

## How It Works

### Workflow Triggers

1. **Automatic**: Pushes to `main` branch that modify `ModelPipeline/**`
2. **Manual**: Via GitHub Actions UI ("Run workflow" button)

### Deployment Process

#### Backend Deployment (Job 1)
1. ✅ Pull latest image from ECR
2. ✅ Get current task definition
3. ✅ Update with new image tag
4. ✅ Register new task definition revision
5. ✅ Update service (or create if doesn't exist)
6. ✅ Register with Service Discovery
7. ✅ Wait for service to stabilize

#### Frontend Deployment (Job 2)
1. ✅ Pull latest image from ECR
2. ✅ Update task definition with new image
3. ✅ Set `BACKEND_URL=http://backend.finsights.local:8000`
4. ✅ Register new task definition
5. ✅ Update service (or create if doesn't exist)
6. ✅ Wait for service to stabilize
7. ✅ Output public IP address

#### Post-Deployment (Job 3)
1. ✅ Display deployment summary
2. ✅ Show service endpoints

## Usage

### Manual Deployment

1. Go to **Actions** tab in your GitHub repository
2. Select **"Deploy to ECS with Service Discovery"**
3. Click **"Run workflow"**
4. Choose branch (usually `main`)
5. Click **"Run workflow"**

### Monitoring Deployment

1. Watch the workflow progress in the **Actions** tab
2. Each job shows real-time logs
3. Look for the final output showing the public IP

Example output:
```
🌐 Frontend is accessible at: http://44.210.78.32:8501
🔗 Backend is accessible at: http://backend.finsights.local:8000 (internal)
```

## Configuration

### Modifying the Workflow

Edit `.github/workflows/deploy-ecs.yml` to customize:

```yaml
env:
  AWS_REGION: us-east-1              # Change AWS region
  ECS_CLUSTER: finsights-cluster     # Change cluster name
  SECURITY_GROUP: sg-xxx             # Change security group
  SUBNETS: subnet-xxx,subnet-yyy     # Change subnets
```

### Changing Desired Count

To run multiple instances:

```yaml
- name: Create new service
  run: |
    aws ecs create-service \
      --desired-count 2            # Changed from 1 to 2
      ...
```

### Adding Environment Variables

Update the task definition modification step:

```yaml
- name: Update task definition
  run: |
    jq --arg IMAGE "${{ env.ECR_REGISTRY }}/${{ env.BACKEND_ECR_REPOSITORY }}:latest" \
       --arg NEW_VAR "new_value" \
      '(.containerDefinitions[0].image = $IMAGE) | 
       (.containerDefinitions[0].environment += [{"name": "NEW_VAR", "value": $NEW_VAR}])' \
      task-def-clean.json > new-task-def.json
```

## Troubleshooting

### Issue: "Service already exists"
**Solution**: The workflow automatically detects and updates existing services. No action needed.

### Issue: "Task definition not found"
**Solution**: Create initial task definitions manually first, then the workflow will update them.

### Issue: "Security group not found"
**Solution**: Update the `SECURITY_GROUP` environment variable in the workflow file.

### Issue: "Image not found in ECR"
**Solution**: Push images to ECR before running the workflow:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 729472661729.dkr.ecr.us-east-1.amazonaws.com
docker push 729472661729.dkr.ecr.us-east-1.amazonaws.com/finsights-backend:latest
docker push 729472661729.dkr.ecr.us-east-1.amazonaws.com/finsights-frontend:latest
```

### Issue: "Workflow fails with permission errors"
**Solution**: Verify your AWS credentials have these permissions:
- `ecs:*`
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:GetDownloadUrlForLayer`
- `ecr:BatchGetImage`
- `servicediscovery:*`
- `ec2:DescribeNetworkInterfaces`

### Issue: "User is not authorized to perform: s3vectors:QueryVectors"
**Solution**: The ECS Task Role needs S3 Vectors permissions. Add this inline policy to `aws_ecs_task_rules`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3vectors:QueryVectors",
        "s3vectors:GetVectors",
        "s3vectors:GetVectorIndex",
        "s3vectors:DescribeVectorIndex",
        "s3vectors:ListVectorIndexes"
      ],
      "Resource": "arn:aws:s3vectors:*:*:bucket/finrag-embeddings-s3vectors/*"
    }
  ]
}
```

Apply via CLI:
```bash
aws iam put-role-policy \
  --role-name aws_ecs_task_rules \
  --policy-name S3VectorsQueryPolicy \
  --policy-document file://s3vectors-policy.json
```

Then restart the backend service:
```bash
aws ecs update-service \
  --cluster finsights-cluster-new \
  --service finsights-backend \
  --force-new-deployment
```

## Best Practices

1. **Always test locally** before pushing to main
2. **Monitor CloudWatch logs** after deployment
3. **Use semantic versioning** for image tags (not just `:latest`)
4. **Set up Slack/email notifications** for workflow failures
5. **Review logs** in the Actions tab after each deployment

## Next Steps

- [ ] Set up staging environment
- [ ] Add automated testing before deployment
- [ ] Configure CloudWatch alarms
- [ ] Set up auto-scaling policies
- [ ] Add deployment notifications
