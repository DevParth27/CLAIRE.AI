#!/bin/bash

# Railway Deployment Script
echo "Deploying to Railway..."

# Install Railway CLI if not installed
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login to Railway
echo "Please login to Railway..."
railway login

# Initialize project
railway init

# Set environment variables
echo "Setting environment variables..."
railway variables set OPENAI_API_KEY="your_openai_api_key_here"
railway variables set BEARER_TOKEN="your_secure_bearer_token_here"
railway variables set RAILWAY_ENVIRONMENT="production"
railway variables set PYTHON_VERSION="3.11"

# Deploy
echo "Deploying application..."
railway up

echo "Deployment complete! Check Railway dashboard for your app URL."