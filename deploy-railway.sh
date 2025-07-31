#!/bin/bash

# Railway Deployment Script
echo "🚂 Deploying to Railway..."

# Install Railway CLI if not installed
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login to Railway (if not already logged in)
echo "Logging into Railway..."
railway login

# Create new project or link existing
echo "Setting up Railway project..."
railway init

# Set environment variables
echo "Setting environment variables..."
railway variables set OPENAI_API_KEY="your-openai-key"
railway variables set BEARER_TOKEN="your-bearer-token"
railway variables set PYTHON_VERSION="3.11"

# Deploy
echo "Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at the Railway-provided URL"