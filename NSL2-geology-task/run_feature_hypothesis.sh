#!/bin/bash
# Run the feature hypothesis task with OpenRouter Deepseek

echo "Feature Hypothesis Task Runner"
echo "=============================="

# Check if .env has been updated
if grep -q "your-openrouter-api-key-here" .env; then
    echo "ERROR: Please update .env with your actual OpenRouter API key"
    echo "Edit .env and replace 'your-openrouter-api-key-here' with your key"
    exit 1
fi

# Step 1: Build Docker containers
echo ""
echo "Step 1: Building Docker containers..."
echo "-------------------------------------"
docker compose -f docker/feature-hypothesis-australia-compose/docker-compose.yml build

# Step 2: Build harness images (if needed)
echo ""
echo "Step 2: Building harness images..."
echo "----------------------------------"
uv run python scripts/build_harness_images.py --config config/config-feature-hypothesis-australia.toml

# Step 3: Run the task
echo ""
echo "Step 3: Running feature hypothesis task..."
echo "-----------------------------------------"
uv run python scripts/run_episode.py --config config/config-feature-hypothesis-australia.toml

echo ""
echo "Task completed!"
