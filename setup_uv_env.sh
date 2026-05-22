#!/bin/bash
# Setup uv environment properly

echo "Setting up uv environment..."
echo "============================"

# Step 1: Clean up old venvs
echo ""
echo "Step 1: Cleaning up old virtual environments..."
rm -rf NSL2-geology-task/.venv
rm -rf voxel-features-mcp/.venv
rm -rf .venv

# Step 2: Let uv handle NSL2-geology-task
echo ""
echo "Step 2: Setting up NSL2-geology-task with uv..."
cd NSL2-geology-task
uv sync
cd ..

# Step 3: Setup voxel-features-mcp with uv
echo ""
echo "Step 3: Setting up voxel-features-mcp with uv..."
cd voxel-features-mcp
uv sync
# Install in editable mode for development
uv pip install -e .
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "uv will automatically manage environments when you run:"
echo "  cd NSL2-geology-task"
echo "  uv run python scripts/run_episode.py --config ..."
echo ""
echo "No manual activation needed - uv handles it!"
