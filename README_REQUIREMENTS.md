# Requirements Installation Guide

This project uses `uv` for dependency management. All scripts expect `uv run` commands.

## Setup with uv (Required)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the setup script from the root directory
./setup_uv_env.sh
```

The setup script will:
1. Clean up any manual virtual environments
2. Run `uv sync` in each project directory
3. Install voxel-features-mcp in editable mode

## Using the Project

All commands must be run with `uv run`:

```bash
# Example: Run an episode
cd NSL2-geology-task
uv run python scripts/run_episode.py --config config/config-feature-hypothesis-aiq.toml

# Example: Build harness images
uv run python scripts/build_harness_images.py --config config/config-feature-hypothesis-aiq.toml
```

**Important**: Never manually activate `.venv` - uv handles this automatically.

## Key Dependencies

### NSL2-geology-task
- **ML/Training**: torch, transformers, unsloth, peft
- **API**: fastapi, mcp, openai, anthropic
- **Infrastructure**: docker, loguru, rich

### voxel-features-mcp
- **Scientific**: numpy, scipy, sklearn (for ridge regression)
- **Data**: polars, duckdb, pyarrow
- **Geospatial**: geopandas, shapely, pyproj

## Notes

- Python 3.12+ required for NSL2-geology-task
- Python 3.11+ required for voxel-features-mcp
- The `unsloth` dependency requires CUDA-capable GPU
- Set environment variables in `.env` (see `.env.example`)
