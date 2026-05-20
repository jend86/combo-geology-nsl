# UV Workflow Guide

This project uses `uv` for dependency management. **DO NOT manually create or activate virtual environments.**

## How uv Works

- `uv` automatically creates and manages `.venv` directories
- Each project has its own `uv.lock` file for reproducible installs
- When you run `uv run <command>`, it automatically activates the right environment

## Setup (One Time)

```bash
./setup_uv_env.sh
```

This will:
1. Clean up any manual venvs
2. Run `uv sync` in each project to install dependencies
3. Install voxel-features-mcp in editable mode

## Running Commands

### Feature Hypothesis Task
```bash
cd NSL2-geology-task
uv run python scripts/run_episode.py --config config/config-feature-hypothesis-aiq.toml
```

### Build Harness Images
```bash
cd NSL2-geology-task
uv run python scripts/build_harness_images.py --config config/config-feature-hypothesis-aiq.toml
```

### Quick Test
```bash
cd NSL2-geology-task
./run_quick_test.sh  # This already uses uv run internally
```

## Important Notes

- **Never manually activate** `.venv` - let uv handle it
- **Always use** `uv run` prefix for Python commands
- **Each directory** manages its own environment
- The `voxel_features` symlink connects the projects

## If Something Goes Wrong

```bash
# Clean everything and start fresh
rm -rf */.venv
./setup_uv_env.sh
```
