# Kazakhstan Geological Analysis System

Implementation of the multi-agent geological AI system for Kazakhstan Teniz Basin dataset.

## Overview

This implementation extends the original Australian Coe Fairbairn system to analyze Kazakhstan geological data covering 66°30'E-71°30'E, 49°30'N-52°30'N with regional-scale voxel analysis.

## Key Features

- **Regional Scale**: 200×200×8 voxel grid covering ~116,160 km² 
- **Voxel Resolution**: ~1.75km × 1.66km × 10m per voxel
- **Kazakhstan Data**: USGS Teniz Basin assessment + Russian geological surveys
- **Dual Format**: English (USGS) and Russian (Smolianova) geological data

## Architecture

### Regional Organization
```
NSL2-geology-task/
├── data/
│   ├── australia/feature-hypothesis/     # Original Coe Fairbairn system
│   └── kazakhstan/feature-hypothesis/    # New Kazakhstan system
└── tasks/
    ├── feature_hypothesis.py           # Australia task
    └── feature_hypothesis_kazakhstan.py # Kazakhstan task
```

### Grid Specifications

**Kazakhstan Grid (`_KAZAKHSTAN_TENIZ_GRID`):**
- Origin: 66.5°E, 49.5°N, 0m
- Maximum: 71.5°E, 52.5°N, 80m  
- Shape: 200×200×8 voxels
- Total voxels: 320,000

**Comparison with Australia:**
| Region     | Longitude Range | Latitude Range | Area (km²) | Voxel Resolution |
|------------|-----------------|----------------|------------|------------------|
| Australia  | 117.832°-117.973° | -27.441°--27.300° | ~460 | ~70m × 79m |
| Kazakhstan | 66.5°-71.5°E | 49.5°-52.5°N | ~116,160 | ~1.75km × 1.66km |

## Dataset Structure

### Kazakhstan Data Sources
- **USGS Data**: English-language Teniz Basin technical reports
- **Russian Surveys**: Comprehensive geological surveys (Smolianova 1984)
- **Spatial Data**: GeoJSON copper prospects, geological structures, basin boundaries
- **Text Data**: 300+ geological text chunks, drill hole logs

### Key Files
- `copper_prospects.geojson`: 113 copper prospects with coordinates, tonnage, grades
- `anticlines_synclines.geojson`: 33 geological fold structures
- `assessment_tract.geojson`: Teniz Basin boundary and tract data (49,714 km²)
- `36572_Smolianova_1984/`: 579 Russian geological survey files

## Usage

### Running Kazakhstan Analysis
```bash
cd NSL2-geology-task

# Run complete Kazakhstan workflow test (recommended)
./test_kazakhstan_workflow.sh

# Or run individual components:

# Start Kazakhstan containers
docker-compose -f docker/feature-hypothesis-kazakhstan-compose/docker-compose.yml up

# Test grid mapping only
python3 test_kazakhstan_grid.py

# Run episode manually
uv run python scripts/run_episode.py config/config-feature-hypothesis-kazakhstan.toml
```

### Environment Variables
```bash
export KAZKHSTAN_DATA_DIR=/path/to/Kazakhstan_data
export FEATURE_STORE_DIR=/path/to/data/kazakhstan/feature-hypothesis/store  
export FEATURE_KG_DIR=/path/to/data/kazakhstan/feature-hypothesis/knowledge
```

## Workflow Adaptations

### Kazakhstan-Specific Changes
1. **Dataset Context**: Updated system prompt for Kazakhstan geological context
2. **Grid Bounds**: Coordinates validated for Kazakhstan region (66.5°-71.5°E, 49.5°-52.5°N)
3. **Spatial Translation**: Adapted for regional-scale features (basins, mineral belts, regional trends)
4. **Data Enhancement**: Support for multilingual data (English/Russian)

### Prompt Adaptations
- Regional geological intelligence focus
- Basin-scale structural geology
- Sediment-hosted copper deposit models
- Integration of USGS + Russian survey data

## Grid Validation Results

✅ **Coordinate Mapping Test Results:**
- Total test coordinates: 11
- Successfully mapped: 7 in-bounds coordinates  
- Coverage area: 116,160 km²
- Voxel resolution: 1.75km × 1.66km × 10m

✅ **Sample Prospect Mapping:**
- Sovetskoe prospect (68.046°, 51.997°) → Voxel (61, 166, 1)
- Kirei prospect (68.029°, 51.963°) → Voxel (61, 164, 1)  
- Teniz Basin prospect (68.358°, 51.995°) → Voxel (74, 166, 1)

## Implementation Status

- ✅ Regional directory restructure completed
- ✅ Kazakhstan task implementation (`feature_hypothesis_kazakhstan.py`)
- ✅ Australia task paths updated to regional structure
- ✅ Data/knowledge directory setup
- ✅ Grid coordinate mapping validated  
- ✅ Docker configuration created

## Technical Notes

### Grid Performance
- Same voxel count as Australia (320K voxels) maintains computational efficiency
- Larger geographic coverage per voxel suits regional geological analysis
- Compatible with existing BIC scoring and spatial operations infrastructure

### Data Integration
- Kazakhstan dataset remains in original location (`Kazkhstan_data/`)
- Analysis sandbox mounts Kazakhstan data at `/workspace/input`
- Separate store/knowledge tracking prevents cross-contamination with Australia experiments

### Scalability
- Clean regional separation enables future geographic expansion
- Shared codebase for core geological analysis logic
- Independent experiment tracking per region

## Future Enhancements

1. **Multi-Regional Comparison**: Cross-region geological pattern analysis
2. **Scale-Adaptive Processing**: Dynamic voxel resolution based on data density
3. **Enhanced Multilingual Support**: Automated Russian-English geological term translation
4. **Regional Ensemble Modeling**: Combined insights from multiple geological datasets
