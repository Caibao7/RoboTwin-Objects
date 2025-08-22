# RoboTwin-Objects

A comprehensive 3D object dataset and annotation pipeline for robotics simulations and vision-language-action (VLA) models. This project provides semantic and physical annotations for thousands of 3D objects using Large Language Model (LLM)-based automated annotation.

## 🏗️ Project Structure

```
RoboTwin-Objects/
├── objects/                     # 3D object assets
│   ├── objects_glb/            # GLB format 3D models with collision meshes (.coacd.ply)
│   ├── objects_glb_pictures/   # Preview images for GLB objects  
│   ├── objects_urdf/           # URDF format models with physics properties
│   ├── objects_xml/            # XML format scene descriptions
│   ├── prompt_test/            # LLM prompt testing and evaluation
│   └── textures/               # Material textures for rendering
├── prompt/                     # LLM prompts for annotation tasks
├── scripts/                    # Automation and processing scripts
│   ├── call_llm/              # LLM API interaction scripts
│   ├── upload_hf/             # Hugging Face dataset upload utilities
│   └── utils/                 # Data processing utilities
└── *.json                     # Dataset metadata and annotations
```

## 📊 Core Data Files

### Primary Object Databases
- **`robotwin_objects.json`** - Base object catalog with IDs, names, categories, and tags
- **`robotwin_info_generated_by_llm.json`** - Complete LLM-generated annotations including:
  - Physical dimensions (`real_size` in meters)
  - Material properties (`density`, `static_friction`, `dynamic_friction`, `restitution`)  
  - Semantic descriptions (`Basic_description`, `Functional_description`)
- **`robotwin_real_sizes_meter.json`** - Real-world size measurements in meters
- **`robotwin_uuid_map.csv`** - UUID to object name mappings

### Generated Datasets
- **`filtered_robotwin_*.json`** - Filtered subsets with image associations
- **`size_valid_refine_scale_results.json`** - Scale validation and refinement results
- **`updated_robotwin_info.json`** - Post-processed annotations

## 🤖 LLM Annotation Pipeline

The project uses multiple LLM-based annotation stages to generate comprehensive object metadata:

### 1. Visual Analysis (`generate_all_captions.txt`)
Analyzes 6-view composite images to extract:
- Object name and category classification
- Real-world size estimation (bounding box in meters)
- Material identification and physical properties
- Functional descriptions and use cases

**Input**: Multi-view object images + object names  
**Output**: Complete semantic and physical annotations

### 2. Size Estimation (`generate_longest_length.txt`) 
Estimates the longest dimension of objects for scale calculation:
- Conservative real-world size estimation
- Handles perspective distortion in images
- Outputs measurements in meters

**Input**: Object images  
**Output**: `{"longest_m": <float>}` 

### 3. Size Validation (`validate_real_size.txt`)
Validates computed object sizes against real-world expectations:
- Checks if dimensions are reasonable for object category
- Identifies objects that are too large/small
- Suggests corrective scale factors

**Input**: Object dimensions, scales, computed sizes  
**Output**: Validation results with suggested corrections

## ⚙️ Processing Scripts

### LLM API Integration (`scripts/call_llm/`)
- **`robotwin_call_gpt_image.py`** - Multi-threaded GPT API calls with image inputs
- **`robotwin_call_gpt_image_new.py`** - Updated version with refined prompts
- **`robotwin_generate_real_size.py`** - Dedicated size estimation pipeline
- **`validate_real_sizes.py`** - Automated size validation

### Data Processing (`scripts/utils/`)
- **`compute_scale_from_longest.py`** - Calculate scaling factors from longest dimensions
- **`calculate_real_sizes.py`** - Convert model units to real-world meters
- **`convert_real_size_to_m.py`** - Unit conversion utilities
- **`proceed_output_scale.py`** - Apply scale corrections to object data

### Dataset Management (`scripts/upload_hf/`)
- **`hf_upload_info.py`** - Upload datasets to Hugging Face Hub

## 🔬 Object Categories

The dataset covers diverse object categories including:
- **Household items**: mugs, plates, utensils, containers
- **Food items**: fruits, packaged foods, beverages  
- **Furniture**: tables, chairs, cabinets, shelves
- **Electronics**: appliances, devices
- **Tools and equipment**: kitchen tools, office supplies
- **Personal items**: shoes, books, accessories

## 📏 Physical Properties

Each object includes detailed physical annotations:

### Dimensions
- **`real_size`**: 3D bounding box [width, depth, height] in meters
- **`scale`**: Scaling factors to convert from model units to meters

### Material Properties  
- **`density`**: Mass density in g/cm³
- **`static_friction`**: Static friction coefficient (μₛ) on wood surfaces
- **`dynamic_friction`**: Dynamic friction coefficient (μₖ) on wood surfaces  
- **`restitution`**: Coefficient of restitution (bounce) on wood surfaces

### Semantic Descriptions
- **`Basic_description`**: Concise physical description
- **`Functional_description`**: List of primary use cases and functions
- **`category`**: Object classification (mug, food, furniture, etc.)
- **`tags`**: "StructuralEntities" (large/fixed) or "DynamicEntities" (movable)

## 🧪 Prompt Testing

The `objects/prompt_test/` directory contains comparative testing of different LLM prompts:

### Available Prompts
- **`claude_prompt.txt`** - Anthropic Claude optimized prompts
- **`gpt_prompt.txt`** - OpenAI GPT optimized prompts  
- **`doubao_prompt.txt`** - ByteDance Doubao prompts
- **`deepseek_prompt.txt`** - DeepSeek model prompts
- **`grok_prompt.txt`** - xAI Grok prompts

### Evaluation Results
- **`results.md`** - Comparative performance analysis
- **`robotwin_scale_generated_by_*_*.json`** - Results from different model/prompt combinations
- **`sort_json.py`** - Utility for organizing test results

## 🚀 Usage Examples

### Loading Object Data
```python
import json

# Load complete object database
with open('robotwin_info_generated_by_llm.json', 'r') as f:
    objects = json.load(f)

# Get object properties
obj = objects['00aff23a-2075-44d5-a4eb-da6d5998a409']
print(f"Object: {obj['object_name']}")
print(f"Size: {obj['real_size']} meters")
print(f"Density: {obj['density']} g/cm³")
print(f"Functions: {obj['Functional_description']}")
```

### Size Validation
```python
# Run size validation
python scripts/call_llm/validate_real_sizes.py \
    --input robotwin_info_generated_by_llm.json \
    --output size_validation_results.json
```

### Scale Computation
```python
# Compute scales from longest dimensions
python scripts/utils/compute_scale_from_longest.py \
    -l robotwin_longest_m_by_gpt41.json \
    -d filtered_robotwin_dim_img.json \
    -o robotwin_scale_from_longest.json
```

## 📋 Data Format Examples

### Object Entry Structure
```json
{
  "00aff23a-2075-44d5-a4eb-da6d5998a409": {
    "object_name": "boxed_playing_cards",
    "category": "cards", 
    "real_size": [0.065, 0.022, 0.09],
    "density": 0.65,
    "static_friction": 0.45,
    "dynamic_friction": 0.34,
    "restitution": 0.3,
    "Basic_description": "A rectangular box containing a standard deck of playing cards",
    "Functional_description": [
      "used for card games",
      "used for magic tricks", 
      "used for educational purposes"
    ]
  }
}
```

### Size Validation Result
```json
{
  "is_proper": true,
  "assessment": null,
  "typical_size_range": "0.06–0.10 m length, 0.02–0.03 m thickness", 
  "suggested_scale": null
}
```

## 🔧 Configuration

### Environment Variables
- **`OPENAI_MODEL`** - Model selection (gpt-4o, gpt-4o-mini, etc.)
- **`WORKERS`** - Concurrent processing threads (default: 6)
- **API Keys** - Configure OpenAI, Anthropic, or other LLM provider credentials

### Processing Parameters
- **`MAX_RETRIES`** - API retry attempts (default: 3)
- **Scale precision** - 4 decimal places with truncation
- **Image formats** - Support for PNG, JPG composite views

## 🎯 Applications

This dataset is designed for:
- **Robotics simulation** with accurate physics properties
- **Vision-Language-Action (VLA) model training**
- **3D scene understanding** and object recognition
- **Physics-based AI** and manipulation planning
- **Synthetic data generation** for robotic learning

## 📚 Citation

If you use this dataset in your research, please cite:
```bibtex
@dataset{robotwin_objects,
  title={RoboTwin-Objects: A Large-Scale Dataset of 3D Objects with LLM-Generated Annotations},
  year={2024},
  note={Comprehensive 3D object dataset with semantic and physical properties}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our guidelines for:
- Adding new object categories
- Improving annotation prompts  
- Extending physical property models
- Enhancing validation procedures

## 📄 License

This dataset is released under [LICENSE] for research and educational purposes.