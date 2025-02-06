# organoid_map3d

## Overview
**organoid_map3d** is a Python package for **3D mapping of organoids**, utilizing **VTK, PIL, SciPy, and Seaborn** for data processing, visualization, and analysis. This package is designed for reproducible research and scientific computing.

## Features
- Efficient **3D mapping** and visualization using VTK.
- Functions for **data preprocessing** and **contour analysis**.
- Support for **Intan RHS file formats** for neural data integration.
- Modular design for easy expansion and reproducibility.

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/organoid_map3d.git
cd organoid_map3d
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### **Basic Import**
```python
import organoid_map3d as om
```

### **Example: Data Processing**
```python

```

### **Example: 3D Visualization with VTK**
```python

```

---

## Modules
| Module | Description |
|--------|------------|
| `data.py` | Handles data loading and transformations |
| `preprocessing.py` | Data cleaning and preparation utilities |
| `map.py` | Spatial mapping functions |
| `spike.py` | Spike detection and neural data analysis |
| `visualization/` | VTK-based 3D rendering and plotting |
| `io/` | File handling (including Intan RHS format support) |

---

## Handling Intan RHS Data
This package depends on **`load_intan_rhs_format.py`** from Intan Technologies.

### **Option 1: Manually Download It**
1. Download `load_intan_rhs_format.py` from [Intan Technologies](https://intantech.com).
2. Place it inside the `organoid_map3d/io/` directory.

### **Option 2: Auto-Download (Optional)**
Run this script to fetch it automatically:
```python
from organoid_map3d.io.download_intan import download_intan_script
download_intan_script()
```

---

## Development Setup
### **Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### **Run Tests**
```bash
pytest tests/
```

---

## Citation
If you use `organoid_map3d` in your research, please cite:

**[Your Paper Title]**  
**[Your Authors]**  
**[Journal Name, Year]**  
**[DOI/ArXiv Link]**  

Alternatively, use the citation metadata provided in [`CITATION.cff`](CITATION.cff).

---

## License
This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

## Acknowledgments
- Intan Technologies for `load_intan_rhs_format.py`
- VTK, SciPy, Seaborn for scientific computing tools
- Research community for open-source contributions

