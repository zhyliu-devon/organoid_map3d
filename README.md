# organoid_map3d

## Overview
**organoid_map3d** is a Python package for **3D mapping of cardiac organoids** with **SciPy, VTK, and Matplotlib** for data processing, visualization, and analysis. This package is designed for reproducible research and scientific computing.

## Features
- Efficient **3D mapping** with Radial Basis Functions and visualization using VTK.
- Functions for **electrophysiology data preprocessing**, **calcium data preprocessing** and **Conduction Velocity analysis**.
- Support for **Intan RHS file formats** for neural data integration.

---


## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/organoid_map3d.git
cd organoid_map3d
```

### **2. Install Software Dependencies**
```bash
pip install -r requirements.txt
```

## Handling Intan RHS Data
This package depends on **`load_intan_rhs_format.py`** from Intan Technologies.

### **Option 1: Manually Download It**
1. Download `load_intan_rhs_format.py` from [Intan Technologies](https://intantech.com).
2. Place it inside the `organoid_map3d/electrophysiology_mapping` directory.



## Citation
If you use `organoid_map3d` in your research, please cite:

**[Your Paper Title]**  
**[Your Authors]**  
**[Journal Name, Year]**  
**[DOI/ArXiv Link]**  


---

## License
This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

## Acknowledgments
- Intan Technologies for `load_intan_rhs_format.py`
- VTK, SciPy, Seaborn for scientific computing tools
- Research community for open-source contributions


