# **organoid_map3d**

## **Overview**
**organoid_map3d** is a set of Python example code for **3D mapping and analysis of cardiac organoids**, leveraging **SciPy, VTK, and Matplotlib** for data processing, visualization, and computational analysis. It is designed for **reproducible research** in bioengineering and scientific computing.

## **Features**
- **3D spatial mapping** using **Radial Basis Functions (RBFs)** and visualization via **VTK**.
- **Electrophysiology** and **calcium imaging** data preprocessing functions.
- **Conduction velocity analysis** for cardiac signal propagation.
- **Integration** with **Intan RHS file formats**.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/zhyliu-devon/organoid_map3d.git
cd organoid_map3d
```

### **2. Dependencies**
The following software versions have been tested with this package:

| Package              | Version   |
|----------------------|-----------|
| `load_intan_rhs_format` | September 2023 |
| `VTK`               | 9.3.0     |
| `Pillow`            | 10.3.0    |
| `SciPy`             | 1.10.0    |
| `Seaborn`           | 0.13.2    |
| `NumPy`             | 1.23.5    |
| `Matplotlib`        | 3.8.4     |

No specialized hardware is required.

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Handling Intan RHS Data**
This package requires **`load_intan_rhs_format.py`** from **Intan Technologies**.

1. Download `load_intan_rhs_format` from [Intan Technologies](https://intantech.com/files/load_intan_rhs_format.zip).
2. Place the folder inside your working directory.

Typical installation time: **~5 minutes**.

---

## **Usage**
### **Demo**
- The package includes **example scripts** showcasing all data analysis steps.
- Example data (used in our paper) is provided to reproduce key figures.
- Due to space limitations, not all raw data is included.
- The script in the folder can be directly executed after installation.
---


---

## **License**
This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

## **Acknowledgments**
- **Intan Technologies** for `load_intan_rhs_format.py`.
- **VTK, SciPy, Seaborn**, and other open-source tools enabling scientific computing.
- The **open-source research community** for contributions.

---


