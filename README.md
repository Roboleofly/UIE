# Underwater Image Enhancement: A Defect-Aware Multi-Scale Fusion Approach

This repository implements an **underwater image enhancement algorithm based on defect detection**, designed to improve visual quality and preserve fine details in underwater scenes. The method targets common underwater degradations such as color distortion, low contrast, and blurriness, and enhances image regions adaptively by first detecting local defects and then guiding a multi-scale enhancement and fusion process.

---

## 🧠 Algorithm Overview

Underwater images often suffer from severe quality degradation due to **light absorption and scattering**, leading to poor visibility, color shift, and contrast loss. Traditional enhancement methods like Retinex, UDCP, and histogram-based corrections typically operate at the pixel level, without considering spatial or semantic defect variations across the image.

This project proposes a **content- and defect-aware enhancement strategy**:

### 1. Defect Detection Module
- Detects visually degraded regions using a combination of:
  - **Luminance flatness**
  - **Color deviation**
  - **Local contrast weakness**
- Generates a **defect map** to highlight problematic areas.

### 2. Multi-Scale Enhancement
- Applies adaptive enhancement operators (e.g., contrast stretching, white balance, dehazing) at different scales.
- Considers both global and local image properties to avoid over-enhancement.

### 3. Guided Fusion
- Combines enhanced results from multiple scales and operations.
- Uses the defect map to assign spatially varying weights, ensuring that severely degraded areas receive stronger correction while preserving good regions.

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- scikit-image
- matplotlib
