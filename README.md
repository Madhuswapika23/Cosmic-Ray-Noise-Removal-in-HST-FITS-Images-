# Cosmic Ray Noise Removal in HST FITS Images

## Overview

This project focuses on removing cosmic ray noise from astronomical FITS images obtained from the Hubble Space Telescope (HST). The notebook implements a complete deep learning and image-processing pipeline using Python, PyTorch, Astropy, and L.A.Cosmic.

The system:

* Reads calibrated HST FITS images
* Detects cosmic ray artifacts
* Generates pseudo-labels using the L.A.Cosmic algorithm
* Creates training datasets for machine learning
* Trains a neural network-based denoising pipeline
* Saves cleaned FITS outputs

---

## Features

* HST CAL FITS image reader
* Robust percentile normalization
* Cosmic ray detection using L.A.Cosmic
* Synthetic cosmic ray augmentation
* PyTorch dataset and dataloader pipeline
* Model training and validation support
* FITS file cleaning and export
* Visualization of masks and cleaned outputs

---

## Technologies Used

* Python
* PyTorch
* Astropy
* Astro-SCRAPPY (L.A.Cosmic)
* NumPy
* SciPy
* Scikit-image
* Matplotlib
* Pandas

---

## Dataset Structure

The dataset should contain calibrated HST FITS files in the following format:

```bash
Dataset/
 └── CAL/
      └── HST/
           └── *_cal.fits
```

---

## Installation

Install required libraries:

```bash
pip install astropy astroscrappy scipy scikit-image tqdm matplotlib pandas torch
```

---

## Project Workflow

### 1. Environment Setup

The notebook installs and imports all required libraries.

### 2. FITS File Reading

Reads HST calibrated FITS images safely from different HDU formats.

### 3. Image Normalization

Uses percentile normalization to reduce the effect of extreme cosmic ray values.

### 4. Cosmic Ray Detection

Applies the L.A.Cosmic algorithm to generate:

* Cosmic ray masks
* Cleaned pseudo-target images

### 5. Dataset Preparation

Creates image patches and performs:

* Augmentation
* Synthetic cosmic ray generation
* Data splitting for training, validation, and testing

### 6. Visualization

Displays:

* Original contaminated images
* Cosmic ray masks
* Cleaned images

### 7. FITS Export

Saves cleaned FITS files with:

* Cleaned image data
* Cosmic ray mask extensions

---

## Example Output

The notebook generates:

* Cosmic ray masks
* Cleaned astronomical images
* Training-ready datasets
* Exported cleaned FITS files

---

## Applications

* Astronomical image preprocessing
* Space telescope image restoration
* Scientific image denoising
* AI-based astrophysics research

---

## Future Improvements

* U-Net based deep learning architecture
* Transformer-based denoising models
* Real-time inference pipeline
* GPU optimization
* Better synthetic cosmic ray simulation

---

## Author

Madhu Swapnika

---

## License

This project is intended for educational and research purposes.
