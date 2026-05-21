# Cosmic Ray Noise Removal for HST Calibrated FITS Images

A deep learning solution for detecting and removing cosmic ray artifacts from Hubble Space Telescope (HST) calibrated FITS images using a PSF-Aware U-Net architecture.

## 📋 Overview

Cosmic rays are high-energy particles that create artifacts in astronomical images, appearing as bright, isolated pixels or small clusters. This project implements a specialized convolutional neural network trained to identify and remove these artifacts from HST calibrated science imagery while preserving genuine astronomical features.

**Key Features:**
- ✅ PSF-Aware U-Net architecture for accurate cosmic ray detection
- ✅ Handles HST CAL FITS files from multiple instruments
- ✅ L.A.Cosmic pseudo-labeling for automatic dataset generation
- ✅ Threshold optimization with comprehensive evaluation metrics
- ✅ GPU-optimized training with mixed precision (AMP)
- ✅ Detailed benchmark results and per-file analysis

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Validation F1 Score** | 0.9943 |
| **Test F1 Score** | 0.9914 |
| **Test Precision** | 0.9865 |
| **Test Recall** | 0.9967 |
| **Test IoU** | 0.9832 |
| **Accuracy** | 0.9999 |
| **Optimal Threshold** | 0.70 |

The model achieves excellent performance with 99.67% recall and 98.65% precision, meaning it catches nearly all cosmic rays while maintaining minimal false positives.

## 📦 Requirements

```
Python 3.8+
PyTorch 2.0+
astropy
astroscrappy
scipy
scikit-image
numpy
pandas
matplotlib
tqdm
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cosmic-ray-removal.git
cd cosmic-ray-removal

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

- **Source**: HST CAL FITS files (calibrated science images)
- **Total Files**: 330+ images
- **Instruments**: Multiple HST instruments supported
- **Format**: FITS (Flexible Image Transport System)
- **Data Structure**:
  ```
  dataset/CAL/MAST_2026-04-29T1538/HST/*_cal.fits
  ```

### Automatic Label Generation

The project uses **L.A.Cosmic** (Laplacian-Astroscrappy Cosmic ray detection) to automatically generate pseudo-labels from unlabeled FITS data. This enables semi-supervised learning without manual annotation.

```python
# L.A.Cosmic is applied during preprocessing
cosmic_mask = astroscrappy.detect_cosmics(image)
```

## 🚀 Quick Start

### 1. Data Preparation

```python
from pathlib import Path
from notebook_utils import read_hst_cal_fits

# Read HST FITS file
image, header, hdu_name = read_hst_cal_fits("path/to/image_cal.fits")

# Normalize
from notebook_utils import percentile_normalize
normalized_image = percentile_normalize(image)
```

### 2. Training

```python
from model import PSFAwareUNet
from train import train_model

# Initialize model
model = PSFAwareUNet(in_channels=1, out_channels=1, depth=4)

# Train
history = train_model(
    model, 
    train_loader, 
    val_loader,
    epochs=50,
    learning_rate=1e-4,
    device="cuda"
)
```

### 3. Inference

```python
from infer import remove_cosmic_rays

# Remove cosmic rays from image
cleaned_image = remove_cosmic_rays(
    image, 
    model, 
    threshold=0.70,
    patch_size=128
)

# Save cleaned image
save_fits_image(cleaned_image, "cleaned_image_cal.fits")
```

## 🏗️ Architecture

### PSF-Aware U-Net

The model uses a U-Net encoder-decoder architecture with:

- **Encoder**: Convolutional blocks with batch normalization and ReLU activation
- **Bottleneck**: Dense feature representation
- **Decoder**: Transposed convolutions for upsampling
- **Skip Connections**: Direct pathways from encoder to decoder
- **PSF Awareness**: Incorporates point spread function information for instrument-specific optimization

```
Input (H × W × 1)
    ↓
Encoder (4 levels of downsampling)
    ↓
Bottleneck
    ↓
Decoder (4 levels of upsampling)
    ↓
Output (H × W × 1) - Cosmic ray probability map
```

## 📈 Training Pipeline

1. **Data Loading**: Stream HST FITS files with automatic caching
2. **Preprocessing**: Percentile normalization, patch extraction (128×128)
3. **Augmentation**: Random rotations, flips, and intensity jittering
4. **Training**: Mixed precision (AMP) with AdamW optimizer
5. **Validation**: Threshold sweep for F1 optimization
6. **Testing**: Per-file and per-patch metrics
7. **Benchmark**: Comparison against L.A.Cosmic baseline

## 📊 Evaluation Metrics

- **Pixel-Level**: Precision, Recall, F1, IoU, Accuracy
- **Patch-Level**: Per-256×256 patch performance aggregation
- **File-Level**: Per-FITS-file cosmic ray fraction
- **Threshold Analysis**: F1 vs. Threshold curves

## 📁 Output Files

```
model_results/
├── psf_aware_unet_final.pth          # Trained model weights
├── training_history.csv              # Loss & metrics per epoch
├── benchmark_results.csv             # L.A.Cosmic comparison
├── test_threshold_results.csv        # F1 vs. threshold on test set
└── val_threshold_results.csv         # F1 vs. threshold on validation set
```

## 🔧 Configuration

Key hyperparameters (configurable in notebook):

```python
# Model
PATCH_SIZE = 128
DEPTH = 4  # U-Net depth

# Training
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW"

# Data
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Inference
OPTIMAL_THRESHOLD = 0.70
```

## 💡 Usage Examples

### Remove Cosmic Rays from Single Image

```python
import torch
from pathlib import Path

# Load model
model = PSFAwareUNet(in_channels=1, out_channels=1)
checkpoint = torch.load("psf_aware_unet_final.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Process image
with torch.no_grad():
    image_tensor = torch.from_numpy(normalized_image).unsqueeze(0).unsqueeze(0)
    prediction = model(image_tensor).squeeze().numpy()
    
# Apply threshold
cosmic_mask = prediction > 0.70
cleaned = image.copy()
cleaned[cosmic_mask] = scipy.ndimage.median_filter(cleaned, size=3)[cosmic_mask]
```

### Batch Processing Directory

```python
from pathlib import Path
from infer import process_fits_directory

# Process all FITS files in directory
process_fits_directory(
    input_dir="raw_fits/",
    output_dir="cleaned_fits/",
    model_path="psf_aware_unet_final.pth",
    threshold=0.70
)
```

## 📚 References

- **U-Net Architecture**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **L.A.Cosmic**: van Dokkum, "Cosmic-Ray Rejection by Laplacian Edge Detection" (2001)
- **Astroscrappy**: Python implementation of L.A.Cosmic
- **HST Data**: https://mast.stsci.edu/

## ⚠️ Limitations & Future Work

### Current Limitations
- Model trained on specific HST instruments (generalization to other telescopes untested)
- Assumes cosmic rays are isolated to single/few pixels
- Does not handle saturated hot pixels or bad columns
- FITS files must be in SCI, PRIMARY, or IMAGE HDU

### Future Improvements
- [ ] Support for extended cosmic ray tracks
- [ ] Multi-instrument transfer learning
- [ ] Uncertainty quantification (Bayesian approach)
- [ ] Real-time GPU streaming for large mosaics
- [ ] JWST FITS file compatibility
- [ ] Confidence maps alongside cleaned images

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@software{cosmic_ray_removal_2026,
  author = {Your Name},
  title = {Cosmic Ray Noise Removal for HST Calibrated FITS Images},
  year = {2026},
  url = {https://github.com/yourusername/cosmic-ray-removal}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an Issue on GitHub
- Contact: [your-email@example.com]
- Kaggle Notebook: [link-to-kaggle-notebook]

## 🙏 Acknowledgments

- Hubble Space Telescope mission for providing invaluable astronomical data
- Astroscrappy library for L.A.Cosmic implementation
- PyTorch and scientific Python community
- All contributors and collaborators

---

**Last Updated**: May 2026  
**Version**: 1.0.0
