# SpectraCNN
SpectraCNN models and datasets
Dataset download website:https://www.kaggle.com/datasets/zhangdejia/spectracnn/data
# Dataset Description

## Data Source and Subjects

This dataset was collected from the same area in Changchun, Jilin Province, China (2020). Samples comprise three rice varieties bred by the Northeast Institute of Geography and Agroecology, Chinese Academy of Sciences (NEIGAE, CAS): Dongdao 4 (d4), Dongdao 12 (d12), and Dongdao 122 (d122). All varieties were harvested in the same season to ensure consistent provenance and climatic conditions, thereby reducing environmental effects on spectral measurements.

## Imaging System and Experimental Platform

Hyperspectral acquisition was conducted at NEIGAE, CAS (Changchun, China). The experimental platform consisted of a CCD camera, a hyperspectral imaging camera, a mounting tower, a linear translation stage, illumination modules with dedicated power supplies, and SpectroNET Pro software running on a computer (see Figure 1).
The imaging spectrometer covered 392.38–1011.01 nm with 462 spectral channels, providing a spatial resolution of approximately 0.15 mm/pixel and a spectral resolution of ~1.3 nm. Supporting data-processing software included ENVI 5.3, The Unscrambler X 10.4, Excel 2019, Origin 2018, MATLAB R2020, and PyCharm, enabling end-to-end analysis from raw images to feature extraction and modeling.

## Acquisition Design and Sample Size

For each variety, 200 seeds were randomly selected and placed in a single layer on a black translation stage using a 10×20 template, ensuring no overlap or adhesion. Each layout was scanned to obtain both hyperspectral and RGB images. This procedure was repeated for 5 groups, yielding 15 hyperspectral image sets across the three varieties and a total of 3,000 seeds (approximately 1,000 per class).

## Radiometric Calibration and Quality Control

To account for illumination nonuniformity and CCD dark current, all hyperspectral data underwent white/black reference calibration [@Spectral_image_correction1; @Spectral_image_correction2] to convert radiance to reflectance. The pixel-wise correction is:

$$
I=\frac{I_{r}-I_{d}}{I_{w}-I_{d}}
$$

where (I_r) is the raw hyperspectral image, (I_d) is the dark frame acquired with the lens fully occluded, (I_w) is the white reference acquired from a standard polyurethane panel, and (I) is the calibrated reflectance image. Following expert review, outlier samples were removed to improve the reliability of subsequent modeling.

## Preprocessing and Segmentation Pipeline

1. **Invalid region removal**: Scene cropping was applied to remove the translation stage and other irrelevant regions, reducing background reflections that could interfere with seed segmentation.
2. **Mask generation**: Using the grayscale image at band 200 (656.1300 nm), where the seed–background contrast was maximal, an appropriate threshold was set to obtain an initial binary mask for each image.
3. **Object segmentation and representation**: Connected components were extracted from the binary mask to obtain the boundary pixels of each seed. An ellipse was fitted to each object to compute its center and rotation angle; the fitted boundary then delimited all pixel coordinates within each whole seed.
4. **Spectral feature aggregation**: For each seed, across all 462 bands, three statistics—**mean**, **median**, and **mode**—were computed over the pixels within the seed region, producing three seed-level hyperspectral datasets (“mean dataset,” “median dataset,” and “mode dataset”), each record containing 462 features.

## Data Split and Size Summary

All three datasets were split into training and test sets in a **2:1** ratio using stratification by class. Out of 3,000 seeds in total, approximately 2,000 were assigned to training and ~1,000 to testing; per class this corresponds to ~667 training seeds and ~333 test seeds (exact integer counts are provided in Table 1).

## Overview of Spectral Characteristics

Figure 2 shows the average reflectance curves for the three rice varieties under the three statistical protocols (mean/median/mode), with wavelength on the x-axis and reflectance on the y-axis. The figure illustrates spectral-shape differences in the visible–near-infrared range and highlights key absorption/reflectance features, informing subsequent feature selection, band prioritization, and classification modeling.

## Potential Tasks and Applications

* **Variety identification and classification**: Supervised classification and domain adaptation using the 462-dimensional seed-level spectral vectors.
* **Band selection/model interpretability**: Contribution analysis for key bands such as 656.1300 nm.
* **Traditional vs. deep methods**: Benchmarking and replication with PLS-DA, SVM, RF, and 1D/2D spectral neural networks.
* **Image-to-spectrum pipeline verification**: Reproducing and optimizing the end-to-end workflow from radiometric calibration and mask-based segmentation to seed-level aggregation.

## Reproducibility Notes

* Use the same white/black reference calibration strategy and formula as above.
* Generate masks by thresholding the **200th band (656.1300 nm)**, then perform connected-component labeling and ellipse fitting.
* Aggregate pixel spectra within each seed to obtain three 462-dimensional feature sets (mean/median/mode).
* Apply a stratified **2:1** train/test split to maintain class balance.
