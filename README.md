# OpticalDeviceLib3D 🛠️🔦

**OpticalDeviceLib3D** is a library for modeling optical schemes using synchrotron X‑ray radiation. It enables the calculation of wavefield propagation in free space based on the Huygens–Fresnel principle, and accounts for the interaction of radiation with matter through the complex refractive index. In the projection approximation, the object’s effect is described by a complex transmission function — a standard approach for modelling images of optically thin samples.

The theoretical foundation is provided by the principles of coherent X‑ray optics [1, 2]. Numerical wavefield propagation from the source to the object and then to the detector is performed using algorithms that implement the Huygens–Fresnel principle. This approach allows the computation of complex amplitude and intensity distributions at every stage of the model experiment.

## Features

- Point and extended sources with adjustable energy or wavelength  
- Apertures and stops  
- Parabolic [CRL lenses](https://en.wikipedia.org/wiki/Compound_refractive_lens)  
- 2D and 3D layout construction

## Project structure

The project includes two main Python libraries:

- [`opticaldevicelib_1d`](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_1d.py) — for 2D modelling  
- [`opticaldevicelib_2d`](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/opticaldevicelib_2d.py) — for 3D modelling  

Usage examples are provided in Jupyter notebooks:  
- [2D notebook](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_1d.ipynb)  
- [3D notebook](https://github.com/mamichberdey/OpticalDeviceLib3D/blob/main/test_2d.ipynb)

## Optical setup example

<img width="1331" height="1001" alt="image" src="https://github.com/user-attachments/assets/99056c8f-a060-4a96-a1c3-2a3f03b53f4c" />


## Applications

The framework has been used in the following projects:

### 1. [XRayImagePro](https://github.com/mamichberdey/XRayImagePro)

The library was employed to validate X‑ray image processing methods by comparing experimental data with simulation results. Calculations were performed to generate model images of a test object using the parameters of the experimental setup.

### 2. Automation with neural networks

*(coming soon)*

---

## References

[1] Kohn V.G., Folomeshkin M.S. Feasibility of X-ray beam nanofocusing with compound refractive lenses. *Synch. Rad.*, 2021, V. 28, № 2, P. 419-428. https://doi.org/10.1107/S1600577520016495  
[2] Kohn V.G. 2024. https://xray-optics.ucoz.ru/XR/xrwp.htm
