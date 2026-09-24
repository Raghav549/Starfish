# Starfish Research Basis

This document records research directions and algorithm families used to guide the Starfish engine. It does not claim forensic or identification-grade validation unless separately benchmarked.

## Core research-derived components

### 1. Contextual fingerprint enhancement
Hong, Wan and Jain, IEEE TPAMI (1998), describe adaptive enhancement using local ridge orientation and ridge frequency, with contextual filtering. Starfish follows this principle with local orientation/frequency maps and oriented Gabor filtering.

### 2. STFT-based intrinsic estimation
Chikkerur, Cartwright and Govindaraju, Pattern Recognition (2007), use short-time Fourier analysis to jointly estimate fingerprint foreground, local ridge orientation and local ridge frequency. Starfish exposes these as explicit intermediate fields for diagnostics and enhancement.

### 3. Orientation-field and singular-point analysis
O. Cappelli, Maio and Maltoni, IEEE TPAMI (2002), present systematic directional-field computation and Poincare-index singular-point detection. This motivates adding singular-point maps and orientation consistency checks.

### 4. Enhancement alternatives
Hsieh, Lai and Wang, Pattern Recognition (2003), study wavelet-based enhancement for ridge continuity. Greenberg, Aladjem and Kogan compare local histogram equalization, Wiener filtering, binarization and anisotropic filtering. These motivate a pluggable enhancement stack rather than relying on one filter.

### 5. Minutiae extraction and matching
He et al., Pattern Recognition Letters (2003), describe minutiae sets carrying position, orientation and ridge-neighborhood information and matching with alignment plus ridge information. This motivates richer serialized minutiae templates and a future matcher.

### 6. Deep minutiae extraction
Nguyen, Cao and Jain, MinutiaeNet (2017), integrate enhanced image, orientation field and segmentation map into a neural minutiae extractor, followed by a refinement network. This motivates a future learned detector while retaining an interpretable classical path.

### 7. Fingerprint quality
NIST NFIQ 2 is the reference implementation for ISO/IEC 29794-4 quality assessment and links quality measurements to recognition performance for applicable 500-PPI optical/ink fingerprints. Starfish therefore treats quality as a first-class output and documents applicability limits.

### 8. Open reference systems
NIST NBIS provides MINDTCT for minutiae detection, BOZORTH3 for minutiae-based matching, NFSEG for segmentation and SIVV for spectral validation. These are useful external baselines for future conformance/benchmark work.

## Planned research-grade architecture

Input
→ capture validation / scale estimation
→ segmentation + quality map
→ normalization
→ orientation field + coherence
→ ridge frequency / STFT features
→ multi-method enhancement (Gabor / Wiener / wavelet)
→ binarization + morphology
→ skeletonization
→ crossing-number candidate generation
→ ridge tracing / neighborhood validation
→ false-minutiae suppression
→ minutiae quality
→ singular points / global pattern descriptors
→ compact template serialization
→ optional matcher / benchmark layer.

## Important boundary

Image enhancement can improve visibility of structures that are actually present. It cannot reliably reconstruct arbitrary information absent from the input. A high-contrast output is therefore not itself evidence of higher biometric correctness. Accuracy must be measured against labeled datasets and independent matching/quality metrics.

## Sources
1. Hong, L., Wan, Y., Jain, A. K. “Fingerprint image enhancement: algorithm and performance evaluation.” IEEE TPAMI 20(8), 777–789, 1998. DOI: 10.1109/34.709565.
2. Chikkerur, S., Cartwright, A. N., Govindaraju, V. “Fingerprint enhancement using STFT analysis.” Pattern Recognition 40(1), 198–211, 2007. DOI: 10.1016/j.patcog.2006.05.036.
3. Cappelli, R., Maio, D., Maltoni, D. “Systematic methods for the computation of the directional fields and singular points of fingerprints.” IEEE TPAMI 24(7), 905–919, 2002. DOI: 10.1109/TPAMI.2002.1017618.
4. Hsieh, C.-T., Lai, E., Wang, Y.-C. “An effective algorithm for fingerprint image enhancement based on wavelet transform.” Pattern Recognition 36(2), 303–312, 2003. DOI: 10.1016/S0031-3203(02)00032-8.
5. He, Y., Tian, J., Luo, X., Zhang, T. “Image enhancement and minutiae matching in fingerprint verification.” Pattern Recognition Letters 24(9–10), 1349–1360, 2003. DOI: 10.1016/S0167-8655(02)00376-8.
6. Nguyen, D.-L., Cao, K., Jain, A. K. “Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge.” arXiv:1712.09401, 2017.
7. Tabassi, E. et al. “NIST Fingerprint Image Quality 2.” NISTIR 8382, 2021. ISO/IEC 29794-4 reference implementation.
8. NIST Biometric Image Software (NBIS): MINDTCT, BOZORTH3, NFSEG, SIVV.
