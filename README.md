# PINN for 2D Navier-Stokes Flow / 2B Navier-Stokes Akışı için PINN

Physics-Informed Neural Network (PINN) solving the 2D incompressible Navier-Stokes equations, enhanced with Fourier Feature Encoding.

2 boyutlu sıkıştırılamaz Navier-Stokes denklemlerini çözen, Fourier Feature Encoding ile güçlendirilmiş Fizik Bilgili Sinir Ağı (PINN).

[English](#english) | [Türkçe](#türkçe)

---

## English

### Overview

This project implements a Physics-Informed Neural Network (PINN) to solve the 2D incompressible Navier-Stokes equations. Given spatial coordinates `(x, y)` and time `t`, the model predicts the velocity field `(u, v)` and pressure `p`, while being constrained during training to also satisfy the governing PDEs (continuity and momentum equations), not just fit the data.

This work was originally developed as a graduation project (Bitirme Projesi II) at Istanbul University – Cerrahpaşa, Department of Mechanical Engineering, and an extended version was published as a conference abstract at **ICAENS 2025** (8th International Conference on Applied Engineering and Natural Sciences).

### Method

- **Dataset:** High-resolution 2D incompressible, inhomogeneous flow simulation data (`ns_incomp_inhom_2d_512-0sample0.h5`), shared by the University of Stuttgart. The first 10 time steps were used, downsampled from 512×512 to 128×128 via `scipy.ndimage.zoom` to reduce computational cost.
- **Input encoding:** Fourier Feature Encoding was applied to the `(x, y, t)` inputs to counter *spectral bias* — the tendency of standard neural networks to learn low-frequency patterns more easily than high-frequency ones. Six frequency bands were used per input dimension.
- **Architecture:** Fully connected network with 5 hidden layers, 128 neurons per layer, `tanh` activation. Output layer predicts `u`, `v`, and `p`.
- **Loss function:** Weighted sum of data loss and physics loss (residuals of the Navier-Stokes and continuity equations, computed via automatic differentiation):

  ```
  L_total = λ_data · L_data + λ_phys · L_phys
  ```

  with `λ_data = 1.0` and `λ_phys = 10.0`, giving physical consistency a dominant role.

- **Training:** Adam optimizer, 1000 epochs, initial learning rate `0.001` halved every 200 epochs. Trained on TensorFlow 2.0 with GPU acceleration (Google Colab), taking ~30 minutes.

### Results

- Mean Relative Error (MRE) for both `u` and `v` velocity components: **< 1%**.
- Predicted vs. ground-truth velocity fields show strong agreement, particularly in low- and mid-frequency flow regions.
- Vortex structures and flow direction changes are captured accurately.
- Higher local error observed in high-frequency / sharp-transition regions — a known limitation of the current architecture, suggested as a direction for future work (deeper networks, adaptive loss weighting, positional encoding).

### Repository Structure

```
.
├── README.md
├── src/                  # Model, training, and preprocessing scripts
├── notebooks/             # Jupyter notebooks (if any)
├── results/                # Output plots (error maps, prediction vs. ground truth)
└── requirements.txt
```

*(Adjust this section to match your actual folder layout once files are uploaded.)*

### Requirements

```
tensorflow>=2.0
numpy
scipy
h5py
matplotlib
```

### Reference

Onur Eren Atar, Mehmet Şirin Demir — *Fourier Özellik Kodlaması ile Geliştirilmiş Fizik Bilgili Sinir Ağları: İki Boyutlu Navier-Stokes Denklemlerine Bir Uygulama.* 8th International Conference on Applied Engineering and Natural Sciences (ICAENS 2025), Konya, Türkiye.

### Key Citations

- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.* Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2018.10.045
- Karniadakis, G. E. et al. (2021). *Physics-Informed Machine Learning.* Nature Reviews Physics. https://doi.org/10.1038/s42254-021-00314-5
- Dataset: University of Stuttgart, DaRUS Data Repository — `ns_incomp_inhom_2d_512-0sample0.h5`

---

## Türkçe

### Genel Bakış

Bu proje, 2 boyutlu sıkıştırılamaz Navier-Stokes denklemlerini çözmek için bir Fizik Bilgili Sinir Ağı (Physics-Informed Neural Network, PINN) uygular. Model, konum `(x, y)` ve zaman `t` bilgileri verildiğinde hız alanı `(u, v)` ve basınç `p` değerlerini tahmin eder; eğitim sırasında sadece veriye uymakla kalmaz, aynı zamanda temel fiziksel denklemleri (süreklilik ve momentum denklemleri) de sağlayacak şekilde kısıtlanır.

Bu çalışma, İstanbul Üniversitesi – Cerrahpaşa Makine Mühendisliği Bölümü'nde bir bitirme projesi (Bitirme Projesi II) olarak geliştirilmiş, genişletilmiş bir versiyonu ise **ICAENS 2025** (8. Uluslararası Uygulamalı Mühendislik ve Doğa Bilimleri Konferansı) özet kitabında yayımlanmıştır.

### Yöntem

- **Veri seti:** Stuttgart Üniversitesi tarafından paylaşılan yüksek çözünürlüklü, 2 boyutlu sıkıştırılamaz ve homojen olmayan akış simülasyonu verisi (`ns_incomp_inhom_2d_512-0sample0.h5`). Hesaplama maliyetini azaltmak amacıyla ilk 10 zaman adımı kullanılmış, veri `scipy.ndimage.zoom` ile 512×512'den 128×128 çözünürlüğe indirgenmiştir.
- **Girdi kodlama:** Modelin yüksek frekanslı bileşenleri daha iyi öğrenebilmesi için `(x, y, t)` girdilerine Fourier Feature Encoding uygulanmıştır — bu, standart sinir ağlarının düşük frekanslı örüntüleri daha kolay öğrenme eğilimi olan "spectral bias" sorununu azaltır. Her girdi boyutu için 6 farklı frekans kullanılmıştır.
- **Mimari:** 5 gizli katmanlı, her katmanda 128 nöron bulunan, `tanh` aktivasyon fonksiyonlu tam bağlantılı sinir ağı. Çıkış katmanı `u`, `v` ve `p` değerlerini üretir.
- **Kayıp fonksiyonu:** Veri kaybı ile fiziksel kaybın (Navier-Stokes ve süreklilik denklemlerinin otomatik türev ile hesaplanan ihlalleri) ağırlıklı toplamı:

  ```
  L_toplam = λ_data · L_veri + λ_phys · L_fizik
  ```

  `λ_data = 1.0` ve `λ_phys = 10.0` olarak belirlenmiş, fiziksel tutarlılığa baskın rol verilmiştir.

- **Eğitim:** Adam optimizasyon algoritması, 1000 epoch, başlangıç öğrenme oranı `0.001`, her 200 epochta yarıya düşürülmüştür. TensorFlow 2.0 ile Google Colab üzerinde GPU destekli olarak eğitilmiş, yaklaşık 30 dakika sürmüştür.

### Sonuçlar

- `u` ve `v` hız bileşenleri için Ortalama Oransal Hata (MRE): **%1'in altında**.
- Tahmin edilen ve gerçek hız alanları, özellikle düşük ve orta frekanslı bölgelerde güçlü bir uyum göstermektedir.
- Girdap yapıları ve akış yönü değişimleri doğru şekilde yakalanmıştır.
- Yüksek frekanslı / keskin geçiş bölgelerinde göreli hata daha yüksektir — bu, mevcut mimarinin bilinen bir sınırlaması olup, gelecek çalışmalar için bir yön olarak önerilmiştir (daha derin ağlar, adaptif kayıp ağırlıklandırma, positional encoding).

### Kaynak

Onur Eren Atar, Mehmet Şirin Demir — *Fourier Özellik Kodlaması ile Geliştirilmiş Fizik Bilgili Sinir Ağları: İki Boyutlu Navier-Stokes Denklemlerine Bir Uygulama.* 8. Uluslararası Uygulamalı Mühendislik ve Doğa Bilimleri Konferansı (ICAENS 2025), Konya, Türkiye.

### Danışman

Dr. Öğr. Üyesi Mehmet Şirin Demir — İstanbul Üniversitesi – Cerrahpaşa

---

## License / Lisans

MIT
