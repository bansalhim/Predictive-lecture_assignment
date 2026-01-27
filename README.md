# Assignment 2 – Learning Probability Density Functions using Data Only

## Objective
To learn an unknown probability density function (PDF) of a **transformed random variable** using a **Generative Adversarial Network (GAN)**, without assuming any parametric form of the distribution.

---
## 🧮 Transformation
The transformation applied to each NO₂ sample `x` is:
z = x + a_r * sin(b_r * x)
a_r = 0.5 * (ROLL_NO % 7)
b_r = 0.3 * ((ROLL_NO % 5) + 1)
For roll number **102303786**: 
a_r = 2.0
b_r = 0.6

## 🧰 Methodology
1. **Dataset Loading:**  
   - NO₂ data extracted from `archive (3).zip`.  
   - Handles both `no2` and `NO2` column names.  
   - Missing values removed and only values in `[0, 1000)` retained.

2. **Data Transformation and Scaling:**  
   - Each sample transformed to `z = x + a_r * sin(b_r * x)`.  
   - Standarized using `StandardScaler` for stable GAN training.

3. **GAN Model Design:**  
   - **Generator:** Fully connected neural network with LeakyReLU and BatchNorm layers. Input: Gaussian noise (`N(0,1)`), Output: fake `z` values.  
   - **Discriminator:** Fully connected network with LeakyReLU and Sigmoid output. Classifies real vs generated samples.

4. **Training:**  
   - Loss: Binary Cross-Entropy (BCE).  
   - Optimizer: Adam  
     ```
     Generator lr = 0.0002, betas = (0.5, 0.999)
     Discriminator lr = 0.0004, betas = (0.5, 0.999)
     ```  
   - Batch size: 256, Epochs: 1000.  
   - Labels: Real = 0.9, Fake = 0.0.  
   - Generator and discriminator updated alternately.

5. **Sample Generation and PDF Estimation:**  
   - 100,000 fake samples generated from the trained generator.  
   - Inverse-transformed to original scale using `StandardScaler`.  
   - Kernel Density Estimation (KDE) used to approximate PDF.  
   - Compared gnerated PDF with histogram of real transformed data.

## 📊 Results
| Parameter | Value |
|-----------|-------|
| Roll Number | 102303786 |
| a_r | 2.0 |
| b_r | 0.6 |
| Epochs | 1000 |
| Batch Size | 256 |
| Learning Rate | 0.0002 (Generator), 0.0004 (Discriminator) |
| Noise Source | N(0,1) |

### **Result Graphs**
- Real vs Generated PDF comparison plot saved at `output_gan/z_pdf.png` that contain plot.

## 💬 Observations
- **Mode coverage:** Generator captures main modes of the real `z` distribution.  
- **Training stability:** Loss curves stabilize after several hundred epochs.  
- **Distribution quality:** Generated samples follow the shape of real `z` distribution. Minor variance at tails due to data imbalance.

## 🧠 Conclusion
The GAN successfully learns the underlying probability distribution of the transformed NO₂ variable **without assuming any parametric form**.  
Generated PDF closely matches empirical distribution from real data, showing GAN’s effectiveness in modeling unknown distributions purely from samples.

## 🗂️ Files Included
- `Predictive_lecture_assignment.ipynb` → Colab notebook with full code, training, and plots.  
- `output_gan/z_pdf.png` → PDF comparison plot.  
- `README.md` → Explanation of methodology, results, and observations.

## 📌 References
- Kaggle Dataset: [India Air Quality Data](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data)  
  this is datset used in this assignment


