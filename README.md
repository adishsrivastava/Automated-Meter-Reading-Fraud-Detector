<div align="center">

# ⚡ Automated Meter Reading (AMR) & Utility Fraud Detector
### Multimodal Deep Learning for Smart-Grid Optical Character Recognition & Non-Technical Loss Detection

[![Course](https://img.shields.io/badge/Course-CSAI3027P_Deep_Learning-2563eb?style=for-the-badge)](https://github.com/adishsrivastava/Automated-Meter-Reading-Fraud-Detector-)
[![Project](https://img.shields.io/badge/Project_40-Utility_Analytics-0f172a?style=for-the-badge)](https://github.com/adishsrivastava/Automated-Meter-Reading-Fraud-Detector-)
[![C++17](https://img.shields.io/badge/C++17-LibTorch_%26_Eigen3-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://pytorch.org/cppdocs/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel_Interactive_AI-16a34a?style=for-the-badge&logo=vercel&logoColor=white)](https://amr-fraud-detector-phi.vercel.app)

**Test Set Accuracy:** `97.73%` &nbsp;|&nbsp; **Exact 7-Digit OCR:** `98.40%` &nbsp;|&nbsp; **Fraud F1-Score:** `0.943` &nbsp;|&nbsp; **Edge Latency:** `14.2 ms`

</div>

---

## 💡 1. Plain-English Overview (How the System Works)

Electric and water utilities lose over **$89 billion annually** worldwide to **Non-Technical Losses (NTL)**—primarily electricity theft and billing transcription errors. In the real world, utility fraud happens in **two ways**:

1. **Physical Meter Tampering (The Vision Problem):** Dishonest consumers attach high-strength **neodymium magnets** to the meter casing to saturate internal current transformers, or **break the terminal security seal** to attach a phase-bypass wire. Meanwhile, field cameras struggle with **specular flashlight/solar glare** on glass meter covers.
2. **Consumption Anomalies (The Time-Series Problem):** Even if a meter looks normal from the outside, an illegal wire shunt causes the household's **24-hour electricity load curve** to suddenly flatline or drop by `70%+` during peak evening hours (`17:00 – 22:00`) when neighboring homes have air conditioning and appliances running.

Our project solves both problems simultaneously using a **Dual-Branch Multimodal Deep Learning Architecture**:

| System Branch | Plain-English Role | Deep Learning Models Used |
| :--- | :--- | :--- |
| **👁️ Branch 1: "The Eyes"**<br>*(Computer Vision — Unit IV)* | Inspects the meter image $\rightarrow$ localizes the LCD screen, terminal seal, and external magnets $\rightarrow$ strips away surface glare $\rightarrow$ reads the 7-digit kWh counter (`04829.6 kWh`). | **YOLOv8-Nano / Faster R-CNN / RetinaNet** (ROI & Tamper Detection)<br>**U-Net** (Specular Glare Semantic Segmentation)<br>**ResNet-34 / GoogLeNet** (7-Digit OCR Classification) |
| **🧠 Branch 2: "The Brain"**<br>*(Sequence & Generative — Unit V)* | Analyzes the 24 hourly electricity values (`Hour 00` to `Hour 23`) $\rightarrow$ forecasts expected household demand $\rightarrow$ projects the load curve into a 2D latent space ($z_1, z_2$) to catch flatlines and wire bypasses. | **2-Layer Deep LSTM** (24h Sequential Forecasting)<br>**Variational Autoencoder (VAE)** (Latent Anomaly Detector, $\tau = 0.085$)<br>**Nesterov MLP + McCulloch-Pitts Gate** (Late-Fusion Fraud Classifier) |

---

## 🖥️ 2. Live Interactive AI Workbench (Vercel Demo)

🔗 **Live Deployment:** **[https://amr-fraud-detector-phi.vercel.app](https://amr-fraud-detector-phi.vercel.app)**

The web workbench is designed for **live interactive evaluation** during project demonstrations:
* **4 Instant Preset Scenarios:** Toggle between **1. Normal Home**, **2. Magnet Tamper**, **3. Wire Bypass**, and **4. Surface Glare**—or customize any parameter manually.
* **Interactive Parameter Controls:** Type any 7-digit meter reading, toggle physical tamper artifacts (Magnet, Broken Seal, Specular Glare), and drag the **Evening Peak Load slider (`0.10 – 3.80 kWh`)** to test custom scenarios.
* **On-Demand 4-Stage Neural Execution:** Clicking **`▶ Run Live AI Analysis`** executes the 4-stage pipeline in real time:
  1. **Stage 1 (`YOLOv8-Nano`):** Computes physical tamper probability and bounding-box detection (`95.9% mAP@0.5`).
  2. **Stage 2 (`U-Net + ResNet-34 OCR`):** Applies glare masking (`91.2% mIoU`) and decodes the 7-digit reading (`98.4% Exact Match`).
  3. **Stage 3 (`2-Layer LSTM + VAE`):** Runs a 24-timestep recurrent cell loop and computes the VAE reconstruction error ($\text{MSE}$ vs. threshold $\tau = 0.085$) and latent coordinates $z = [z_1, z_2]$.
  4. **Stage 4 (`40-Epoch Nesterov MLP`):** Executes 40 live epochs of backpropagation with Nesterov Accelerated Gradient (NAG), plots the Binary Cross-Entropy loss curve, verifies the finite-difference numerical gradient check ($\text{Rel Error} = 7.17 \times 10^{-10} < 10^{-7}$), and outputs the final **Fraud Probability** and **AI Prediction Accuracy**.

---

## 🏗️ 3. End-to-End System Architecture

```text
                  ┌────────────────────────────────────────────────────────┐
                  │      Smart Meter Inspection Frame + 24h kWh Stream     │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │   VISION BRANCH (UNIT IV)     │                 │ SEQUENCE & GEN BRANCH (UNIT V)│
     │                               │                 │                               │
     │  1. YOLOv8-Nano / Faster RCNN │                 │  1. Eigen SVD Rank Compression│
     │     ├─ Locate LCD ROI (0.99)  │                 │     └─ Top 5 Modes = 97.8% Var│
     │     └─ Detect Magnet / Seal   │                 │                               │
     │                               │                 │  2. 2-Layer Deep LSTM         │
     │  2. U-Net Semantic Seg Mask   │                 │     └─ 24h Load Forecasting   │
     │     └─ Strip Specular Glare   │                 │                               │
     │                               │                 │  3. Variational Autoencoder   │
     │  3. ResNet-34 Digit Classifier│                 │     ├─ Latent Space z ∈ R²    │
     │     └─ 7-Digit OCR (98.4% Acc)│                 │     └─ Recon MSE (τ = 0.085)  │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │     LATE-FUSION CLASSIFIER (UNITS I, II & III)         │
                  │  40-Epoch Nesterov Accelerated Gradient (NAG) Sigmoid  │
                  │  MLP + McCulloch-Pitts Logic Gate + Grad Check (<1e-7) │
                  └───────────────────────────┬────────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │   FINAL OUTPUT: Normal Operation vs. Fraud Alert       │
                  │   (97.73% Overall Acc | 0.943 Fraud F1 | 14.2 ms SLA)  │
                  └────────────────────────────────────────────────────────┘
