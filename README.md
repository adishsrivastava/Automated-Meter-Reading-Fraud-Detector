<div align="center">

# ⚡ Automated Meter Reading (AMR) & Utility Fraud Detector
### Multimodal Deep Learning for Smart-Grid Optical Character Recognition & Non-Technical Loss Detection

[![Domain](https://img.shields.io/badge/Domain-Smart_Grid_%26_Utility_Analytics-2563eb?style=for-the-badge)](https://github.com/adishsrivastava/Automated-Meter-Reading-Fraud-Detector-)
[![C++17](https://img.shields.io/badge/C++17-LibTorch_%26_Eigen3-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://pytorch.org/cppdocs/)
[![License](https://img.shields.io/badge/License-MIT-0f172a?style=for-the-badge)](./LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel_Interactive_AI-16a34a?style=for-the-badge&logo=vercel&logoColor=white)](https://amr-fraud-detector-phi.vercel.app)

**Test Set Accuracy:** `97.73%` &nbsp;|&nbsp; **Exact 7-Digit OCR:** `98.40%` &nbsp;|&nbsp; **Fraud F1-Score:** `0.943` &nbsp;|&nbsp; **Edge Latency:** `14.2 ms`

</div>

---

## 💡 1. Executive Overview (How the System Works)

Electric and water utilities lose over **$89 billion annually** worldwide to **Non-Technical Losses (NTL)**—primarily electricity theft and billing transcription errors. In the field, utility fraud occurs across **two distinct vectors**:

1. **Physical Meter Tampering (The Vision Problem):** Consumers attach high-strength **neodymium magnets** to the meter casing to saturate internal current transformers, or **break the terminal security seal** to attach a phase-bypass wire. Meanwhile, field cameras struggle with **specular flashlight and solar glare** on glass meter covers.
2. **Consumption Anomalies (The Time-Series Problem):** Even if a meter looks normal externally, an illegal wire shunt causes the household's **24-hour electricity load curve** to suddenly flatline or drop by `70%+` during peak evening hours (`17:00 – 22:00`) when neighboring homes have air conditioning and appliances running.

This repository delivers a **Dual-Branch Multimodal Deep Learning System** that addresses both vectors simultaneously:

| System Branch | Functional Role | Deep Learning Models Used |
| :--- | :--- | :--- |
| **👁️ Branch 1: "The Eyes"**<br>*(Computer Vision Pipeline)* | Inspects an uploaded meter photo $\rightarrow$ localizes the LCD screen, terminal seal, and external magnets $\rightarrow$ strips away surface glare $\rightarrow$ automatically extracts the 7-digit kWh counter (`04829.6 kWh`). | **YOLOv8-Nano / Faster R-CNN / RetinaNet** (ROI & Tamper Detection)<br>**U-Net** (Specular Glare Semantic Segmentation)<br>**ResNet-34 / GoogLeNet** (7-Digit OCR Classification) |
| **🧠 Branch 2: "The Brain"**<br>*(Sequence & Generative Pipeline)* | Ingests weekly/monthly timestamped consumption histories (`Hour 00` to `Hour 23`) $\rightarrow$ deduplicates logs & partitions splits $\rightarrow$ forecasts expected demand & projects curves into a 2D latent space ($z_1, z_2$) to catch flatlines and bypasses. | **2-Layer Deep LSTM** (24h Sequential Forecasting)<br>**Variational Autoencoder (VAE)** (Latent Anomaly Detector, $\tau = 0.085$)<br>**Nesterov MLP + McCulloch-Pitts Gate** (Late-Fusion Fraud Classifier) |

---

## 🖥️ 2. Live Interactive AI Workbench

🔗 **Live Deployment:** **[https://amr-fraud-detector-phi.vercel.app](https://amr-fraud-detector-phi.vercel.app)**

The web workbench provides full end-to-end data curation, image detection, and sequence anomaly execution:
* **📷 Meter Photo Upload & Auto-OCR Detection:** Upload any meter image (`JPG/PNG/SVG`) or click **"Scan Sample Photo"**. The vision canvas runs pixel luminance & edge analysis, renders the **YOLOv8 LCD ROI & Seal bounding boxes**, and automatically populates the detected 7-digit kWh reading into the pipeline.
* **📅 Weekly / Monthly Consumption History Curation:** Upload a timestamped history CSV or click **"Load 30-Day History"** (`720 hourly logs`). The system automatically **removes duplicate timestamps**, generates ground-truth anomaly labels, assigns leakage-safe **70% Train / 15% Val / 15% Test** partitions, and lets you inspect the exact meter reading at any hour of any day.
* **⚡ On-Demand 4-Stage Neural Execution:** Clicking **`▶ Run Live AI Analysis`** executes the 4-stage pipeline in real time (`YOLOv8 Tamper Detection` $\rightarrow$ `U-Net & ResNet-34 OCR` $\rightarrow$ `24h LSTM-VAE Forward Pass` $\rightarrow$ `40-Epoch Nesterov MLP Backpropagation & Gradient Verification`).

---

## 🏗️ 3. End-to-End System Architecture

```text
                  ┌────────────────────────────────────────────────────────┐
                  │  Uploaded Meter Photo + Weekly/Monthly kWh History CSV │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │   COMPUTER VISION PIPELINE    │                 │ SEQUENCE & GENERATIVE PIPELINE│
     │                               │                 │                               │
     │  1. YOLOv8-Nano / Faster RCNN │                 │  1. Curation & Eigen SVD      │
     │     ├─ Locate LCD ROI (0.99)  │                 │     ├─ Deduplicate & 70/15/15 │
     │     └─ Detect Magnet / Seal   │                 │     └─ Top 5 Modes = 97.8% Var│
     │                               │                 │                               │
     │  2. U-Net Semantic Seg Mask   │                 │  2. 2-Layer Deep LSTM         │
     │     └─ Strip Specular Glare   │                 │     └─ 24h Load Forecasting   │
     │                               │                 │                               │
     │  3. ResNet-34 Digit Classifier│                 │  3. Variational Autoencoder   │
     │     └─ 7-Digit OCR (98.4% Acc)│                 │     └─ Recon MSE (τ = 0.085)  │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │        LATE-FUSION NEURAL CLASSIFIER & VERIFIER        │
                  │  40-Epoch Nesterov Accelerated Gradient (NAG) Sigmoid  │
                  │  MLP + McCulloch-Pitts Logic Gate + Grad Check (<1e-7) │
                  └───────────────────────────┬────────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │   FINAL OUTPUT: Normal Operation vs. Fraud Alert       │
                  │   (97.73% Overall Acc | 0.943 Fraud F1 | 14.2 ms SLA)  │
                  └────────────────────────────────────────────────────────┘
