# 🚀 ModelForge AutoML

<p align="center">
  <b>From Data to Decisions — Automatically.</b><br>
  A production-ready AutoML platform that transforms raw data into powerful machine learning models with minimal effort.
</p>

---

## 🌟 Overview

**ModelForge AutoML** is an intelligent, end-to-end machine learning system designed to automate the complete model development lifecycle. It eliminates manual complexity by integrating data preprocessing, feature engineering, model selection, and hyperparameter tuning into a seamless pipeline.

Built with scalability and performance in mind, it empowers users to rapidly develop high-quality predictive models without deep expertise in machine learning.

---

## ⚡ Key Capabilities

- 📂 **Seamless Data Ingestion** — Upload large CSV datasets effortlessly  
- 🧠 **Automated Preprocessing** — Missing value handling, encoding, and scaling  
- 📊 **Exploratory Data Analysis** — Descriptive statistics & insights  
- 📈 **Distribution Analysis** — KDE visualization & normalization  
- ⚙️ **Feature Engineering** — Skewness correction using PowerTransformer  
- 🤖 **Multi-Model Training** — Classification & regression pipelines  
- 🏆 **Model Leaderboard** — Compare performance across models  
- 🎯 **Hyperparameter Optimization** — GridSearchCV for best results  
- 📉 **Feature Importance Insights** — Understand key drivers  
- 💾 **Export Ready Outputs** — Download cleaned datasets & results  

---
## 🧠 System Workflow

```mermaid
flowchart TD
A[CSV Upload] --> B[Target Selection]
B --> C[EDA Report]
C --> D[Distribution Analysis]
D --> E[Feature Engineering]
E --> F[Run AutoML]
F --> G[Data Cleaning]
G --> H[Model Training]
H --> I[Hyperparameter Tuning]
I --> J[Feature Importance]
J --> K[Model Download]
