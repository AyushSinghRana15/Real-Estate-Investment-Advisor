# 🏠 Real Estate Investment Advisor

[![Streamlit](https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-0C85EF?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-478FBF?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Git LFS](https://img.shields.io/badge/Git_LFS-EA4B26?style=for-the-badge&logo=git-lfs&logoColor=white)](https://git-lfs.github.com/)

**ML-powered property evaluation app** that predicts investment potential and 5-year ROI using XGBoost models trained on 250K+ Indian real estate records. Enter property details to get data-driven recommendations with 99.9% accuracy.

## ✨ Features

- **Dual XGBoost Models**: Classifier (AUC=1.000, Precision=0.973) + Regressor (R²=0.999)
- **28 Engineered Features**: Density scores, age_score=1/(1+age), amenity_score, floor ratios
- **Production MLOps**: MLflow experiment tracking, registered models (v1), joblib artifacts
- **Interactive Streamlit UI**: Real-time predictions with confidence scores and ROI%
- **Robust Preprocessing**: RobustScaler, OneHotEncoder(drop='first'), target encoding

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| **Classifier** | Accuracy | 0.999 |
| | Precision | 0.973 |
| | Recall | 0.995 |
| | ROC-AUC | 1.000 |
| **Regressor** | RMSE | 0.001 |
| | MAE | 0.001 |
| | R² | 0.999 |

## 🏗️ Tech Stack

Data (250K rows) → Feature Engineering (28 feats) → Preprocessing Pipeline  <br>
↓<br>
XGBoost Classifier + Regressor ← MLflow Tracking (SQLite) <br>
↓<br>
Streamlit App ← joblib(scalers/encoders) + MLflow Models v1


**Preprocessing**: RobustScaler(10 numerics) + OneHotEncoder(7→11 cats) + TargetEncoder(Locality)

## 🚀 Quick Start

### 1. Clone with LFS
git lfs install<br>
git clone https://github.com/AyushSinghRana15/Real-Estate-Investment-Advisor.git <br>
cd “Real-Estate-Investment-Advisor”

### 2. Install Dependencies

pip install -r requirements.txt<br>
or<br>
pip install streamlit mlflow xgboost scikit-learn pandas numpy joblib


### 3. Run Streamlit App

streamlit run app.py


**Demo**: https://share.streamlit.io/AyushSinghRana15/real-estate-investment-advisor/app.py

## 📁 Project Structure

├── app.py                         # Streamlit dashboard  <br>
├── num_scaler.pkl                 # RobustScaler (fitted) Git LFS <br>
├── cat_ohe.pkl                    # OneHotEncoder (fitted) Git LFS<br>
├── target_encoder_locality.pkl    # TargetEncoder (fitted) Git LFS<br>
├── mlflow.db                      # Experiment tracking Git LFS<br>
├── lookup_values.csv              # UI dropdown values Git LFS <br>
├── training_notebook.ipynb        # Full training pipeline <br>
├── .gitattributes                 # LFS patterns <br>
├── .gitignore                     # Python/ML cleanup <br>
└── requirements.txt               # Dependencies


## 🔬 Key Innovations

### Feature Engineering (28 features)
Density scores <br>
school_density_score = Nearby_Schools / (Size_in_SqFt / 1000) hospital_density_score = Nearby_Hospitals / (Size_in_SqFt / 1000)<br>
Positional & quality metrics<br>
floor_position_ratio = Floor_No / Total_Floors age_score = 1 / (1 + Age_of_Property) amenity_score = (Parking==“Yes”) + (Security==“Yes”) + (Furnished!=“Unfurnished”)


### Preprocessing Pipeline

Raw (25 feats) → RobustScaler(10 nums) | OHE(7→11 cats) | TargetEnc(Locality) → 28 feats


## 🛠️ MLflow Experiments

mlflow ui  # View at http://localhost:5000


**Registered Models**:
- `RealEstate_GoodInvestment_Classifier` (v1)
- `RealEstate_FuturePrice_Regressor` (v1)

## 📈 Usage Example

1. **Input**: 3BHK, 1200sqft, ₹100L, Bangalore (Koramangala), 8yrs old
2. **Output**:
Investment Decision: Good Investment (92.3% confidence)<br> Estimated Price (5Y): ₹178.5 Lakhs <br>Expected 5Y ROI: +78.5%

## 🔗 Key Files Explained

| File | Purpose | Size |
|------|---------|------|
| `app.py` | Streamlit inference app | 2KB |
| `cat_ohe.pkl` | OneHotEncoder (7 cats → 11 dummies) | 15MB |
| `num_scaler.pkl` | RobustScaler (10 numerics) | 2MB |
| `mlflow.db` | Experiment tracking + metrics | 42MB |
| `lookup_values.csv` | Valid dropdown options | 1MB |

## 📊 Business Impact

- **Investment Screening**: Filter 1000s of listings in seconds
- **ROI Forecasting**: 5-year price appreciation with 99.9% R²
- **Risk Assessment**: Confidence scores prevent bad investments
- **Scalable**: Handles new localities via target encoding

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built during Labmentix AI/ML Internship (RIT Roorkee B.Tech CSE AI/ML). Special thanks to the open-source community!

---

**⭐ Star this repo if it helps your ML journey!**

