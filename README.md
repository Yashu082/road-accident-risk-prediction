# Road Accident Risk Prediction

A machine learning project that predicts road accident risk based on various road and environmental factors using regression analysis.

## 🚀 Overview

This project implements a comprehensive road accident risk prediction system that analyzes multiple features including road conditions, weather, lighting, time of day, and historical accident data to provide accurate risk assessments. The system uses a Random Forest Regressor model trained on a dataset of over 500,000 road segments.

## 📊 Dataset

The project uses a comprehensive dataset with the following features:

### **Road Characteristics**
- `road_type`: Type of road (urban, rural, highway)
- `num_lanes`: Number of lanes (1-10)
- `curvature`: Road curvature index (0.0-1.0)
- `speed_limit`: Speed limit in km/h (25-70)
- `road_signs_present`: Boolean indicating presence of road signs
- `public_road`: Boolean indicating if it's a public road

### **Environmental Conditions**
- `weather`: Weather conditions (clear, foggy, rainy)
- `lighting`: Lighting conditions (daylight, dim, night)
- `time_of_day`: Time period (morning, afternoon, evening)

### **Contextual Factors**
- `holiday`: Boolean indicating if it's a holiday
- `school_season`: Boolean indicating if it's school season
- `num_reported_accidents`: Historical accident count

### **Target Variable**
- `accident_risk`: Continuous risk score (0.0-1.0)

## 🛠️ Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Web application framework
- **joblib**: Model serialization

## 📁 Project Structure

```
road-accident-risk-prediction/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── train.csv                   # Training dataset (517,754 records)
├── road_accident_risk_prediction.ipynb  # Jupyter notebook with model training
├── streamlit_app.py            # Streamlit web application
└── model.pkl                   # Trained model (to be generated)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd road-accident-risk-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Model Training

Open the Jupyter notebook to train the model:

```bash
jupyter notebook road_accident_risk_prediction.ipynb
```

The notebook includes:
- Data preprocessing and encoding
- Feature engineering
- Random Forest Regressor training
- Model evaluation with comprehensive metrics

#### 2. Web Application

Run the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The web app provides:
- Interactive form for input parameters
- Real-time risk prediction
- Model upload functionality
- Feature visualization
- CSV export capabilities

## 📈 Model Performance

The trained Random Forest Regressor achieves the following performance metrics:

- **R² Score**: 0.8729 (87.29% variance explained)
- **Adjusted R²**: 0.8729
- **RMSE**: 0.0592
- **MAE**: 0.0460
- **MSE**: 0.0035

## 🤖 Model Details

### Algorithm
- **Random Forest Regressor** with 300 trees
- Uses all CPU cores for parallel processing
- Handles both numerical and categorical features effectively

### Feature Engineering
- One-hot encoding for categorical variables
- Boolean conversion for binary features
- Comprehensive feature preprocessing pipeline

### Training Parameters
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

## 🌐 Web Application Features

### Input Interface
- **Road Parameters**: Type, lanes, curvature, speed limit
- **Environmental Conditions**: Weather, lighting, time of day
- **Contextual Factors**: Holiday, school season, historical accidents

### Output
- Real-time risk prediction (0.0-1.0 scale)
- Processed feature visualization
- CSV download for further analysis

### Model Management
- Upload custom trained models
- Automatic model loading
- Fallback handling for missing models

## 🔧 Configuration

### Model Path
The application expects a trained model at `model.pkl`. You can:
1. Train the model using the provided notebook
2. Upload a pre-trained model through the web interface
3. Place your own `model.pkl` file in the project root

### Feature Columns
The model expects 20 preprocessed features:
- Numerical: num_lanes, curvature, speed_limit, num_reported_accidents
- Boolean: road_signs_present, public_road, holiday, school_season
- One-hot encoded: road_type (3), lighting (3), weather (3), time_of_day (3)

## 📝 Development Notes

### Data Processing
- Missing values: None detected in the dataset
- Outlier handling: Built-in Random Forest robustness
- Feature scaling: Not required for tree-based models

### Model Deployment
The Streamlit application provides a production-ready interface with:
- Error handling and validation
- User-friendly input forms
- Real-time predictions
- Export functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Future Enhancements

- [ ] Additional model algorithms (XGBoost, LightGBM)
- [ ] Geospatial mapping integration
- [ ] Real-time weather API integration
- [ ] Mobile application development
- [ ] Advanced feature engineering
- [ ] Time series analysis for temporal patterns

## 📞 Contact

For questions or suggestions regarding this project, please open an issue in the repository.

---

**Note**: Ensure you have sufficient computational resources when training the model on the full dataset, as it contains over 500,000 records with multiple features.
