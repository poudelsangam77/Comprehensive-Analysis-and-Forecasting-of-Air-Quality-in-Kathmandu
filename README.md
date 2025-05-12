#  Comprehensive Analysis and Forecasting of Air Quality in Kathmandu

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://kathmanduairqualityforecasting.streamlit.app/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An end-to-end data science project that analyzes historical air quality trends in Kathmandu, predicts AQI using machine learning, and provides real-time forecasting using Facebook Prophet — all wrapped in an interactive Streamlit dashboard.

> 🔗 **Live App:** [kathmanduairqualityforecasting.streamlit.app](https://kathmanduairqualityforecasting.streamlit.app/)

---

##  Table of Contents

- [ Table of Contents](#-table-of-contents)
- [ Project Overview](#-project-overview)
- [ Objectives](#-objectives)
- [ Features](#️-features)
- [ Methodology](#-methodology)
- [ Models Used](#-models-used)
- [ Dashboard Preview](#-dashboard-preview)
- [ Technologies Used](#️-technologies-used)
- [ Folder Structure](#-folder-structure)
- [ Run Locally](#-run-locally)
- [ Results](#-results)
- [ References](#-references)

---

##  Project Overview

Air pollution in Kathmandu has become a growing public health concern. This project provides a comprehensive analysis and predictive framework for monitoring and forecasting the **Air Quality Index (AQI)** using real-world environmental and meteorological data.

Using statistical modeling and machine learning, we built:
- A **predictive AQI model** based on environmental inputs.
- A **time-series forecasting model** for future AQI estimation.
- An interactive **Streamlit dashboard** for visualization, prediction, and forecasting.

---

##  Objectives

- Analyze historical AQI and weather data of Kathmandu.
- Forecast short-term AQI trends (24, 48, 72 hours).
- Develop a Random Forest-based model for real-time AQI prediction.
- Deploy an interactive web dashboard for public and policymaker use.

---

##  Features

-  **Visualize:** Interactive charts (line, bar, heatmap, pie) showing AQI trends, statistics, and category breakdowns.
-  **Predict:** Input current environmental parameters to get predicted AQI and health advisories.
-  **Forecast:** Generate 24/48/72-hour AQI forecasts using Facebook Prophet.
-  **Live Dashboard:** Deployed and accessible online.

---

##  Methodology

1. **Data Collection:** Open-Meteo API + government air quality datasets.
2. **Preprocessing:** Handled missing values, normalized features, converted time series.
3. **Exploratory Analysis:** Visualized trends, correlations, and seasonal effects.
4. **Modeling:**
   - Random Forest Regressor for real-time AQI prediction.
   - Facebook Prophet for time-series forecasting.
5. **Deployment:** Streamlit Cloud

---

##  Models Used

| Model                   | Purpose                                  | Metric (R² Score) |
|-------------------------|------------------------------------------|-------------------|
| Random Forest Regressor | Predict AQI from environment inputs     | 0.91              |
| Facebook Prophet        | Forecast AQI (24–72 hrs)                | N/A (Time series) |

---

##  Dashboard Preview

![Dashboard Preview](https://i.imgur.com/w9qgmgO.jpeg)

---

##  Technologies Used

- **Python**
- **Streamlit** (Dashboard)
- **Facebook Prophet** (Time-series forecasting)
- **Scikit-learn** (Machine Learning)
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Plotly** (EDA & Visualization)

---

##  Sample Folder Structure

```bash
├── Dashboard/
│   ├── app.py                   # Main Streamlit app
│   ├── models/
│   │   └── model.pkl            # Trained Random Forest model
│   ├── data/
│   │   └── air_quality_data.csv # Historical dataset
│   └── models/
│       └── prophet_model.py     # Forecasting functions
```

---

##  Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/poudelsangam77/Comprehensive-Analysis-and-Forecasting-of-Air-Quality-in-Kathmandu.git
cd kathmandu-air-quality-forecasting/Dashboard
```

### 2. Create virtual environment & activate
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

##  Results

- AQI prediction achieved **R² = 0.91** with Random Forest Regressor.
- Seasonal trends and daily pollution cycles clearly visualized.
- Streamlit dashboard enabled public-friendly access to air quality insights and forecasts.

---

##  References

- [Facebook Prophet Docs](https://facebook.github.io/prophet/docs/quick_start.html)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api)
- [OpenAQ API](https://docs.openaq.org/)

---

>  **Developed by**  
> Sangam Paudel · Saroj Rawal · Subesh Yadav  
>  Department of Electronics and Computer Engineering, Pulchowk Campus  
>  Tribhuvan University, Nepal
