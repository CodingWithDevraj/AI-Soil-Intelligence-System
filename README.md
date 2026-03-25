# AI Soil Intelligence System

An end-to-end machine learning system that analyzes soil and environmental parameters to recommend the most suitable crop, identify soil type, and suggest optimal fertilizer usage. This project demonstrates a complete data science and production pipeline, from exploratory data analysis to model deployment through an interactive web application.

---

## Live Demo

[https://ai-soil-intelligence-system-3tf5bfmu8ejkdaqdqd9r6w.streamlit.app/](https://ai-soil-intelligence-system-3tf5bfmu8ejkdaqdqd9r6w.streamlit.app/)

---

## Overview

This system is designed to assist farmers and agricultural stakeholders in making data-driven decisions. By leveraging machine learning models trained on agricultural datasets, the application provides real-time recommendations based on soil nutrients, weather conditions, and soil chemistry.

The project integrates multiple datasets and builds independent yet coordinated models for:

* Crop recommendation
* Soil type classification
* Fertilizer recommendation

---

## Key Features

* Multi-model architecture for different prediction tasks
* Data-driven crop recommendation using environmental and soil inputs
* Soil classification based on chemical and physical properties
* Fertilizer recommendation to optimize yield
* Interactive web interface for real-time predictions
* Production-ready pipeline with modular code structure

---

## Tech Stack

Backend:

* Python
* Flask (API layer)

Machine Learning:

* Scikit-learn (Random Forest)
* Pandas, NumPy

Frontend:

* Streamlit

Deployment:

* Streamlit Cloud

Version Control:

* Git and Git LFS

---

## Machine Learning Pipeline

1. Data Collection
   Two datasets were used:

   * Crop Recommendation Dataset
   * Fertilizer Recommendation Dataset

2. Exploratory Data Analysis

   * Distribution analysis of features
   * Correlation analysis
   * Outlier detection
   * Class balance verification

3. Feature Engineering

   * Standardized feature naming across datasets
   * Domain-based feature alignment (N, P, K, temperature, humidity, pH)
   * Inclusion of soil chemistry features such as organic carbon and electrical conductivity

4. Model Training
   Three separate models were trained:

   * Crop Model
   * Soil Type Model
   * Fertilizer Model

   Techniques used:

   * Train-test split
   * Random Forest Classifier
   * Class imbalance handling using weighted models

5. Evaluation

   * Accuracy and classification reports
   * Model tuning for improved performance

6. Deployment

   * Flask API for serving models (local)
   * Streamlit application for user interaction
   * Cloud deployment for public access

---

## Model Performance

* Crop Recommendation Model
  High accuracy (~99%) due to well-structured dataset and strong feature relationships

* Soil Type Classification Model
  Improved accuracy using additional soil chemistry features such as pH, organic carbon, and electrical conductivity

* Fertilizer Recommendation Model
  Balanced performance after handling class imbalance using weighted Random Forest

---

## Project Structure

```
soil-ai-ml-project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── eda notebooks
│
├── src/
│   ├── train_model.py
│
├── models/
│   ├── crop_model.pkl
│   ├── soil_model.pkl
│   ├── fertilizer_model.pkl
│
├── app/
│   ├── app.py
│
├── streamlit_app/
│   ├── app.py
│
├── requirements.txt
├── README.md
```

---

## How to Run Locally

1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/AI-Soil-Intelligence-System.git
cd AI-Soil-Intelligence-System
```

2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run Streamlit application

```
cd streamlit_app
streamlit run app.py
```

---

## API (Optional Backend)

To run Flask API locally:

```
cd app
python app.py
```

Endpoint:

```
POST /predict
```
## Screenshots

* Inputs 
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/627c57ed-a72e-4e82-b3bd-69921619050d" />


* Predictions
<img width="1917" height="1079" alt="image" src="https://github.com/user-attachments/assets/480e360a-cf22-4e8d-8eb1-bacd65b2a979" />



---

## Use Case

This system can be used by:

* Farmers for selecting suitable crops based on soil conditions
* Agricultural consultants for fertilizer planning
* AgriTech platforms for intelligent recommendations
* Researchers working on precision agriculture

---

## Highlights

* End-to-end machine learning project
* Real-world agricultural use case
* Multi-dataset integration
* Production-ready architecture
* Deployed and accessible live application

---

## Author

**Devraj Choudhary**

B.Tech – Gurukul Kangri University
Haridwar, Uttarakhand

Email: [devrajror366@gmail.com](mailto:devrajror366@gmail.com)

LinkedIn
[https://www.linkedin.com/in/devraj-choudhary-3889412bb/](https://www.linkedin.com/in/devraj-choudhary-3889412bb/)
