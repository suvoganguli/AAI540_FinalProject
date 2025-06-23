# AAI 540 – Machine Learning Operations Final Project  
**University of San Diego – Spring 2025**

## Project Overview
This repository contains the final project for AAI 540: Machine Learning Operations, developed as part of the M.S. in Applied Artificial Intelligence program at the University of San Diego.

## Project Goal
To design and implement a complete MLOps pipeline that predicts hospital readmissions using the Diabetes 130-US hospitals dataset. The pipeline spans data ingestion, feature engineering, model training, batch inference, model monitoring, and CI/CD automation using AWS SageMaker.

## Project Scope
- Clean and process raw healthcare data
- Store features in SageMaker Feature Store
- Train a Logistic Regression model using `StandardScaler` and one-hot encoding
- Evaluate model performance (recall, precision, accuracy)
- Perform batch inference and capture outputs
- Monitor prediction quality using SageMaker Model Monitor
- Build a CI/CD pipeline with SageMaker Pipelines and simulate retraining

## Team Members

| Name                | Email                        |
|---------------------|------------------------------|
| Muhammad Harris     | mharis@sandiego.edu          |
| Daniel Shifrin      | dshifrin@sandiego.edu        |
| Subhabrata Ganguli  | sganguli@sandiego.edu        |

## Pipeline Overview

This project follows a 7-stage modular pipeline, implemented across individual notebooks:

| Step | Notebook                                | Description |
|------|------------------------------------------|-------------|
| 01   | `01_data_and_featurestore.ipynb`         | Ingest and clean raw data; write to SageMaker Feature Store |
| 02   | `02_feature_engineering.ipynb`           | Apply one-hot encoding, scale features, and split data |
| 03   | `03_model_training.ipynb`                | Train logistic regression model with best hyperparameters |
| 04   | `04_evaluation_and_reporting.ipynb`      | Evaluate model performance and generate metrics |
| 05   | `05_monitoring_and_registry.ipynb`       | Define baseline constraints and prepare monitoring setup |
| 06   | `06_batch_inference.ipynb`               | Run SageMaker batch transform and save output for monitoring |
| 07   | `07_cicd_pipeline.ipynb`                 | Define and trigger full CI/CD pipeline including model quality monitoring |


## Project Management
We are using Asana to coordinate tasks, manage milestones, and track progress for this project.
[View our Asana Project Board](https://app.asana.com/1/952672460738672/project/1210280905126017/board/1210281001415249)


## Tools & Technologies
- Python 3.x
- AWS SageMaker (Feature Store, Pipelines, Model Monitor)
- Jupyter Notebooks
- Git & GitHub
- Pandas, Scikit-learn, Boto3

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and distribute this codebase for academic or non-commercial purposes, provided proper attribution is given to the authors. See the `LICENSE` file for full terms.
