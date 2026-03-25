# Probabilistic Forecasting of Adult G&A Occupied Beds (NHS England SitRep)

## Project Overview
This repository contains the codebase for my empirical study on forecasting daily bed occupancy within the National Health Service (NHS). The project addresses the critical operational challenge of managing urgent and emergency care (UEC) capacity by transitioning from classical deterministic forecasting to advanced probabilistic frameworks. 

The models are trained and evaluated on the NHS England UEC Daily Situation Reports (SitRep) for the Winter 2024–2025 period, focusing specifically on adult general and acute (G&A) occupied beds.

## Methodology
To rigorously evaluate forecasting performance, this project implements a progression of models, incorporating specific temporal geometry (cyclic topology encodings) to capture intra-week seasonality:

* **Classical Baseline (ARIMAX):** An `ARIMAX(7,1,1)` model leveraging sine and cosine transformations of the day-of-week as exogenous variables. This model achieved the lowest deterministic point error and yielded Ljung-Box p-values of 1.0, confirming the successful extraction of periodic signals.
* **Tree-based Approximations (LightGBM):** Evaluated for both point forecasting and quantile regression (minimising the asymmetric pinball loss) to construct dynamic 80% prediction intervals.
* **Distributional Forecasting (NGBoost):** Natural Gradient Boosting is employed to parameterise a full continuous distribution (Normal) for each time step. By optimising the Continuous Ranked Probability Score (CRPS) over a Riemannian manifold, the model provides operationally actionable uncertainty bounds for tail-risk management.

## Advanced Experiments & Interpretability
* **Validation Robustness:** Demonstrates the severe impact of Data Leakage by contrasting standard Random K-Fold cross-validation against a strict Time-Series (Rolling-Origin) protocol.
* **Distributional Calibration:** Applies Isotonic Regression as a post-hoc calibration step, significantly reducing the Expected Calibration Error (ECE) and improving empirical coverage.
* **Model Interpretability (SHAP):** Uses Shapley Additive Explanations to decode the non-linear interaction between recent momentum (autoregressive lags) and the cyclic topology.

## Repository Structure
* `data/` : Contains the raw NHS SitRep time series CSV file.
* `scripts/` : Contains the Python implementation for EDA, model training, SHAP analysis, and calibration experiments.
* `requirements.txt` : Lists all necessary dependencies to reproduce the environment.

## How to Run
To replicate the environment and run the models, install the required packages using:
`pip install -r requirements.txt`
