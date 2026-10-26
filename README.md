# Campus Facility Reservation Monitoring System

## System Overview
This application is a decision-support system built with Streamlit to monitor campus facility reservations and predict cancellation or no-show risks using machine learning.

## Objectives & SAD Scope
- **Predict Risk**: Identify reservations with high probability of cancellation or no-show.
- **Facility Dashboard**: Monitor facility utilization rates and review high-risk bookings.
- **Decision Support**: Provide actionable intelligence for facility administrators.

## Application Structure
- `app.py`: Main Streamlit app containing navigation and modules.
- `requirements.txt`: Python package dependencies.
- `facility_reservations.csv`: Dataset for validation and exploratory data analysis.
- `models/trained_model.joblib`: Exported machine learning model pipeline.
- `assets/use_case_diagram.png`: System Use Case Diagram.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
