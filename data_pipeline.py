import pandas as pd
import numpy as np

def categorize_lead_time(days):
    if pd.isna(days) or days < 0:
        return 'Unknown'
    elif days <= 2:
        return 'Last Minute'
    elif days <= 7:
        return 'Short Lead'
    elif days <= 14:
        return 'Medium Lead'
    else:
        return 'Early Booking'

def derive_outcome(status):
    status_str = str(status).strip().title()
    if status_str == 'Used':
        return 'Completed'
    elif status_str in ['Cancelled', 'No-Show', 'Noshow']:
        return 'Not Completed'
    else:
        return 'Not Completed'

def calc_attendance_ratio(row):
    part = row['Participants']
    att = row['Actual_Attendance']
    if pd.isna(part) or part <= 0 or pd.isna(att):
        return 0.0
    return round(min(att / part, 1.0), 4)

def clean_and_transform_data(df_raw):
    """Clean and apply feature engineering on the facility reservations dataset."""
    df = df_raw.copy().drop_duplicates()

    # Standardize string categories
    categorical_cols = ['Facility_Type', 'User_Category', 'Usage_Status']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Handle numerical missing/invalid values
    if 'Participants' in df.columns:
        df['Participants'] = df['Participants'].apply(lambda x: np.nan if (pd.notna(x) and x < 0) else x)
        df['Participants'] = df['Participants'].fillna(df['Participants'].median())

    if 'Actual_Attendance' in df.columns:
        df['Actual_Attendance'] = df['Actual_Attendance'].apply(lambda x: np.nan if (pd.notna(x) and x < 0) else x)
        df['Actual_Attendance'] = df['Actual_Attendance'].fillna(0)

    # Feature Engineering
    df['Lead_Time_Category'] = df['Lead_Time_Days'].apply(categorize_lead_time)
    df['Reservation_Outcome'] = df['Usage_Status'].apply(derive_outcome)
    df['Attendance_Ratio'] = df.apply(calc_attendance_ratio, axis=1)

    return df
