import numpy as np
import pandas as pd
from fairlearn.metrics import (
    equalized_odds_difference,
    demographic_parity_difference,
    equalized_odds_ratio,
)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Load and preprocess data
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Handle missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['Clinical.M.Stage'] = df['Clinical.M.Stage'].fillna(0)
    df = df.dropna(subset=['gender', 'Overall.Stage', 'deadstatus.event'])

    # Convert categorical features
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Overall.Stage', 'Histology'], dummy_na=False)

    # Ensure deadstatus.event is binary with correct orientation
    df['deadstatus.event'] = df['deadstatus.event'].astype(int)

    return df[['gender', 'age', 'deadstatus.event'] +
              [c for c in df.columns if 'Overall.Stage' in c or 'Histology' in c]]


# Prepare data with SMOTE oversampling
def prepare_data(df):
    X = df.drop('deadstatus.event', axis=1)
    y = df['deadstatus.event']

    # Split before resampling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # Apply SMOTE only to training data
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    return X_res, X_test, y_res, y_test


# Train model with correct class orientation
def train_model(X_train, y_train):
    # Explicitly specify class weight for deadstatus.event=1
    model = LogisticRegression(
        max_iter=1000,
        class_weight={1: 0.7, 0: 0.3},
        # Mortality prediction (class1) is more critical than survival prediction (class2)
        solver='liblinear',
        random_state=42,
        penalty='l2',
        C=0.1
    )
    model.fit(X_train, y_train)
    return model


# Calculate fairness metrics
def calculate_fairness(y_true, y_pred, sensitive_feature):
    groups = ['male' if g == 0 else 'female' for g in sensitive_feature]

    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=groups)
    eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=groups)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=groups)

    return eo_diff, eo_ratio, dp_diff


def demographic_parity(df):
    # Demographic Parity
    df['gender'] = df['gender'].map({0: 'Male', 1: 'Female'})
    mortality_rate = df.groupby('gender')['deadstatus.event'].mean().to_frame('Mortality Rate')
    male_mortality_rate = mortality_rate.loc['Male', 'Mortality Rate']
    female_mortality_rate = mortality_rate.loc['Female', 'Mortality Rate']
    percentage_difference = ((male_mortality_rate - female_mortality_rate) / female_mortality_rate) * 100
    if male_mortality_rate > female_mortality_rate:
        gender = 'Male'
    else:
        gender = 'Female'
    return mortality_rate, gender, percentage_difference


def main():
    csv_path = 'test_data_radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'
    # Load and preprocess data
    df = load_and_preprocess(csv_path)

    # Prepare balanced dataset
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create test_df containing test data and predictions
    test_df = X_test.copy()
    test_df['prediction'] = y_pred  # Add predictions as new column
    test_df['deadstatus.event'] = y_test  # Ensure true labels are present

    # Calculate fairness metrics
    eo_diff, eo_ratio, dp_diff = calculate_fairness(y_test, y_pred, X_test['gender'])

    # Calculate Demographic Parity
    mrt_rate, gender, percentage_difference = demographic_parity(df)

    # Mortality Rate per gender
    print('**Mortality Rate**')
    print(mrt_rate)

    print('**Fairness Metrics**')
    # Equalized Odds Difference
    print(
        f"A {eo_diff * 100:.2f}% disparity in error rates between gender groups is detected. (Equalized Odds Difference: {eo_diff:.4f})")
    print(
        'Both genders receive equally reliable predictions. The <5% required difference for medical models as outlined by REALM is met.') if eo_diff <= 0.05 else (
        print('The <5% required difference for medical models as outlined by REALM is not met.'))
    print(f'{"=" * 40}')

    # Equalized Odds Ratio
    print(
        f"A parity of {eo_ratio * 100:.2f}% in error rate rations is detected between groups. (Equalized Odds Ratio: {eo_ratio:.4f})")
    if 1.0 >= eo_ratio >= 0.9: print('A nearly perfect alignment with the ideal ratio of 1.0 is shown.')
    print(f'{"=" * 40}')

    # Demographic Parity
    print(f'{gender}s have a {abs(percentage_difference):.2f}% greater chance of death')
    print(
        f'A {dp_diff * 100:.2f}% in positive prediction rates (mortality) is found. (Demographic Parity Difference: {dp_diff:.4f})')
    print('Within demographic parity acceptable range of <3%, per REALM guidelines') if dp_diff <= 0.03 else (
        print('Not within demographic parity acceptable range of <3%, per REALM guidelines'))

    # Feature importance analysis with correct interpretation
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'odds_ratio': np.exp(model.coef_[0])  # Convert to odds ratios
    }).sort_values('odds_ratio', ascending=False)

    print(f"\n{'=' * 40}\nTop 5 Predictive Features (Odds Ratios):\n{'=' * 40}")
    print(coef_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
