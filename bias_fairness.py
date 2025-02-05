import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    equalized_odds_difference,
    demographic_parity_difference,
    equalized_odds_ratio,
)
from imblearn.over_sampling import SMOTE
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

    # Select features of interest
    cols = ['gender', 'age', 'deadstatus.event'] + [c for c in df.columns if 'Overall.Stage' in c or 'Histology' in c]
    return df[cols]


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
        class_weight={1: 0.7, 0: 0.3},  # Mortality prediction is more critical
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
    gender = 'Male' if male_mortality_rate > female_mortality_rate else 'Female'
    return mortality_rate, gender, percentage_difference


def plot_group_confusion_matrices(y_true, y_pred, sensitive_feature):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate prevalence for each group
    male_mask = (sensitive_feature == 0)
    female_mask = (sensitive_feature == 1)
    male_prev = male_mask.mean() * 100
    female_prev = female_mask.mean() * 100

    # Plot confusion matrix for male group
    male_cm = confusion_matrix(y_true[male_mask], y_pred[male_mask])
    disp1 = ConfusionMatrixDisplay(confusion_matrix=male_cm)
    disp1.plot(ax=ax1, cmap='viridis')
    ax1.set_title(f'Male Group\nPrevalence: {male_prev:.1f}%')

    # Plot confusion matrix for female group
    female_cm = confusion_matrix(y_true[female_mask], y_pred[female_mask])
    disp2 = ConfusionMatrixDisplay(confusion_matrix=female_cm)
    disp2.plot(ax=ax2, cmap='viridis')
    ax2.set_title(f'Female Group\nPrevalence: {female_prev:.1f}%')

    plt.tight_layout()
    return fig


def generate_pdf_report(metrics, figures, output_path='report.pdf'):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=12,
        alignment=1
    )

    story.append(Paragraph("Medical Model Fairness Analysis Report", title_style))

    # Key Metrics Table
    metrics_data = [
        ["Metric", "Value", "REALM Guideline"],
        ["Equalized Odds Difference", f"{metrics['eo_diff'] * 100:.2f}%", "<5%"],
        ["Demographic Parity", f"{metrics['dp_diff'] * 100:.2f}%", "<3%"],
        ["Mortality Rate Difference", f"{metrics['mortality_diff']:.2f}%", "N/A"]
    ]

    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.25 * inch))

    # Add Figures
    for fig in figures:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
            story.append(Image(tmpfile.name, width=6 * inch, height=4 * inch))
            story.append(Spacer(1, 0.25 * inch))

    # Feature Importance
    story.append(Paragraph("Top Predictive Features:", styles['Heading2']))
    feature_data = [["Feature", "Odds Ratio"]] + metrics['feature_importance']
    feature_table = Table(feature_data)
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    story.append(feature_table)

    doc.build(story)


def main():
    csv_path = 'test_data_radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'

    # Load and preprocess data
    df = load_and_preprocess(csv_path)

    # Prepare balanced dataset
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train model and predict
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
        f"A {eo_diff * 100:.2f}% disparity in error rates between gender groups is detected (Equalized Odds Difference: {eo_diff:.4f}).")
    if eo_diff <= 0.05:
        print(
            "Both genders receive equally reliable predictions. The <5% required difference for medical models as outlined by REALM is met.")
    else:
        print("The <5% required difference for medical models as outlined by REALM is not met.")
    print('=' * 40)

    # Equalized Odds Ratio
    print(
        f"A parity of {eo_ratio * 100:.2f}% in error rate ratios is detected between groups (Equalized Odds Ratio: {eo_ratio:.4f}).")
    if 1.0 >= eo_ratio >= 0.9:
        print("A nearly perfect alignment with the ideal ratio of 1.0 is shown.")
    print('=' * 40)

    # Demographic Parity
    print(f"{gender}s have a {abs(percentage_difference):.2f}% greater chance of death.")
    print(
        f"A {dp_diff * 100:.2f}% difference in positive prediction rates (mortality) is found (Demographic Parity Difference: {dp_diff:.4f}).")
    if dp_diff <= 0.03:
        print("Within the acceptable demographic parity range (<3%), per REALM guidelines.")
    else:
        print("Not within the acceptable demographic parity range (<3%), per REALM guidelines.")

    # Print confusion matrix (text output)
    cm = confusion_matrix(y_test, y_pred)
    print('\n**Confusion Matrix**')
    print(cm)

    # Also plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    plot_group_confusion_matrices(y_test, y_pred, X_test['gender'])

    # Feature importance analysis with correct interpretation
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'odds_ratio': np.exp(model.coef_[0])  # Convert coefficients to odds ratios
    }).sort_values('odds_ratio', ascending=False)

    print(f"\n{'=' * 40}\nTop 5 Predictive Features (Odds Ratios):\n{'=' * 40}")
    print(coef_df.head(5).to_string(index=False))

    # Collect metrics for PDF report
    report_metrics = {
        'eo_diff': eo_diff,
        'dp_diff': dp_diff,
        'mortality_diff': percentage_difference,
        'feature_importance': coef_df.head(5).values.tolist()
    }

    # Generate figures for PDF
    figures = []

    # Save confusion matrices to figures list
    fig = plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Overall Confusion Matrix')
    figures.append(fig)
    plt.close(fig)

    fig = plot_group_confusion_matrices(y_test, y_pred, X_test['gender'])
    figures.append(fig)
    plt.close(fig)

    # Generate PDF
    generate_pdf_report(report_metrics, figures)

    print(f"\nPDF report generated: fairness_report.pdf")


if __name__ == "__main__":
    main()
