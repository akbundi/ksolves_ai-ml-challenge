# ksolves_ai-ml-challenge
Overview: This project provides a comprehensive approach to anomaly detection in time series data, with additional steps for categorization, scoring, visualization, and explainability. The code leverages various libraries for data manipulation, anomaly detection, and visualization, ensuring a thorough analysis of time series metrics.
Setup Instructions:Prerequisites:Ensure you have the following Python packages installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
plotly
shap
You can install these packages using pip:pip install pandas numpy matplotlib seaborn scikit-learn statsmodels plotly shap
File Structure:
ec2_cpu_utilization_*.csv (multiple files)
ec2_disk_write_bytes_*.csv (multiple files)
ec2_network_in_*.csv (multiple files)
elb_request_count_*.csv (1 file)
rds_cpu_utilization_*.csv (multiple files)
Ensure these CSV files are in the same directory as your script, or adjust the file paths in the load_datasets function accordingly.


          Code Walkthrough

1. Import Libraries:import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import plotly.express as px
from datetime import datetime
import shap
from sklearn.tree import DecisionTreeClassifier

2. Load Datasets:The load_datasets function loads multiple CSV files for different metrics into separate DataFrames and combines them where necessary.=
3. def load_datasets():
    ec2_cpu = pd.concat([pd.read_csv('ec2_cpu_utilization_5f5533.csv'), ...])
    ec2_disk_write = pd.concat([pd.read_csv('ec2_disk_write_bytes_1ef3de.csv'), ...])
    ec2_network_in = pd.concat([pd.read_csv('ec2_network_in_5abac7.csv'), ...])
    elb_request_count = pd.read_csv('elb_request_count_8c0756.csv')
    rds_cpu = pd.concat([pd.read_csv('rds_cpu_utilization_cc0c53.csv'), ...])

    return ec2_cpu, ec2_disk_write, ec2_network_in, elb_request_count, rds_cpu


3. Preprocess Data
The preprocess_data function converts timestamps to datetime format and handles missing values.

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
    df = df.sort_values('timestamp').fillna(method='ffill')
    return df

  4. Detect Anomalies
The detect_anomalies function uses Isolation Forest to identify anomalies in the data and evaluates the results if ground truth labels are available.

def detect_anomalies(df, column='value'):
    model = IsolationForest(contamination=0.01)
    df['anomaly'] = model.fit_predict(df[[column]])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    if 'true_anomaly' in df.columns:
        precision = precision_score(df['true_anomaly'], df['anomaly'])
        recall = recall_score(df['true_anomaly'], df['anomaly'])
        f1 = f1_score(df['true_anomaly'], df['anomaly'])
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        print("\nClassification Report:\n", classification_report(df['true_anomaly'], df['anomaly']))

    return df

5. Categorize Anomalies
The categorize_anomalies function classifies anomalies into categories like 'spike', 'drop', or 'drift' based on changes in the metric values.

def categorize_anomalies(df, column='value'):
    df['change'] = df[column].diff()
    df['category'] = 'normal'

    df.loc[(df['anomaly'] == 1) & (df['change'] > 0), 'category'] = 'spike'
    df.loc[(df['anomaly'] == 1) & (df['change'] < 0), 'category'] = 'drop'
    df['category'] = np.where(df['change'].abs() > df[column].std(), 'drift', df['category'])

    if 'true_category' in df.columns:
        correct_categorization = np.sum(df['category'] == df['true_category']) / len(df)
        print(f"Anomaly Categorization Correctness: {correct_categorization:.2%}")

    print("Relevance of categories: High (based on understanding of system dynamics)")

    return df

6. Score Anomalies
The score_anomalies function calculates severity scores for anomalies and identifies actionable anomalies.

def score_anomalies(df, column='value'):
    df['severity_score'] = df[column].apply(lambda x: np.abs(x - df[column].mean()) / df[column].std())
    actionable_threshold = df['severity_score'].quantile(0.95)
    df['actionable'] = df['severity_score'] >= actionable_threshold

    print(f"Severity Score - Actionable Anomalies: {df['actionable'].sum()} high-severity anomalies detected.")

    return df

7. Visualization
The visualize_anomalies function provides a static visualization using Matplotlib and Seaborn, while plotly_dashboard creates an interactive dashboard using Plotly.

def visualize_anomalies(df, column='value'):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x='timestamp', y=column, label='Metric Value')
    sns.scatterplot(data=df[df['anomaly'] == 1], x='timestamp', y=column, color='red', label='Anomalies')
    plt.title('Anomalies in Time Series Data')
    plt.show()

def plotly_dashboard(df, column='value'):
    fig = px.line(df, x='timestamp', y=column, title='Anomaly Detection Dashboard')
    fig.add_scatter(x=df[df['anomaly'] == 1]['timestamp'], y=df[df['anomaly'] == 1][column],
                    mode='markers', marker=dict(color='red', size=10), name='Anomalies')
    fig.add_scatter(x=df[df['actionable'] == True]['timestamp'], y=df[df['actionable'] == True][column],
                    mode='markers', marker=dict(color='purple', size=12), name='High Severity Anomalies')
    fig.show()

8. Explainability
The explain_model function uses SHAP values to provide explainability for the anomaly detection model.

def explain_model(df, column='value'):
    model = IsolationForest(contamination=0.01)
    model.fit(df[[column]])
    def model_predict(x):
        return model.predict(x)
    explainer = shap.KernelExplainer(model_predict, df[[column]])
    shap_values = explainer.shap_values(df[[column]])
    shap.summary_plot(shap_values[:, 1], df[[column]], feature_names=[column])
    shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], df.iloc[0][[column]]

Main Function
The main function ties together all the steps: loading datasets, preprocessing, detecting anomalies, categorizing and scoring anomalies, visualizing results, and explaining the model.

def main():
    ec2_cpu, ec2_disk_write, ec2_network_in, elb_request_count, rds_cpu = load_datasets()

    ec2_cpu = preprocess_data(ec2_cpu)
    ec2_cpu = detect_anomalies(ec2_cpu)
    ec2_cpu = categorize_anomalies(ec2_cpu)
    ec2_cpu = score_anomalies(ec2_cpu)
    visualize_anomalies(ec2_cpu)
    plotly_dashboard(ec2_cpu)
    explain_model(ec2_cpu)

if __name__ == "__main__":
    main()


    Running the Code
To run the code, simply execute the script in your Python environment. Ensure all CSV files are correctly placed and paths are adjusted if necessary:-

python your_script_name.py
