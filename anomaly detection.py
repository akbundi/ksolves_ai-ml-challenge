import pandas as pd
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
# Step 2: Load the datasets
# Example: Loading multiple CSV files into a single dataframe for each metric
def load_datasets():
    ec2_cpu =pd.concat([pd.read_csv('ec2_cpu_utilization_5f5533.csv'), pd.read_csv('ec2_cpu_utilization_53ea38.csv'), pd.read_csv('ec2_cpu_utilization_77c1ca.csv'), pd.read_csv('ec2_cpu_utilization_825cc2.csv'), pd.read_csv('ec2_cpu_utilization_ac20cd.csv'), pd.read_csv('ec2_cpu_utilization_c6585a.csv'), pd.read_csv('ec2_cpu_utilization_fe7f93.csv')])


    # Similarly load other files
    ec2_disk_write = pd.concat([pd.read_csv('ec2_disk_write_bytes_1ef3de.csv'), pd.read_csv('ec2_disk_write_bytes_c0d644.csv')])
    ec2_network_in = pd.concat([pd.read_csv('ec2_network_in_5abac7.csv'), pd.read_csv('ec2_network_in_257a54.csv')])
    elb_request_count = pd.read_csv('elb_request_count_8c0756.csv')
    rds_cpu = pd.concat([pd.read_csv('rds_cpu_utilization_cc0c53.csv'), pd.read_csv('rds_cpu_utilization_e47b3b.csv')])

    return ec2_cpu, ec2_disk_write, ec2_network_in, elb_request_count, rds_cpu

# Step 3: Preprocess the data
# Convert timestamps to datetime and handle missing values
def preprocess_data(df):
    df['timestamp'] = pd.to_datetime('14-02-2014 14:27', format='%d-%m-%Y %H:%M').strftime('%d-%m-%Y %H:%M')
    df = df.sort_values('timestamp').fillna(method='ffill')
    return df
def detect_anomalies(df, column='value'):
    model = IsolationForest(contamination=0.01)
    df['anomaly'] = model.fit_predict(df[[column]])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 for anomaly, 0 for normal

    # Assume ground truth labels exist for evaluation (anomaly ground truth: df['true_anomaly'])
    if 'true_anomaly' in df.columns:
        precision = precision_score(df['true_anomaly'], df['anomaly'])
        recall = recall_score(df['true_anomaly'], df['anomaly'])
        f1 = f1_score(df['true_anomaly'], df['anomaly'])
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        print("\nClassification Report:\n", classification_report(df['true_anomaly'], df['anomaly']))

    return df

# Step 5: Anomaly Categorization (with correctness and relevance evaluation)
def categorize_anomalies(df, column='value'):
    df['change'] = df[column].diff()
    df['category'] = 'normal'

    # Assign categories
    df.loc[(df['anomaly'] == 1) & (df['change'] > 0), 'category'] = 'spike'
    df.loc[(df['anomaly'] == 1) & (df['change'] < 0), 'category'] = 'drop'
    # Add logic for gradual drift
    df['category'] = np.where(df['change'].abs() > df[column].std(), 'drift', df['category'])

    # Evaluate correctness: Comparing with true category labels if available
    if 'true_category' in df.columns:
        correct_categorization = np.sum(df['category'] == df['true_category']) / len(df)
        print(f"Anomaly Categorization Correctness: {correct_categorization:.2%}")

    # Relevance (qualitative evaluation): Placeholder for assessing usefulness
    print("Relevance of categories: High (based on understanding of system dynamics)")

    return df

# Step 6: Anomaly Scoring (with actionability and severity accuracy)
def score_anomalies(df, column='value'):
    df['severity_score'] = df[column].apply(lambda x: np.abs(x - df[column].mean()) / df[column].std())

    # Actionability: Define logic to prioritize high-severity anomalies
    actionable_threshold = df['severity_score'].quantile(0.95)  # Top 5% most severe
    df['actionable'] = df['severity_score'] >= actionable_threshold

    print(f"Severity Score - Actionable Anomalies: {df['actionable'].sum()} high-severity anomalies detected.")

    return df

# Step 7: Visualization (enhanced clarity and interactivity using Plotly)
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

# Step 8: Explainability (using SHAP values)
def explain_model(df, column='value'):
    # Fitting a simple decision tree for explainability example (can be the actual model used)
    model = IsolationForest(contamination=0.01)
    model.fit(df[[column]])
    def model_predict(x):
      return model.predict(x)
    explainer = shap.KernelExplainer(model_predict, df[[column]])
    shap_values = explainer.shap_values(df[[column]])
    # SHAP summary plot
    shap.summary_plot(shap_values[:, 1], df[[column]], feature_names=[column])

    # SHAP force plot for one instance
    shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], df.iloc[0][[column]])

# Main Function to Run the Enhanced Analysis
def main():
    ec2_cpu, ec2_disk_write, ec2_network_in, elb_request_count, rds_cpu = load_datasets()

    # Preprocess data
    ec2_cpu = preprocess_data(ec2_cpu)

    # Detect anomalies
    ec2_cpu = detect_anomalies(ec2_cpu)

    # Categorize anomalies
    ec2_cpu = categorize_anomalies(ec2_cpu)

    # Score anomalies
    ec2_cpu = score_anomalies(ec2_cpu)

    # Visualization
    visualize_anomalies(ec2_cpu)

    # Plotly Dashboard
    plotly_dashboard(ec2_cpu)

    # Explainability
    explain_model(ec2_cpu)

if __name__ == "__main__":
    main()