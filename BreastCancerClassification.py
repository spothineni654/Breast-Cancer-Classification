import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, auc

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['target'] = df['diagnosis'].map({'B': 0, 'M': 1})
    df.drop('diagnosis', axis=1, inplace=True)
    return df

def plot_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df, palette=["#1f77b4", "#ff7f0e"])
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.title('Distribution of Target Variable')
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Full Correlation Heatmap of Features')
    plt.show()

def detect_and_plot_outliers(df):
    lof = LocalOutlierFactor()
    X_score = lof.fit_predict(df.drop('target', axis=1))
    radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())
    plt.figure(figsize=(8, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color="k", s=3, label="Data Point")
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=1000 * radius, edgecolors="r", facecolors="none", label="Outlier Score")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.grid(True)
    plt.title('Outlier Detection using Local Outlier Factor (LOF)')
    plt.show()
    threshold_outlier = -2.5
    outlier_index = np.where(X_score < threshold_outlier)[0]
    return df.drop(outlier_index)

def apply_pca(df):
    x = df.drop('target', axis=1)
    x_standardized = (x - x.mean()) / x.std()
    pca = PCA().fit(x_standardized)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    principal_components = PCA(n_components=optimal_components).fit_transform(x_standardized)
    return principal_components, optimal_components, cumulative_variance_ratio

def plot_pca(cumulative_variance_ratio, optimal_components):
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(cumulative_variance_ratio), marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.annotate(f'Optimal Components: {optimal_components}', 
                 xy=(optimal_components, cumulative_variance_ratio[optimal_components-1]),
                 xytext=(optimal_components + 2, cumulative_variance_ratio[optimal_components-1] - 0.05),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    plt.grid(True)
    plt.show()

def train_models_and_plot_roc(principal_components, target):
    x_train, x_test, y_train, y_test = train_test_split(principal_components, target, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    results = []
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        }
        results.append(metrics)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    return pd.DataFrame(results)

def main():
    df = load_and_prepare_data('data.csv')
    plot_distribution(df)
    plot_heatmap(df)
    df = detect_and_plot_outliers(df)
    principal_components, optimal_components, cumulative_variance_ratio = apply_pca(df)
    plot_pca(cumulative_variance_ratio, optimal_components)
    results_df = train_models_and_plot_roc(principal_components, df['target'])
    print(results_df)
    results_df.set_index("Model").plot(kind="bar", rot=45)
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    main()
