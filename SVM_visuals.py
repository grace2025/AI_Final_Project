""" Visualize SVM """
import matplotlib.pyplot as plt
from SVM import run_svm
import numpy as np 
import seaborn as sns
import pandas as pd

def prep_plot_data():
    
    results, pred_genres_dict, classification_report = run_svm(report=True, return_data=True)
    y_test = results['y_test']
    y_pred = results['y_pred']
    genres = results['genres']

    return y_test, y_pred, genres, classification_report


def create_heatmap(genres, classification_report):

    plt.figure(figsize=(12, 8))
    
    precision_scores = [classification_report[genre]['precision'] for genre in genres]
    recall_scores = [classification_report[genre]['recall'] for genre in genres]
    f1_scores = [classification_report[genre]['f1-score'] for genre in genres]
    support_values = [classification_report[genre]['support'] for genre in genres]
    
    metrics_df = pd.DataFrame({
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores,
        'Support': support_values
    }, index=genres)
    
    heatmap_data = metrics_df[['Precision', 'Recall', 'F1-Score']].T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Score'})
    plt.title('Genre Performance Metrics')
    plt.xlabel('Genres')
    plt.ylabel('Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('matrix_svm_cr.png')
    plt.show()

    
    
def create_barchart(y_test, y_pred, genres):

    plt.figure(figsize=(14, 6))
    
    actual_freq = np.sum(y_test, axis=0)
    pred_freq = np.sum(y_pred, axis=0)
    
    x = np.arange(len(genres))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, actual_freq, width, label='Actual', alpha=0.8, color='lightblue')
    bars2 = plt.bar(x + width/2, pred_freq, width, label='Predicted', alpha=0.8, color='lightpink')
    
    plt.xlabel('Genres')
    plt.ylabel('Frequency')
    plt.title('Genre Frequency: Actual vs Predicted')
    plt.xticks(x, genres, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('barchart_svm_cr.png')
    plt.show()


def main():

    data = prep_plot_data()
    create_barchart(data[0],data[1],data[2])
    create_heatmap(data[2], data[3])

if __name__ == "__main__":
    main()