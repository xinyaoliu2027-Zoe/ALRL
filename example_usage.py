from improved_lifelong_regression import AdvancedLifelongRegressionModel
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

# The single source of truth for dataset loading lives in
# forgetting_prevention_comparison.load_datasets — it does the
# 80/20 train/test split with StandardScaler fit on train only.
# We re-export it here so this demo script and the main runner share
# exactly the same data pipeline.
from forgetting_prevention_comparison import load_datasets  # noqa: F401


def _legacy_load_datasets_DO_NOT_USE():
    """Kept only for reference; the real load_datasets is imported above."""
    datasets = {}

    # 1. California Housing dataset
    california = fetch_california_housing()
    X_cal = california.data
    y_cal = california.target
    # Standardize features and target
    scaler_X = StandardScaler()
    X_cal = scaler_X.fit_transform(X_cal)
    scaler_y = StandardScaler()
    y_cal = scaler_y.fit_transform(y_cal.reshape(-1, 1)).ravel()

    datasets['california'] = {
        'X': X_cal,
        'y': y_cal,
        'description': "California Housing (8 features)",
        'feature_names': california.feature_names
    }

    # 2. Diabetes dataset
    print("Loading Diabetes dataset")
    diabetes = load_diabetes()
    X_dia = diabetes.data
    y_dia = diabetes.target

    # Standardize
    scaler_X = StandardScaler()
    X_dia = scaler_X.fit_transform(X_dia)
    scaler_y = StandardScaler()
    y_dia = scaler_y.fit_transform(y_dia.reshape(-1, 1)).ravel()

    datasets['diabetes'] = {
        'X': X_dia,
        'y': y_dia,
        'description': "Diabetes Progression Prediction (10 features)",
        'feature_names': diabetes.feature_names
    }

    # 3. Wine dataset (classification dataset used for regression)
    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target.astype(float)  # Convert classification labels to float

    # Standardize
    scaler_X = StandardScaler()
    X_wine = scaler_X.fit_transform(X_wine)
    scaler_y = StandardScaler()
    y_wine = scaler_y.fit_transform(y_wine.reshape(-1, 1)).ravel()

    datasets['wine'] = {
        'X': X_wine,
        'y': y_wine,
        'description': "Wine Quality (13 features)",
        'feature_names': wine.feature_names
    }

    # 4. Breast Cancer dataset (classification dataset used for regression)
    cancer = load_breast_cancer()
    X_can = cancer.data
    y_can = cancer.target.astype(float)

    # Standardize
    scaler_X = StandardScaler()
    X_can = scaler_X.fit_transform(X_can)
    scaler_y = StandardScaler()
    y_can = scaler_y.fit_transform(y_can.reshape(-1, 1)).ravel()

    datasets['cancer'] = {
        'X': X_can,
        'y': y_can,
        'description': "Breast Cancer Prediction (30 features)",
        'feature_names': cancer.feature_names
    }

    # 5. Create California Housing Subset
    # Keep features that matter most for house price (intentionally similar)
    california = fetch_california_housing()
    X_cal2 = california.data[:, [0, 1, 2, 3, 4]]  # Use first 5 features
    y_cal2 = california.target + np.random.normal(0, 0.05, california.target.shape)  # Add minor noise

    # Standardize
    scaler_X = StandardScaler()
    X_cal2 = scaler_X.fit_transform(X_cal2)
    scaler_y = StandardScaler()
    y_cal2 = scaler_y.fit_transform(y_cal2.reshape(-1, 1)).ravel()

    datasets['california_subset'] = {
        'X': X_cal2,
        'y': y_cal2,
        'description': "California Housing Subset (5 features)",
        'feature_names': california.feature_names[:5]
    }

    # 6. California Older dataset (high-house-age subset; related to California Housing)
    print("Creating California Older Housing dataset")
    california = fetch_california_housing()

    # Filter by HouseAge (column 1) to get the older-housing half of the data
    older_indices = california.data[:, 1] > np.median(california.data[:, 1])
    X_older = california.data[older_indices]
    y_older = california.target[older_indices]

    # For intentional similarity with original dataset, keep feature structure
    # but add minor modifications to values
    X_older_modified = X_older.copy()
    # Mildly increase the AveBedrms feature (column 4) in the older subset
    X_older_modified[:, 4] = X_older[:, 4] * 1.05

    # Standardize
    scaler_X = StandardScaler()
    X_older_modified = scaler_X.fit_transform(X_older_modified)
    scaler_y = StandardScaler()
    y_older = scaler_y.fit_transform(y_older.reshape(-1, 1)).ravel()

    datasets['california_older'] = {
        'X': X_older_modified,
        'y': y_older,
        'description': "California Older Housing (8 features)",
        'feature_names': california.feature_names
    }

    print(f"Loaded {len(datasets)} datasets")
    for name, data in datasets.items():
        print(f"  - {name}: {data['description']}, samples: {data['X'].shape[0]}, features: {data['X'].shape[1]}")

    return datasets


def run_lifelong_learning_experiment():
    """Run lifelong learning experiment on various regression datasets"""
    print("Loading datasets...")
    datasets = load_datasets()

    # Create visualization directory
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)

    # Create model with improved parameters
    model = AdvancedLifelongRegressionModel(
        buffer_size=3000,  # Larger memory buffer
        shared_dim=128,  # Increased shared representation dimension
        embedding_dim=64,  # Increased task embedding dimension
        similarity_threshold=0.25,  # Lower threshold to more easily detect task similarities
        visualization_dir=vis_dir  # Set visualization directory
    )

    # Define task sequence - order matters for showcasing forgetting and transfer
    task_sequence = ['california', 'wine', 'diabetes', 'cancer', 'california_older', 'california_subset']

    # Performance tracking
    all_metrics = []
    forgetting_metrics = {task_id: {'r2': [], 'mse': [], 'mae': []} for task_id in task_sequence}

    # Train all tasks sequentially
    for i, task_id in enumerate(task_sequence):
        print(f"\n\n===== Training Task {i + 1}: {task_id} ({datasets[task_id]['description']}) =====")

        # Train on the train split; evaluation later uses the held-out test split.
        X = datasets[task_id]['X_train']
        y = datasets[task_id]['y_train']

        # Train model with optimized parameters
        metrics = model.train_task(
            task_id, X, y,
            epochs=80,  # Training epochs
            batch_size=32,  # Batch size
            lr=0.001,  # Learning rate
            ewc_lambda=100  # EWC regularization strength
        )
        all_metrics.append((task_id, metrics))

        # Evaluate all previous tasks on their HELD-OUT TEST SPLIT
        # (not a random subset of the training set — that would leak).
        for prev_task_id in task_sequence[:i]:
            prev_X = datasets[prev_task_id]['X_test']
            prev_y = datasets[prev_task_id]['y_test']

            prev_metrics = model.evaluate(prev_task_id, prev_X, prev_y)

            forgetting_metrics[prev_task_id]['r2'].append(prev_metrics['r2'])
            forgetting_metrics[prev_task_id]['mse'].append(prev_metrics['mse'])
            forgetting_metrics[prev_task_id]['mae'].append(prev_metrics['mae'])

            print(f"  Task {prev_task_id} after learning {task_id}: R²={prev_metrics['r2']:.4f}")

    # Save forgetting metrics for visualization
    model.forgetting_metrics = forgetting_metrics

    # Print final results
    print("\n\n===== Lifelong Learning Performance Summary =====")
    for task_id, metrics in all_metrics:
        print(f"Task {task_id} ({datasets[task_id]['description']}): "
              f"R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # Analyze catastrophic forgetting
    print("\n===== Catastrophic Forgetting Analysis =====")
    for i, task_id in enumerate(task_sequence):
        final_X = datasets[task_id]['X_test']
        final_y = datasets[task_id]['y_test']

        final_metrics = model.evaluate(task_id, final_X, final_y)
        original_metrics = all_metrics[i][1]

        r2_change = final_metrics['r2'] - original_metrics['r2']
        relative_change = (r2_change / abs(original_metrics['r2'])) * 100 if original_metrics['r2'] != 0 else 0

        print(f"Task {task_id}: Initial R²={original_metrics['r2']:.4f}, Final R²={final_metrics['r2']:.4f}")
        print(f"  R² change: {r2_change:.4f} ({relative_change:.2f}%)")

        # Print other metric changes
        mse_change = final_metrics['mse'] - original_metrics['mse']
        mae_change = final_metrics['mae'] - original_metrics['mae']
        print(f"  MSE change: {mse_change:.4f}, MAE change: {mae_change:.4f}")

    # Visualize results
    model.visualize_task_similarities()
    model.visualize_forgetting()
    model.visualize_performance()

    # Analyze housing dataset similarities
    print("\n===== Housing Dataset Similarity Analysis =====")
    housing_datasets = ['california', 'california_older', 'california_subset']
    for i in range(len(housing_datasets)):
        for j in range(i + 1, len(housing_datasets)):
            dataset1 = housing_datasets[i]
            dataset2 = housing_datasets[j]

            if dataset1 in model.task_similarities and dataset2 in model.task_similarities[dataset1]:
                sim = model.task_similarities[dataset1][dataset2]
                print(f"{dataset1} and {dataset2} similarity: {sim:.4f}")
            elif dataset2 in model.task_similarities and dataset1 in model.task_similarities[dataset2]:
                sim = model.task_similarities[dataset2][dataset1]
                print(f"{dataset1} and {dataset2} similarity: {sim:.4f}")
            else:
                print(f"{dataset1} and {dataset2} similarity data not available")

    # Analyze more unexpected similarity relationships
    print("\n===== Unexpected Similarity Analysis =====")
    # Check Cancer and California Older (highlighted in the presentation)
    if 'cancer' in model.task_similarities and 'california_older' in model.task_similarities['cancer']:
        sim = model.task_similarities['cancer']['california_older']
        print(f"Cancer and California Older similarity: {sim:.4f}")
        if model.task_id_mapping.get('cancer', 'cancer') and model.task_id_mapping.get('california_older',
                                                                                       'california_older'):
            model.analyze_parameter_transfer('cancer', 'california_older')

    # Check Diabetes and Cancer
    if 'diabetes' in model.task_similarities and 'cancer' in model.task_similarities['diabetes']:
        sim = model.task_similarities['diabetes']['cancer']
        print(f"Diabetes and Cancer similarity: {sim:.4f}")
        if model.task_id_mapping.get('diabetes', 'diabetes') and model.task_id_mapping.get('cancer', 'cancer'):
            model.analyze_parameter_transfer('diabetes', 'cancer')

    # Analyze overall task relationships
    model.analyze_task_relationships()

    # Save model results to file
    model.save_results()

    print(f"\nAll visualizations and analysis saved to {model.visualization_dir} directory")

    return model


def visualize_from_saved_results():
    """Visualize results from saved data (if available)"""
    vis_dir = 'visualizations'

    try:
        import pickle
        with open(os.path.join(vis_dir, 'model_results.pkl'), 'rb') as f:
            results = pickle.load(f)

        # Task similarity matrix
        task_ids = list(results['task_metrics'].keys())
        n_tasks = len(task_ids)

        # Create similarity matrix
        sim_matrix = np.zeros((n_tasks, n_tasks))

        for i, task1 in enumerate(task_ids):
            for j, task2 in enumerate(task_ids):
                if task1 == task2:
                    sim_matrix[i, j] = 1.0
                elif task1 in results['task_similarities'] and task2 in results['task_similarities'][task1]:
                    sim_matrix[i, j] = results['task_similarities'][task1][task2]
                elif task2 in results['task_similarities'] and task1 in results['task_similarities'][task2]:
                    sim_matrix[i, j] = results['task_similarities'][task2][task1]

        # Set matplotlib font
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans',
                                           'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Similarity')

        # Add labels
        plt.xticks(range(n_tasks), task_ids, rotation=45)
        plt.yticks(range(n_tasks), task_ids)

        # Add value labels
        for i in range(n_tasks):
            for j in range(n_tasks):
                plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center',
                         color='white' if sim_matrix[i, j] < 0.7 else 'black')

        plt.title('Task Similarity Matrix (from saved results)')
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(vis_dir, 'task_similarity_matrix_from_saved.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations from saved results created in {vis_dir}")

    except FileNotFoundError:
        print("No saved results found. Run the experiment first.")
    except Exception as e:
        print(f"Error visualizing from saved results: {e}")


if __name__ == "__main__":
    # Run the full experiment
    model = run_lifelong_learning_experiment()

    # Or visualize from saved results
    # visualize_from_saved_results()