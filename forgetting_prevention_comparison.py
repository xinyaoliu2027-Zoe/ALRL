# forgetting_prevention_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from improved_lifelong_regression import AdvancedLifelongRegressionModel

# 创建实验结果目录
EXP_DIR = 'experiment_results'
os.makedirs(EXP_DIR, exist_ok=True)

# Train/test split is fixed across runs so that all model comparisons see
# the same evaluation data, while training dynamics are still seeded per
# experiment via run_experiment(seed=...).
SPLIT_SEED = 42
TEST_SIZE = 0.2


def _make_entry(X, y, description, feature_names):
    """80/20 train/test split + standardisation fit on the train portion only.

    Standardising on the full dataset (and then evaluating on a random
    subset of the training rows) was the source of the over-optimistic
    R² ≈ 1.0 numbers in the original pipeline; this helper fixes that
    leak by fitting both the input and target StandardScalers on the
    train split only, then applying the same transform to the test split.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, shuffle=True
    )
    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr)
    X_te = sx.transform(X_te)
    sy = StandardScaler()
    y_tr = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = sy.transform(y_te.reshape(-1, 1)).ravel()
    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_test":  X_te, "y_test":  y_te,
        "description": description,
        "feature_names": feature_names,
    }


def load_datasets():
    """Load the six benchmark regression tasks with proper 80/20 splits."""
    datasets = {}

    # 1. California Housing
    california = fetch_california_housing()
    datasets['california'] = _make_entry(
        california.data, california.target,
        "California Housing (8 features)",
        california.feature_names,
    )

    # 2. Diabetes Progression
    print("Loading Diabetes dataset")
    diabetes = load_diabetes()
    datasets['diabetes'] = _make_entry(
        diabetes.data, diabetes.target,
        "Diabetes Progression Prediction (10 features)",
        diabetes.feature_names,
    )

    # 3. Wine (sklearn 3-class) — integer label used as ordinal regression target
    wine = load_wine()
    datasets['wine'] = _make_entry(
        wine.data, wine.target.astype(float),
        "Wine (sklearn, 13 features; ordinal class target)",
        wine.feature_names,
    )

    # 4. Breast Cancer (binary diagnosis used as regression target)
    cancer = load_breast_cancer()
    datasets['cancer'] = _make_entry(
        cancer.data, cancer.target.astype(float),
        "Breast Cancer (30 features; binary diagnosis target)",
        cancer.feature_names,
    )

    # 5. California Housing Subset (first 5 features, mild target noise)
    california = fetch_california_housing()
    X_cal_sub = california.data[:, [0, 1, 2, 3, 4]]
    y_cal_sub = california.target + np.random.normal(
        0, 0.05, california.target.shape
    )
    datasets['california_subset'] = _make_entry(
        X_cal_sub, y_cal_sub,
        "California Housing Subset (5 features)",
        california.feature_names[:5],
    )

    # 6. California Older (high-house-age subset; related to California Housing)
    print("Creating California Older Housing dataset")
    california = fetch_california_housing()
    older_idx = california.data[:, 1] > np.median(california.data[:, 1])
    X_older = california.data[older_idx].copy()
    # Mildly increase AveBedrms (column 4) in the older subset
    X_older[:, 4] *= 1.05
    y_older = california.target[older_idx]
    datasets['california_older'] = _make_entry(
        X_older, y_older,
        "California Older Housing (8 features)",
        california.feature_names,
    )

    print(f"Loaded {len(datasets)} datasets (80/20 train/test split per task)")
    for name, data in datasets.items():
        n_tr, n_te = data['X_train'].shape[0], data['X_test'].shape[0]
        d = data['X_train'].shape[1]
        print(f"  - {name}: {data['description']}, "
              f"train={n_tr}, test={n_te}, features={d}")

    return datasets


def run_experiment(use_ewc=True, use_improved_buffer=True, use_enhanced_replay=True,
                   seed=None):
    """运行一次实验，可以控制启用/禁用每种防遗忘机制。

    Parameters
    ----------
    seed : int or None
        If provided, all numpy / PyTorch / Python random sources are
        seeded with this value, and the result pickle is saved under
        ``EXP_DIR/<config>/seed_<seed>/`` so that multi-seed runs do
        not overwrite each other. If ``None`` (legacy default), the
        previous single-run behaviour and path are preserved.
    """
    # ---- Reproducibility: seed all random sources -------------------
    if seed is not None:
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 创建配置名称用于保存结果
    config_name = []
    if use_ewc:
        config_name.append("EWC")
    if use_improved_buffer:
        config_name.append("ImprovedBuffer")
    if use_enhanced_replay:
        config_name.append("EnhancedReplay")

    if not config_name:
        config_name = ["Baseline"]  # 如果都不使用，则为基准模型

    config_str = "_".join(config_name)
    seed_tag = f" (seed={seed})" if seed is not None else ""
    print(f"\n\n{'=' * 30} Running experiment: {config_str}{seed_tag} {'=' * 30}\n")

    # 创建此配置的目录（多 seed 模式时增加 seed_<i> 子目录）
    if seed is not None:
        config_dir = os.path.join(EXP_DIR, config_str, f"seed_{seed}")
    else:
        config_dir = os.path.join(EXP_DIR, config_str)
    os.makedirs(config_dir, exist_ok=True)

    # 加载数据集 (含可选的 UCI Concrete 真实回归任务)
    datasets = load_datasets()
    try:
        from datasets_extra import add_extra_datasets
        datasets.update(add_extra_datasets(which=("concrete",)))
        print("  [+] Added UCI Concrete Compressive Strength dataset.")
    except Exception as exc:
        print(f"  [warn] Could not load extra datasets: {exc}")
        print(f"         Continuing with the original 6-task benchmark.")

    # 创建模型
    model = AdvancedLifelongRegressionModel(
        buffer_size=3000,
        shared_dim=128,
        embedding_dim=64,
        similarity_threshold=0.25,
        visualization_dir=config_dir
    )

    # 定义任务序列
    # Insert 'concrete' between the medical block and the California-housing
    # follow-up block so that (a) the 7-task stream interleaves natively
    # continuous-target regression with the rest of the benchmark, and
    # (b) the California cluster still benefits from same-domain warm-start.
    base_sequence = ['california', 'wine', 'diabetes', 'cancer',
                     'california_older', 'california_subset']
    if 'concrete' in datasets:
        task_sequence = ['california', 'wine', 'diabetes', 'cancer',
                         'concrete', 'california_older', 'california_subset']
    else:
        task_sequence = base_sequence

    # 性能跟踪
    all_metrics = []
    forgetting_metrics = {task_id: {'r2': [], 'mse': [], 'mae': []} for task_id in task_sequence}

    # 顺序训练所有任务
    for i, task_id in enumerate(task_sequence):
        print(f"\n===== Training Task {i + 1}: {task_id} ({datasets[task_id]['description']}) =====")

        # 获取数据集 (训练只看 train split)
        X = datasets[task_id]['X_train']
        y = datasets[task_id]['y_train']

        # 取该任务的 test split (用于 "刚训完时" 的 fair initial R²)
        X_test_t = datasets[task_id]['X_test']
        y_test_t = datasets[task_id]['y_test']

        # 训练模型 - 根据配置决定是否使用每种防遗忘机制
        ewc_lambda = 100 if use_ewc else 0  # 如果不使用EWC，将lambda设为0

        # 临时修改模型的内存回放功能（使用monkey patch方法）
        original_sample_method = model._sample_from_buffer
        original_process_method = model._process_memory_batch

        # 如果不使用增强的内存缓冲区管理，则使用简单的均匀采样
        if not use_improved_buffer:
            def simple_sample(self, batch_size=32, target_dim=None, target_task=None):
                if len(self.memory_buffer) < batch_size // 2:
                    return None, None, None, None

                # 简单随机采样，无权重
                keys = list(self.memory_buffer.keys())
                sampled_indices = np.random.choice(
                    len(keys),
                    size=min(batch_size, len(keys)),
                    replace=False
                )

                # 分组
                memory_batches = {}
                for idx in sampled_indices:
                    key = keys[idx]
                    item = self.memory_buffer[key]
                    batch_key = f"{item['task_id']}_{item['dim']}"
                    if batch_key not in memory_batches:
                        memory_batches[batch_key] = []
                    memory_batches[batch_key].append(item)

                return memory_batches, None, None, None

            # 替换方法
            model._sample_from_buffer = lambda *args, **kwargs: simple_sample(model, *args, **kwargs)

        # 训练模型
        _ = model.train_task(
            task_id, X, y,
            epochs=80,
            batch_size=32,
            lr=0.001,
            ewc_lambda=ewc_lambda,  # 是否使用EWC
        )
        # 在 held-out test split 上重新评估 "initial" 性能 (覆盖 train_task
        # 返回的 train-set R², 那是 over-optimistic 的)
        metrics = model.evaluate(task_id, X_test_t, y_test_t)
        all_metrics.append((task_id, metrics))
        print(f"  Initial test R² for {task_id}: {metrics['r2']:.4f}, "
              f"MSE: {metrics['mse']:.4f}")

        # Disable replay only when BOTH replay mechanisms are turned off.
        # Previously this used `if not use_enhanced_replay:` alone, which
        # also wiped the buffer in the "improved buffer only" and
        # "EWC + improved buffer" configurations and made them
        # indistinguishable from the baseline / EWC-only runs. The
        # corrected condition keeps the buffer alive whenever either
        # mechanism is active, so each ablation cell is genuinely distinct.
        if not use_enhanced_replay and not use_improved_buffer:
            model.memory_buffer = OrderedDict()

        # 评估所有先前任务（在 held-out test split 上），测量灾难性遗忘
        for prev_task_id in task_sequence[:i]:
            prev_X_test = datasets[prev_task_id]['X_test']
            prev_y_test = datasets[prev_task_id]['y_test']

            prev_metrics = model.evaluate(prev_task_id, prev_X_test, prev_y_test)

            forgetting_metrics[prev_task_id]['r2'].append(prev_metrics['r2'])
            forgetting_metrics[prev_task_id]['mse'].append(prev_metrics['mse'])
            forgetting_metrics[prev_task_id]['mae'].append(prev_metrics['mae'])

            print(f"  Task {prev_task_id} after learning {task_id}: R²={prev_metrics['r2']:.4f}")

        # 恢复原始方法
        if not use_improved_buffer:
            model._sample_from_buffer = original_sample_method

    # 保存遗忘度量以便可视化
    model.forgetting_metrics = forgetting_metrics

    # 打印最终结果
    print("\n===== Lifelong Learning Performance Summary =====")
    for task_id, metrics in all_metrics:
        print(f"Task {task_id} ({datasets[task_id]['description']}): "
              f"R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # 分析灾难性遗忘
    print("\n===== Catastrophic Forgetting Analysis =====")
    forgetting_summary = {}

    for i, task_id in enumerate(task_sequence):
        final_X_test = datasets[task_id]['X_test']
        final_y_test = datasets[task_id]['y_test']

        final_metrics = model.evaluate(task_id, final_X_test, final_y_test)
        original_metrics = all_metrics[i][1]

        r2_change = final_metrics['r2'] - original_metrics['r2']
        relative_change = (r2_change / abs(original_metrics['r2'])) * 100 if original_metrics['r2'] != 0 else 0

        print(f"Task {task_id}: Initial R²={original_metrics['r2']:.4f}, Final R²={final_metrics['r2']:.4f}")
        print(f"  R² change: {r2_change:.4f} ({relative_change:.2f}%)")

        # 保存遗忘摘要
        forgetting_summary[task_id] = {
            'initial_r2': original_metrics['r2'],
            'final_r2': final_metrics['r2'],
            'r2_change': r2_change,
            'relative_change': relative_change
        }

    # 保存实验结果
    experiment_results = {
        'config': {
            'use_ewc': use_ewc,
            'use_improved_buffer': use_improved_buffer,
            'use_enhanced_replay': use_enhanced_replay,
            'config_name': config_str
        },
        'task_metrics': {task_id: metrics for task_id, metrics in all_metrics},
        'forgetting_metrics': forgetting_metrics,
        'forgetting_summary': forgetting_summary
    }

    # 保存结果
    results_file = os.path.join(config_dir, 'experiment_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(experiment_results, f)

    # 可视化
    model.visualize_task_similarities()
    model.visualize_forgetting()
    model.visualize_performance()

    # 计算平均遗忘率
    r2_changes = [data['r2_change'] for data in forgetting_summary.values()]
    avg_forgetting = np.mean(r2_changes)

    print(f"\nAverage R² change (forgetting measure): {avg_forgetting:.4f}")
    print(f"Results saved to {config_dir}")

    return experiment_results


def analyze_results():
    """分析所有实验结果并生成比较图表"""
    print("\n\n===== Analyzing All Experiment Results =====")

    # 创建结果比较目录
    comparison_dir = os.path.join(EXP_DIR, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # 收集所有实验配置
    all_configs = []

    for dir_name in os.listdir(EXP_DIR):
        if os.path.isdir(os.path.join(EXP_DIR, dir_name)) and dir_name != 'comparison':
            results_file = os.path.join(EXP_DIR, dir_name, 'experiment_results.pkl')
            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                    all_configs.append(results)

    if not all_configs:
        print("No experiment results found for analysis.")
        return

    # 按配置名称排序
    all_configs.sort(key=lambda x: x['config']['config_name'])

    # 提取配置和平均遗忘率
    config_names = [config['config']['config_name'] for config in all_configs]
    avg_forgetting = []

    for config in all_configs:
        r2_changes = [data['r2_change'] for data in config['forgetting_summary'].values()]
        avg_forgetting.append(np.mean(r2_changes))

    # 创建遗忘率比较图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(config_names, avg_forgetting)

    # 为正负值设置不同颜色
    for i, bar in enumerate(bars):
        if avg_forgetting[i] < 0:
            bar.set_color('tomato')  # 红色表示遗忘（负值）
        else:
            bar.set_color('mediumseagreen')  # 绿色表示改进（正值）

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Average R² Change')
    plt.title('Comparison of Forgetting Prevention Methods')
    plt.xticks(rotation=45, ha='right')

    # 添加数值标签
    for i, value in enumerate(avg_forgetting):
        plt.text(i, value + (0.01 if value >= 0 else -0.01),
                 f"{value:.4f}", ha='center', va='bottom' if value >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'forgetting_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建任务级别的遗忘率比较
    # 获取所有任务ID
    task_ids = list(all_configs[0]['forgetting_summary'].keys())

    # 创建任务级别的比较图
    plt.figure(figsize=(15, 10))

    bar_width = 0.1
    index = np.arange(len(task_ids))

    for i, config in enumerate(all_configs):
        r2_changes = [config['forgetting_summary'][task_id]['r2_change'] for task_id in task_ids]
        plt.bar(index + i * bar_width, r2_changes, bar_width, label=config['config']['config_name'])

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Tasks')
    plt.ylabel('R² Change')
    plt.title('Task-level Forgetting Comparison')
    plt.xticks(index + bar_width * (len(all_configs) - 1) / 2, task_ids)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(comparison_dir, 'task_level_forgetting.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建最终性能比较
    plt.figure(figsize=(15, 10))

    for i, config in enumerate(all_configs):
        final_r2 = [config['forgetting_summary'][task_id]['final_r2'] for task_id in task_ids]
        plt.bar(index + i * bar_width, final_r2, bar_width, label=config['config']['config_name'])

    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good Performance Threshold (R²=0.7)')
    plt.xlabel('Tasks')
    plt.ylabel('Final R² Score')
    plt.title('Final Performance Comparison')
    plt.xticks(index + bar_width * (len(all_configs) - 1) / 2, task_ids)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(comparison_dir, 'final_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison analysis saved to {comparison_dir}")


if __name__ == "__main__":
    import argparse
    from collections import OrderedDict

    parser = argparse.ArgumentParser(
        description="Run forgetting-prevention experiments. "
                    "Pass --seeds 0 1 2 3 4 to run multi-seed.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[None],
        help="One or more integer seeds. If omitted, a single un-seeded "
             "run is executed (legacy behaviour).")
    parser.add_argument(
        "--configs", type=int, nargs="+", default=list(range(8)),
        help="Subset of config indices to run (0..7). Default: all 8.")
    parser.add_argument(
        "--skip-analyze", action="store_true",
        help="Skip the legacy analyze_results() pass at the end (useful "
             "in multi-seed mode where we aggregate separately).")
    args = parser.parse_args()

    # All eight configurations: (use_ewc, use_improved_buffer, use_enhanced_replay)
    CONFIGS = [
        (False, False, False),  # 0  Baseline
        (True,  False, False),  # 1  EWC only
        (False, True,  False),  # 2  ImprovedBuffer only
        (False, False, True),   # 3  EnhancedReplay only
        (True,  True,  False),  # 4  EWC + ImprovedBuffer
        (True,  False, True),   # 5  EWC + EnhancedReplay
        (False, True,  True),   # 6  ImprovedBuffer + EnhancedReplay
        (True,  True,  True),   # 7  Full ALRL
    ]

    print(f"Running experiments with seeds={args.seeds}, "
          f"configs={args.configs} ...")
    for seed in args.seeds:
        for ci in args.configs:
            ewc, ibuf, rep = CONFIGS[ci]
            run_experiment(use_ewc=ewc, use_improved_buffer=ibuf,
                           use_enhanced_replay=rep, seed=seed)

    # Legacy single-run analysis only makes sense when seeds==[None].
    if not args.skip_analyze and args.seeds == [None]:
        analyze_results()