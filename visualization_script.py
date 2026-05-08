# visualization_script.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 创建可视化目录
vis_dir = 'visualizations'
os.makedirs(vis_dir, exist_ok=True)

# 尝试加载已保存的模型或结果
try:
    # 从pickle加载模型或结果
    results_file = os.path.join(vis_dir, 'model_results.pkl')

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        print("已加载保存的结果数据")

        # 提取任务ID
        task_ids = list(results['task_metrics'].keys())

        # 创建相似度矩阵
        sim_matrix = np.zeros((len(task_ids), len(task_ids)))

        # 填充相似度数据
        for i, task1 in enumerate(task_ids):
            for j, task2 in enumerate(task_ids):
                if task1 == task2:
                    sim_matrix[i, j] = 1.0
                elif task1 in results['task_similarities'] and task2 in results['task_similarities'][task1]:
                    sim_matrix[i, j] = results['task_similarities'][task1][task2]
                elif task2 in results['task_similarities'] and task1 in results['task_similarities'][task2]:
                    sim_matrix[i, j] = results['task_similarities'][task2][task1]

        # 绘制相似度热力图（使用Seaborn增强可视化效果）
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=True, cmap='viridis', vmin=0, vmax=1,
                    xticklabels=task_ids, yticklabels=task_ids, fmt='.2f')
        plt.title('任务相似度矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'task_similarity_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制性能比较图
        r2_scores = [results['task_metrics'][tid]['r2'] for tid in task_ids]
        mse_scores = [results['task_metrics'][tid]['mse'] for tid in task_ids]
        mae_scores = [results['task_metrics'][tid]['mae'] for tid in task_ids]

        # 获取任务维度信息（如果有）
        dims = []
        for tid in task_ids:
            if 'task_dims' in results:
                # 获取内部ID（如果有映射）
                if 'task_id_mapping' in results and tid in results['task_id_mapping']:
                    internal_tid = results['task_id_mapping'][tid]
                    dim = results['task_dims'].get(internal_tid, '?')
                else:
                    dim = results['task_dims'].get(tid, '?')
            else:
                dim = '?'
            dims.append(dim)

        # 创建性能对比图
        fig, ax1 = plt.subplots(figsize=(12, 7))

        x = np.arange(len(task_ids))
        width = 0.25

        bars1 = ax1.bar(x - width, r2_scores, width, label='R²', color='royalblue')

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x, mse_scores, width, label='MSE', color='tomato')
        bars3 = ax2.bar(x + width, mae_scores, width, label='MAE', color='green')

        # 添加性能阈值线
        ax1.axhline(y=0.7, color='blue', linestyle='--', alpha=0.5, label='良好性能阈值 (R²=0.7)')

        ax1.set_xlabel('任务')
        ax1.set_ylabel('R² 分数')
        ax2.set_ylabel('MSE/MAE 分数')
        plt.title('各任务最终性能对比')

        # 为每个任务添加维度信息
        labels = [f"{tid}\n({dim}维)" for tid, dim in zip(task_ids, dims)]
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')


        # 在柱状图上添加数值标签
        def add_labels(bars, ax):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)


        add_labels(bars1, ax1)
        add_labels(bars2, ax2)
        add_labels(bars3, ax2)

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'performance_comparison_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制灾难性遗忘分析图（如果有数据）
        if 'forgetting_metrics' in results:
            # R²值随时间变化
            plt.figure(figsize=(10, 6))
            for task_id in results['forgetting_metrics']:
                task_r2 = results['forgetting_metrics'][task_id]['r2']
                epochs = range(len(task_r2))
                plt.plot(epochs, task_r2, marker='o', label=f'任务 {task_id}')

            plt.title('各任务R²值随学习进程的变化')
            plt.xlabel('学习任务数')
            plt.ylabel('R² 分数')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'forgetting_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"所有可视化图表已保存到 '{vis_dir}' 目录")
    else:
        # 如果没有保存结果，创建示例可视化
        print("未找到保存的结果，创建示例可视化...")

        # 示例任务ID
        task_ids = ['california', 'wine', 'diabetes', 'cancer', 'california_older', 'california_subset']

        # 示例相似度矩阵
        sim_matrix = np.zeros((len(task_ids), len(task_ids)))

        # 填充示例数据
        sim_matrix[0, 4] = 0.42  # california与california_older
        sim_matrix[4, 0] = 0.42
        sim_matrix[0, 5] = 0.38  # california与california_subset
        sim_matrix[5, 0] = 0.38
        sim_matrix[2, 3] = 0.31  # diabetes与cancer
        sim_matrix[3, 2] = 0.31
        sim_matrix[4, 5] = 0.35  # california_older与california_subset
        sim_matrix[5, 4] = 0.35

        # 对角线设为1.0
        for i in range(len(task_ids)):
            sim_matrix[i, i] = 1.0

        # 绘制相似度热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=True, cmap='viridis', vmin=0, vmax=1,
                    xticklabels=task_ids, yticklabels=task_ids, fmt='.2f')
        plt.title('任务相似度矩阵 (示例数据)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'task_similarity_example.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 示例性能数据
        r2_scores = [0.93, 1.00, 0.99, 1.00, 0.95, 0.76]

        # 绘制性能柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(task_ids, r2_scores, color='royalblue')
        plt.axhline(y=0.7, color='red', linestyle='--', label='良好性能阈值 (R²=0.7)')
        plt.xlabel('任务')
        plt.ylabel('R² 分数')
        plt.title('各任务性能对比 (示例数据)')
        plt.xticks(rotation=45)

        # 添加数值标签
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'performance_example.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"示例可视化图表已保存到 '{vis_dir}' 目录")

except Exception as e:
    print(f"生成可视化时出错: {e}")