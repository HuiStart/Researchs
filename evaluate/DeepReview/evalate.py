import json
from itertools import combinations

import torch
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

'''
工作流程：读取ai生成的review文本 -> 将ai打分与人类真实的评审分数对比 -> 计算各种数学评估指标并输出markdown表格
'''

# 从ai生成的杂乱的review中，提取出结构化的论文评审内容，整理成一个干净的dict
def get_pred(pred_context):
    """
    Extract and structure the prediction content from the model output.

    Args:
        pred_context: Raw prediction context from the model

    Returns:
        dict: Structured prediction with all review components
    """
    try:
        # Extract the review content from the boxed format
        pred_context = pred_context.split(r'\boxed_review{')[-1].split('\n}')[0]
    except:
        pred_context = pred_context['output'].split(r'\boxed_review{')[-1].split('\n}')[0]

    # Initialize a dictionary with all review components
    pred = {
        'Summary': '',
        'Rating': '',
        'Soundness': '',
        'Presentation': '',
        'Contribution': '',
        'Strengths': '',
        'Weaknesses': '',
        'Suggestions': '',
        'Questions': '',
        'Confidence': '',
        'Decision': ''
    }

    # Parse each section from the prediction context --- 按章节拆分评审文本
    for context in pred_context.split('## '):
        for key in pred:
            if key + ':\n\n' in context:
                pred[key] = context.split(key + ':\n\n')[-1].strip()

    return pred

# 自动阅卷打分函数 ---- 读取模型的审稿预测结果，和人类专家的真实审稿结果做全面对比，计算出评分误差、相关性、录用决策准确率等指标，最终输出一份模型性能成绩单。
def evaluate_deep_reviewer(data_path, mode='standard'):
    """
    评估 DeepReviewer 模型的性能

    Args:
        data_path: DeepReviewer 模型预测结果的路径
        mode: Evaluation mode ('fast', 'standard', or 'best')

    Returns:
        dict: Results of the evaluation with various metrics
    """
    print(f"Evaluating DeepReviewer in {mode} mode...")

    # Load the DeepReviewer predictions
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists for storing metrics
    rating_mse_list_n = []
    rating_mae_list_n = []
    soundness_mse_list_n = []   # soundness 稳健性
    soundness_mae_list_n = []
    presentation_mse_list_n = []    # 表现
    presentation_mae_list_n = []
    contribution_mse_list_n = []    # 贡献
    contribution_mae_list_n = []

    # For spearman correlation ---- 斯皮尔曼等级相关系数
    true_ratings = []
    pred_ratings = []
    true_soundness = []
    pred_soundness = []
    true_presentation = []
    pred_presentation = []
    true_contribution = []
    pred_contribution = []

    # For pairwise comparison ---- 成对比较
    paper_scores = []

    # For decision evaluation
    decision_acc = []
    true_decisions = []
    pred_decisions = []

    # ⭐Process each paper's review
    for item in data:
        # Parse the prediction in the specified mode
        pred = get_pred(item[f'pred_{mode}_mode'])
        item['pred'] = pred

        '''
            针对每一篇论文，对比 AI 的评分与人类专家的评分，并计算各种误差和统计指标。
        '''
        if 'pred' in item:
            try:
                # Skip if no rating is provided ---- 跳过无效数据（无评分的论文）
                if item['pred']['Rating'] == '':
                    continue

                # Extract human review scores for this paper ---- 提取人类专家的真实评分（标准答案）
                rates = torch.tensor([int(r['content']['rating'][0]) for r in item['review']], dtype=torch.float32)
                soundness = torch.tensor([int(r['content']['soundness'][0]) for r in item['review']],
                                         dtype=torch.float32)
                presentation = torch.tensor([int(r['content']['presentation'][0]) for r in item['review']],
                                            dtype=torch.float32)
                contribution = torch.tensor([int(r['content']['contribution'][0]) for r in item['review']],
                                            dtype=torch.float32)

                # Extract model predictions (handling possible newlines in output) ---- 提取ai评分
                estimated_score = torch.tensor(float(item['pred']['Rating'].split('\n')[0]))
                estimated_soundness = torch.tensor(float(item['pred']['Soundness'].split('\n')[0]))
                estimated_presentation = torch.tensor(float(item['pred']['Presentation'].split('\n')[0]))
                estimated_contribution = torch.tensor(float(item['pred']['Contribution'].split('\n')[0]))

                # Calculate average human scores as ground truth proxies
                true_rate_proxy_n = rates.mean()
                true_soundness_proxy_n = soundness.mean()
                true_presentation_proxy_n = presentation.mean()
                true_contribution_proxy_n = contribution.mean()

                # Store scores for pairwise comparison ---- 存储数据 + 计算单篇误差（MSE/MAE）
                paper_scores.append({
                    'true_rating': true_rate_proxy_n.item(),
                    'pred_rating': estimated_score.item(),
                    'true_soundness': true_soundness_proxy_n.item(),
                    'pred_soundness': estimated_soundness.item(),
                    'true_presentation': true_presentation_proxy_n.item(),
                    'pred_presentation': estimated_presentation.item(),
                    'true_contribution': true_contribution_proxy_n.item(),
                    'pred_contribution': estimated_contribution.item()
                })

                # Store values for correlations and other metrics
                true_ratings.append(true_rate_proxy_n.item())
                pred_ratings.append(estimated_score.item())
                true_soundness.append(true_soundness_proxy_n.item())
                pred_soundness.append(estimated_soundness.item())
                true_presentation.append(true_presentation_proxy_n.item())
                pred_presentation.append(estimated_presentation.item())
                true_contribution.append(true_contribution_proxy_n.item())
                pred_contribution.append(estimated_contribution.item())

                # Calculate MSE and MAE for each metric
                rating_mse_list_n.append(torch.pow(estimated_score - true_rate_proxy_n, 2).item())
                rating_mae_list_n.append(torch.abs(estimated_score - true_rate_proxy_n).item())
                soundness_mse_list_n.append(torch.pow(estimated_soundness - true_soundness_proxy_n, 2).item())
                soundness_mae_list_n.append(torch.abs(estimated_soundness - true_soundness_proxy_n).item())
                presentation_mse_list_n.append(torch.pow(estimated_presentation - true_presentation_proxy_n, 2).item())
                presentation_mae_list_n.append(torch.abs(estimated_presentation - true_presentation_proxy_n).item())
                contribution_mse_list_n.append(torch.pow(estimated_contribution - true_contribution_proxy_n, 2).item())
                contribution_mae_list_n.append(torch.abs(estimated_contribution - true_contribution_proxy_n).item())

                # Process decision metrics ---- 评估录用/拒绝
                true_decision = item['decision'].lower()    # 统一转换为小写
                pred_decision = item['pred']['Decision'].lower().strip()    # .strip()去除首尾多余空格和换行符

                # Normalize decision to accept/reject
                # ai可能生成Strongly Accept，只要包含accept，统一转换。 否则一律视为拒绝（reject）
                if 'accept' in pred_decision.lower():
                    pred_decision = 'accept'
                else:
                    pred_decision = 'reject'

                # Convert to binary for F1 calculation ---- 二值化处理（为计算 F1 和 Precision/Recall 做准备）
                if 'accept' in pred_decision:
                    pred_decisions.append(1)
                else:
                    pred_decisions.append(0)    # [0, 1, 1, 0...]
                if 'accept' in true_decision:
                    true_decisions.append(1)
                else:
                    true_decisions.append(0)

                # Calculate decision accuracy --- 计算即时准确率
                '''
                    原理：AI 说 accept 且真实结果包含 accept，记录为 1.0(float方便后续计算)
                    最后，通过 torch.mean(torch.tensor(decision_acc)) 计算这些 0 和 1 的平均值，从而得到最终的总准确率。
                '''
                if pred_decision in true_decision:
                    decision_acc.append(1.)
                else:
                    decision_acc.append(0.)

            except Exception as e:
                print(f"Error processing item: {e}")
                pass

    # Calculate pairwise comparison accuracy ---- 计算两两比较准确率
    pairwise_accuracies = calculate_pairwise_accuracies(paper_scores)

    # Calculate mean values for all metrics
    rating_mse_n = torch.mean(torch.tensor(rating_mse_list_n)).item()
    rating_mae_n = torch.mean(torch.tensor(rating_mae_list_n)).item()
    soundness_mse_n = torch.mean(torch.tensor(soundness_mse_list_n)).item()
    soundness_mae_n = torch.mean(torch.tensor(soundness_mae_list_n)).item()
    presentation_mse_n = torch.mean(torch.tensor(presentation_mse_list_n)).item()
    presentation_mae_n = torch.mean(torch.tensor(presentation_mae_list_n)).item()
    contribution_mse_n = torch.mean(torch.tensor(contribution_mse_list_n)).item()
    contribution_mae_n = torch.mean(torch.tensor(contribution_mae_list_n)).item()

    # Calculate Spearman correlations
    rating_spearman_val, _ = spearmanr(true_ratings, pred_ratings)
    soundness_spearman_val, _ = spearmanr(true_soundness, pred_soundness)
    presentation_spearman_val, _ = spearmanr(true_presentation, pred_presentation)
    contribution_spearman_val, _ = spearmanr(true_contribution, pred_contribution)

    # Format Spearman values
    rating_spearman = f"{rating_spearman_val:.4f}" if not isinstance(rating_spearman_val, float) or not (
                rating_spearman_val is None) else 'nan'
    soundness_spearman = f"{soundness_spearman_val:.4f}" if not isinstance(soundness_spearman_val, float) or not (
                soundness_spearman_val is None) else 'nan'
    presentation_spearman = f"{presentation_spearman_val:.4f}" if not isinstance(presentation_spearman_val,
                                                                                 float) or not (
                presentation_spearman_val is None) else 'nan'
    contribution_spearman = f"{contribution_spearman_val:.4f}" if not isinstance(contribution_spearman_val,
                                                                                 float) or not (
                contribution_spearman_val is None) else 'nan'

    # Calculate decision metrics
    decision_acc_val = torch.mean(torch.tensor(decision_acc)).item()
    precision, recall, f1_val, _ = precision_recall_fscore_support(true_decisions, pred_decisions, average='macro')
    f1_score = float(f1_val)

    # Prepare results dictionary
    results = {
        'Model/Method': f'DeepReviewer {mode} (n={len(data)})',
        'Rating MSE': f"{rating_mse_n:.4f}",
        'Rating MAE': f"{rating_mae_n:.4f}",
        'Soundness MSE': f"{soundness_mse_n:.4f}",
        'Soundness MAE': f"{soundness_mae_n:.4f}",
        'Presentation MSE': f"{presentation_mse_n:.4f}",
        'Presentation MAE': f"{presentation_mae_n:.4f}",
        'Contribution MSE': f"{contribution_mse_n:.4f}",
        'Contribution MAE': f"{contribution_mae_n:.4f}",
        'Rating Spearman': rating_spearman,
        'Soundness Spearman': soundness_spearman,
        'Presentation Spearman': presentation_spearman,
        'Contribution Spearman': contribution_spearman,
        'Decision Accuracy': f"{decision_acc_val:.4f}",
        'Decision F1': f"{f1_score:.4f}",
        'Pairwise Rating Acc': f"{pairwise_accuracies['rating']:.4f}",
        'Pairwise Soundness Acc': f"{pairwise_accuracies['soundness']:.4f}",
        'Pairwise Presentation Acc': f"{pairwise_accuracies['presentation']:.4f}",
        'Pairwise Contribution Acc': f"{pairwise_accuracies['contribution']:.4f}",
    }

    return results

# pairwise：两两对比 ---- 给模型两篇论文，它能否挑出那篇真实得分更高的
def calculate_pairwise_accuracies(paper_scores):
    """
    Calculate pairwise accuracy for each metric by comparing rankings.

    Args:
        paper_scores: List of dictionaries containing true and predicted scores
        paper_scores：包含真实分数和预测分数的词典列表

    Returns:
        dict: Pairwise accuracies for each metric
    """
    total_pairs = 0
    correct_pairs = {
        'rating': 0,
        'soundness': 0,
        'presentation': 0,
        'contribution': 0
    }

    # Get all possible paper pairs using combinations
    for paper1, paper2 in combinations(paper_scores, 2):
        total_pairs += 1

        # Check if model correctly identifies which paper has higher rating
        true_rating_order = paper1['true_rating'] > paper2['true_rating']
        pred_rating_order = paper1['pred_rating'] > paper2['pred_rating']
        if true_rating_order == pred_rating_order:
            correct_pairs['rating'] += 1

        # Check soundness ranking
        true_soundness_order = paper1['true_soundness'] > paper2['true_soundness']
        pred_soundness_order = paper1['pred_soundness'] > paper2['pred_soundness']
        if true_soundness_order == pred_soundness_order:
            correct_pairs['soundness'] += 1

        # Check presentation ranking
        true_presentation_order = paper1['true_presentation'] > paper2['true_presentation']
        pred_presentation_order = paper1['pred_presentation'] > paper2['pred_presentation']
        if true_presentation_order == pred_presentation_order:
            correct_pairs['presentation'] += 1

        # Check contribution ranking
        true_contribution_order = paper1['true_contribution'] > paper2['true_contribution']
        pred_contribution_order = paper1['pred_contribution'] > paper2['pred_contribution']
        if true_contribution_order == pred_contribution_order:
            correct_pairs['contribution'] += 1

    # Calculate accuracies as percentage of correct pairs
    pairwise_accuracies = {
        metric: count / total_pairs if total_pairs > 0 else 0
        for metric, count in correct_pairs.items()
    }

    return pairwise_accuracies


def create_markdown_table(results):
    """
    Create a markdown table for displaying evaluation results.

    Args:
        results: Dictionary containing evaluation results

    Returns:
        str: Markdown table representation
    """
    # Define table header
    header = "| Metric | Value |\n|---|---|\n"

    # Add all results to table
    table_rows = []
    for key, value in results.items():
        if key != "Model/Method":  # Skip the model name
            table_rows.append(f"| {key} | {value} |")

    # Join all rows and return complete table
    return header + "\n".join(table_rows)


def main():
    """
    Main function to run the evaluation.
    """

    # Evaluate both modes and display results
    for mode in ['fast', 'standard','best']:
        results = evaluate_deep_reviewer('sample.json', mode)
        print(f"\nResults for DeepReviewer in {mode} mode:")
        markdown_table = create_markdown_table(results)
        print(markdown_table)


if __name__ == "__main__":
    main()