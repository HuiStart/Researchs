import sys
import os
# 获取当前文件的绝对路径，向上跳2级就是项目根目录Researchs
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from ai_researcher import DeepReviewer

custom_model_name = "/media/a510/AE8801F58801BD391/yzx/HDM/Researchs/.hf_cache/models--WestlakeNLP--DeepReviewer-7B/snapshots/cc31e8d72f62ad24c46f9a255f0cec31f18e8949"
# Initialize DeepReviewer with 14B model
deep_reviewer = DeepReviewer(custom_model_name=custom_model_name,model_size="7B")

# Review a paper with multiple simulated reviewers in Standard Mode
review_results = deep_reviewer.evaluate(
    paper_text,
    mode="Fast Mode",  # Options: "Fast Mode", "Standard Mode", "Best Mode"
    reviewer_num=4         # Simulate 4 different reviewers
)

# Print review results
for i, review in enumerate(review_results[0]['reviews']):
    print(f"Reviewer {i+1} Rating: {review.get('rating', 'N/A')}")
    print(f"Reviewer {i+1} Summary: {review.get('summary', 'N/A')[:100]}...")