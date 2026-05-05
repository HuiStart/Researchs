import sys
import os
# 获取当前文件的绝对路径，向上跳2级就是项目根目录Researchs
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from ai_researcher.deep_reviewer import DeepReviewer

custom_model_name = "/media/a510/AE8801F58801BD391/yzx/HDM/Researchs/.hf_cache/models--WestlakeNLP--DeepReviewer-7B/snapshots/cc31e8d72f62ad24c46f9a255f0cec31f18e8949"
# reviewer = DeepReviewer(custom_model_name="/zhuminjun/llama70/DeepReviewer_14B", device="cuda", tensor_parallel_size=2, gpu_memory_utilization=0.95)
reviewer = DeepReviewer(custom_model_name=custom_model_name, device="cuda", tensor_parallel_size=2, gpu_memory_utilization=0.95)

# Other parameters you can customize:
# - custom_model_name:  Path to a custom DeepReviewer model (overrides model_size).
# - tensor_parallel_size:  并行处理的GPU数量.
# - gpu_memory_utilization:  GPU内存的使用比例.