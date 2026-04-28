from ai_researcher.utils import get_reviewer_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams    # vLLM 库（一个高性能 LLM 推理框架）

'''
    核心功能：输入一篇paper，自动生成一份包含摘要、健全性、贡献度、评分以及最终录用建议的专业评审报告
    功能与作用: 论文评审器。
    模拟4位不同审稿人对论文进行评审，输出： Summary、Soundness、Presentation、Contribution、Rating、Accept/Reject决策等结构化结果。支持批量处理（batch_size=10）
'''

class CycleReviewer:
    """
    A class for evaluating research papers using CycleReviewer models.
    """

    def __init__(self,
                 model_size="8B",
                 custom_model_name=None,    # 直接输入一个 Hugging Face 的模型路径或名称，覆盖默认映射。
                 device="cuda",
                 tensor_parallel_size=1,    # 张量并行大小，用于多 GPU 场景（例如设为 4，表示用 4 张显卡跑一个大模型）
                 gpu_memory_utilization=0.95):  # 控制显存占用率，默认 0.95 意味着它会尽可能多地占据显存以提高吞吐量。
        """
        Initialize the CycleReviewer.

        Args:
            model_size (str): Size of the default model to use. Options: "8B", "70B", "123B"
            custom_model_name (str, optional): Custom model name to override default mapping
            device (str): Device to run the model on. Default is "cuda"
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            gpu_memory_utilization (float): Fraction of GPU memory to use
        """
        # 映射三种不同规模的model
        model_mapping = {
            "8B": "WestlakeNLP/CycleReviewer-ML-Llama3.1-8B",
            "70B": "WestlakeNLP/CycleReviewer-ML-Llama3.1-70B",
            "123B": "WestlakeNLP/CycleReviewer-ML-Pro-123B"
        }

        # Determine model name
        if custom_model_name:
            model_name = custom_model_name
        else:
            if model_size not in model_mapping:
                raise ValueError(f"Invalid model size. Choose from {list(model_mapping.keys())}")
            model_name = model_mapping[model_size]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model using vLLM
        # 使用vllm引擎加载model
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=50000,    # 很大了
            gpu_memory_utilization=gpu_memory_utilization
        )

        # Store model configuration for reference
        self.model_name = model_name
        self.model_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization
        }

    def evaluate(self, paper_context, include_detailed_feedback=True):
        """
        Evaluate a research paper.

        Args:
            paper_context (list or str): Paper to be reviewed
            include_detailed_feedback (bool): Whether to include detailed review sections

        Returns:
            dict: Review of the paper with various components
        """
        # Prepare system prompt for review
        system_prompt = \
            """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:
    
            1. Summary: Briefly outline main points and objectives.
            2. Soundness: Assess methodology and logical consistency.
            3. Presentation: Evaluate clarity, organization, and visual aids.
            4. Contribution: Analyze significance and novelty in the field.
            5. Strengths: Identify the paper's strongest aspects.
            6. Weaknesses: Point out areas for improvement.
            7. Questions: Pose questions for the authors.
            8. Rating: Score 1-10, justify your rating.
            9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).
    
            Maintain objectivity and provide specific examples from the paper to support your evaluation.
    
            You need to fill out **4** review opinions."""

        # Prepare paper context
        if type(paper_context) == str:
            paper_context = [paper_context]     # 转换成list

        generated_reviews = []
        batch_size = 10     # 每 10 篇论文为一个 Batch 进行并行推理
        for n in range(0,len(paper_context),batch_size):
            # Apply chat template
            prompts = []
            for r in range(min(batch_size, len(paper_context) - n)):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paper_context[r+n]}
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(input_text)
            # Prepare sampling parameters
            sampling_params = SamplingParams(
                temperature=0.4,
                top_p=0.95,
                max_tokens=7000
            )

            # Generate review
            outputs = self.model.generate(
                prompts,
                sampling_params
            )

            # Process generated review text
            for output_num in range(len(outputs)):
                # Process generated text
                generated_text = outputs[output_num].outputs[0].text
                # Use existing CycleResearcher utility to parse generated text
                review = get_reviewer_score(generated_text)
                generated_reviews.append(review)

        return generated_reviews
