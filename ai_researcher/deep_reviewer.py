import re
import requests
import json
from collections.abc import Iterable
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams


def _install_transformers_vllm_tokenizer_compat() -> None:
    """Restore tokenizer properties expected by vLLM 0.10 with Transformers 5."""
    if not hasattr(PreTrainedTokenizerBase, "special_tokens_map_extended"):

        @property
        def special_tokens_map_extended(self):
            special_map = getattr(self, "_special_tokens_map", None)
            special_attrs = getattr(self, "SPECIAL_TOKENS_ATTRIBUTES", ())

            if special_map is None:
                return dict(getattr(self, "special_tokens_map", {}))

            return {
                attr: special_map[attr]
                for attr in special_attrs
                if special_map.get(attr) is not None
            }

        PreTrainedTokenizerBase.special_tokens_map_extended = special_tokens_map_extended

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):

        @property
        def all_special_tokens_extended(self):
            all_toks = []
            for attr_value in self.special_tokens_map_extended.values():
                if isinstance(attr_value, Iterable) and not isinstance(attr_value, (str, bytes)):
                    values = attr_value
                else:
                    values = [attr_value]

                for token in values:
                    if token is not None and token not in all_toks:
                        all_toks.append(token)
            return all_toks

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


_install_transformers_vllm_tokenizer_compat()

# Helper Functions for Best Mode
# Adapted from the provided Flask app (main.py)

'''
    基于 DeepReviewer 模型（14B/7B）,支持三种模式
    Fast：快速生成review
    Standard：模拟多审稿人 + 自我验证
    Best：先让模型提出3个背景知识问题，再调用 OpenScholar API 检索真实文献答案，最后结合检索结果生成更可靠的 deepreview
'''
def extract_questions_from_content(content: str) -> list[str]:
    """Extract questions from the questions block (e.g., \boxed_questions{...})."""
    questions = []
    # Attempt to find the content within oxed_questions{}
    # This regex is a common way to find such blocks if they exist.
    # If the questions are simply listed after a header like "❓ Questions", 
    # this part might need adjustment based on actual LLM output format.
    
    # First, try to find a specific block like oxed_questions{}
    boxed_questions_match = re.search(r'\boxed_questions\{(.*?)\}', content, re.DOTALL)
    lines = [] # Initialize lines to an empty list
    if boxed_questions_match:
        questions_block = boxed_questions_match.group(1)
        # Assuming questions within the block are separated by newlines
        lines = [line.strip() for line in questions_block.split('\n') if line.strip()]
    else:
        # Fallback or alternative: if questions are under a "## Questions" or "❓ Questions" header
        # This part might need refinement based on the actual output format from the LLM.
        # For now, let's assume questions are separated by newlines after such a header.
        if "❓ Questions" in content: # Or a similar marker
            potential_questions_section = content.split("❓ Questions", 1)[-1]
            lines = [line.strip() for line in potential_questions_section.split('\n') if line.strip()]
        elif "## Questions" in content: # Handle markdown style headers
            potential_questions_section = content.split("## Questions", 1)[-1]
            lines = [line.strip() for line in potential_questions_section.split('\n') if line.strip()]
        else: # if no specific block found, assume content itself might be questions or needs different parsing.
            # This part needs to be robust. For now, using the provided logic from main.py's extract_questions_from_content
            # This assumes questions are separated by newlines.
            lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Process lines to extract actual questions
    for line in lines:
        # Skip lines that are not questions (headers, etc.)
        # The flask example had:
        # if line.startswith('#') or not line:
        #    continue
        # This might need to be adapted if the LLM output for questions is different.
        # For now, let's assume any non-empty line in this block is a question.
        # A more robust solution might look for lines ending with '?' or starting with a number/bullet.
        cleaned_line = line.lstrip("0123456789. ").strip() # Remove leading numbers/bullets
        if cleaned_line and cleaned_line != "}": # Ensure it's not just the closing brace of a block
            questions.append(cleaned_line)
    
    # Deduplicate questions
    return list(dict.fromkeys(questions))


def retrieve_information(questions: list[str]) -> list[dict]:
    """Retrieve information for questions using the OpenScholar external API."""
    if not questions:
        return []
    try:
        # The URL for the OpenScholar API
        openscholar_api_url = 'http://127.0.0.1:38015/batch_ask'
        response = requests.post(
            openscholar_api_url,
            json={"questions": questions},
            timeout=600  # Set a reasonable timeout (in seconds)
        )

        if response.status_code == 200:
            # Assuming the API returns a JSON with a 'results' key
            # which is a list of dictionaries, one for each question.
            return response.json().get('results', [])
        else:
            # Log error or handle appropriately
            print(f"Error retrieving information from OpenScholar API: {response.status_code} - {response.text}")
            return [{"error": f"API Error {response.status_code}", "output": "", "final_passages": ""} for _ in questions]
    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, etc.
        print(f"Exception during information retrieval: {str(e)}")
        return [{"error": f"RequestException: {str(e)}", "output": "", "final_passages": ""} for _ in questions]


def get_question_and_answer_text(questions: list[str], results: list[dict]) -> str:
    """Format questions and answers for the second model call."""
    qa_text_parts = []
    for i, question in enumerate(questions):
        qa_text_parts.append(f"## Question {i + 1}:\n{question}")
        if i < len(results) and results[i]:
            result = results[i]
            passages = result.get("final_passages", "N/A")
            answer = result.get("output", "N/A")
            # Sanitize content slightly for inclusion in a prompt if necessary, though LLMs are usually robust.
            # The flask app used .replace('"', "'").replace('\\', '') which might be too aggressive.
            # Keeping it simple here.
            qa_text_parts.append(f"### Retrieved Passages:\n{passages}")
            qa_text_parts.append(f"### Answer from OpenScholar:\n{answer}")
        else:
            qa_text_parts.append("### Retrieved Passages:\nNo information retrieved.")
            qa_text_parts.append("### Answer from OpenScholar:\nNo answer retrieved.")
        qa_text_parts.append("**********") # Separator
    
    return "\n\n".join(qa_text_parts)


class DeepReviewer:
    """
    A class for generating automated academic peer reviews using DeepReviewer models.
    """

    def __init__(self,
                 model_size="7B",
                 custom_model_name=None,
                 device="cuda",
                 backend="vllm",
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.95,
                 max_model_len=90000,
                 max_num_seqs=256,
                 tokenizer_mode="auto",
                 enforce_eager=False,
                 disable_custom_all_reduce=False,
                 dtype="auto",
                 hf_token=None,
                 cache_dir=None,
                 trust_remote_code=True):
        """
        Initialize the DeepReviewer.

        Args:
            model_size (str): Size of the default model to use. Options: "14B", "70B", "123B"
            custom_model_name (str, optional): Custom model name to override default mapping
            device (str): Device to run the model on. Default is "cuda"
            backend (str): Inference backend, "vllm" or "transformers".
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            gpu_memory_utilization (float): Fraction of GPU memory to use
            max_model_len (int): Maximum context length for vLLM.
            max_num_seqs (int): vLLM scheduler concurrency / sampler warmup size.
            tokenizer_mode (str): vLLM tokenizer mode. Use "auto" for this model so vLLM can load the fast tokenizer.
            enforce_eager (bool): Disable CUDA graphs to avoid kernel-specific issues.
            disable_custom_all_reduce (bool): Disable vLLM custom all-reduce kernels.
            dtype (str): vLLM dtype, for example "auto" or "float16".
            hf_token (str | None): Optional Hugging Face token.
            cache_dir (str | None): Optional Hugging Face cache dir.
            trust_remote_code (bool): Whether to trust Hugging Face remote code.
        """
        model_mapping = {
            "14B": "WestlakeNLP/DeepReviewer-14B",
            "7B": "WestlakeNLP/DeepReviewer-7B",
        }

        # Determine model name
        if custom_model_name:
            model_name = custom_model_name
        else:
            if model_size not in model_mapping:
                raise ValueError(f"Invalid model size. Choose from {list(model_mapping.keys())}")
            model_name = model_mapping[model_size]

        self.backend = backend
        self.device = device
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = None
        if backend == "vllm":
            self.model = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                tokenizer_mode=tokenizer_mode,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                disable_custom_all_reduce=disable_custom_all_reduce,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif backend == "transformers":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=self._torch_dtype_from_arg(dtype),
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
        else:
            raise ValueError('backend must be "vllm" or "transformers"')

        # Store model configuration for reference
        self.model_config = {
            "backend": backend,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "tokenizer_mode": tokenizer_mode,
            "enforce_eager": enforce_eager,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "dtype": dtype,
        }

    @staticmethod
    def _torch_dtype_from_arg(dtype):
        if dtype == "auto":
            return "auto"
        if dtype in {"bfloat16", torch.bfloat16}:
            return torch.bfloat16
        return torch.float16

    def _build_prompt(self, system_prompt, paper_text, max_tokens):
        max_input_tokens = max(512, self.max_model_len - max_tokens)

        def render_prompt(user_content):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompt = render_prompt(paper_text)
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(input_ids) <= max_input_tokens:
            return prompt

        empty_prompt = render_prompt("")
        overhead = len(self.tokenizer(empty_prompt, add_special_tokens=False).input_ids)
        marker = "\n\n[The middle of the paper was truncated to fit the model context.]\n\n"
        marker_ids = self.tokenizer(marker, add_special_tokens=False).input_ids
        paper_ids = self.tokenizer(paper_text, add_special_tokens=False).input_ids

        def truncate_paper(token_budget):
            if token_budget <= 0:
                return ""
            if len(paper_ids) <= token_budget:
                return paper_text
            if token_budget > len(marker_ids) + 128:
                content_budget = token_budget - len(marker_ids)
                head_budget = max(1, int(content_budget * 0.65))
                tail_budget = max(1, content_budget - head_budget)
                head = self.tokenizer.decode(paper_ids[:head_budget], skip_special_tokens=False)
                tail = self.tokenizer.decode(paper_ids[-tail_budget:], skip_special_tokens=False)
                return head + marker + tail
            return self.tokenizer.decode(paper_ids[:token_budget], skip_special_tokens=False)

        paper_budget = max(0, max_input_tokens - overhead - 32)
        while True:
            prompt = render_prompt(truncate_paper(paper_budget))
            input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            if len(input_ids) <= max_input_tokens or paper_budget == 0:
                return prompt
            paper_budget = max(0, paper_budget - (len(input_ids) - max_input_tokens) - 32)

    @staticmethod
    def _generation_prefill(mode):
        if mode == "Fast Mode":
            return "\\boxed_review{\n## Summary:\n\n"
        return ""

    def _vllm_sampling_params(self, max_tokens):
        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        return SamplingParams(
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.08,
            frequency_penalty=0.1,
            max_tokens=max_tokens,
            stop=["<｜User｜>", "<｜end▁of▁sentence｜>"],
            stop_token_ids=stop_token_ids or None,
        )

    def _generate_with_transformers(self, prompts, max_tokens):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            model_device = self.model.device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.95,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results

    def _generate_system_prompt(self, mode="Standard Mode", reviewer_num=4):
        """
        Generate the system prompt based on the review mode and number of reviewers.

        Args:
            mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num (int): Number of reviewers to simulate

        Returns:
            str: System prompt for the specified mode
        """
        simreviewer_prompt = "When you simulate different reviewers, write the sections in this order: Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Rating and Confidence."

        if mode == "Best Mode":
            prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Best Mode. In this mode, you should aim to provide the most reliable review results by conducting a thorough analysis of the paper. I allow you to use search tools to obtain background knowledge about the paper - please provide three different questions. I will help you with the search. After you complete your thinking, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
            return prompt + simreviewer_prompt
        elif mode == "Standard Mode":
            prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Standard Mode. In this mode, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
            return prompt + simreviewer_prompt
        elif mode == "Fast Mode":
            return """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Fast Mode. In this mode, you should quickly provide the review results.

Output exactly one review in this format:

\\boxed_review{
## Summary:

...

## Soundness:

<numeric score>

## Presentation:

<numeric score>

## Contribution:

<numeric score>

## Strengths:

...

## Weaknesses:

...

## Suggestions:

...

## Questions:

...

## Rating:

<numeric score>

## Confidence:

<numeric score>

## Decision:

<Accept or Reject>
}

Keep every prose section concise. Do not repeat points. Always include all numeric scores and the decision."""
        else:
            return "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers."

    def evaluate(self, paper_context, mode="Standard Mode", reviewer_num=4, max_tokens=35000):
        """
        Generate a peer review for the given academic paper.

        Args:
            paper_context (str): The paper content to review. Can be a single string or a list of strings for batch processing.
            mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num (int): Number of reviewers to simulate
            max_tokens (int): Maximum number of tokens to generate for each LLM call.

        Returns:
            list: A list of structured reviews (dictionaries). Each dictionary corresponds to one input paper_context.
        """
        system_prompt = self._generate_system_prompt(mode, reviewer_num)

        if isinstance(paper_context, str):
            paper_contexts = [paper_context]
        elif isinstance(paper_context, list):
            paper_contexts = paper_context
        else:
            raise TypeError("paper_context must be a string or a list of strings.")

        generated_reviews_batch = []
        
        batch_size = 10 
        for i in range(0, len(paper_contexts), batch_size):
            current_batch_contexts = paper_contexts[i:i + batch_size]
            
            if mode != "Best Mode":
                prefill = self._generation_prefill(mode)
                prompts = []
                for single_paper_context in current_batch_contexts:
                    prompts.append(self._build_prompt(system_prompt, single_paper_context, max_tokens) + prefill)

                if self.backend == "vllm":
                    sampling_params = self._vllm_sampling_params(max_tokens)
                    outputs = self.model.generate(prompts, sampling_params)
                    generated_texts = [prefill + output.outputs[0].text for output in outputs]
                else:
                    generated_texts = [prefill + text for text in self._generate_with_transformers(prompts, max_tokens)]

                for generated_text in generated_texts:
                    review = self._parse_review(generated_text)
                    generated_reviews_batch.append(review)
            else: # Best Mode - Process one by one from the batch due to sequential nature of API calls
                for single_paper_context in current_batch_contexts:
                    # --- First LLM Call (Best Mode) ---
                    messages_step1 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": single_paper_context}
                    ]
                    input_text_step1 = self.tokenizer.apply_chat_template(
                        messages_step1, tokenize=False, add_generation_prompt=True
                    )
                    sampling_params_step1 = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=max_tokens)
                    
                    outputs_step1 = self.model.generate([input_text_step1], sampling_params_step1)
                    generated_text_step1 = outputs_step1[0].outputs[0].text

                    # --- Extract Questions (Best Mode) ---
                    questions = extract_questions_from_content(generated_text_step1)

                    if not questions:
                        # Fallback: parse the step 1 output as the final review.
                        review = self._parse_review(generated_text_step1)
                        generated_reviews_batch.append(review)
                        continue # Next paper in batch

                    # --- Retrieve Information from OpenScholar (Best Mode) ---
                    retrieved_data = retrieve_information(questions)

                    # --- Format Q&A Text (Best Mode) ---
                    qa_text = get_question_and_answer_text(questions, retrieved_data)

                    # --- Second LLM Call (Best Mode) ---
                    messages_step2 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": single_paper_context},
                        {"role": "assistant", "content": generated_text_step1}, 
                        {"role": "user", "content": qa_text} 
                    ]
                    input_text_step2 = self.tokenizer.apply_chat_template(
                        messages_step2, tokenize=False, add_generation_prompt=True
                    )
                    sampling_params_step2 = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=max_tokens)
                    
                    outputs_step2 = self.model.generate([input_text_step2], sampling_params_step2)
                    generated_text_step2 = outputs_step2[0].outputs[0].text
                    
                    review = self._parse_review(generated_text_step2)
                    generated_reviews_batch.append(review)

        return generated_reviews_batch

    def _parse_review(self, generated_text):
        """
        Parse the generated review text into structured format.

        Args:
            generated_text (str): Raw generated review text

        Returns:
            dict: Structured review with metadata and reviews
        """
        result = {
            "raw_text": generated_text,
            "reviews": [],
            "meta_review": {},
            "decision": ""
        }

        # Extract meta review if present
        meta_review_match = re.search(r'\\boxed_review\{(.*?)\n}', generated_text, re.DOTALL)
        if meta_review_match:
            result["meta_review"]['content'] = meta_review_match.group(1).strip()
            section = meta_review_match.group(1).strip()
            # Extract summary
            summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if summary_match:
                result["meta_review"]["summary"] = summary_match.group(1).strip()

            # Extract rating
            rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if rating_match:
                rating_text = rating_match.group(1).strip()
                # Try to extract a numerical rating (1-10)
                number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if number_match:
                    result["meta_review"]["rating"] = float(number_match.group(1))
                else:
                    result["meta_review"]["rating"] = rating_text

            # Extract other sections as needed
            for section_name in ["Soundness", "Presentation", "Contribution",
                                 "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if section_match:
                    result["meta_review"][section_name.lower()] = section_match.group(1).strip()

        # Extract simulated reviewers' feedback
        simreviewer_match = re.search(r'\\boxed_simreviewers\{(.*?)\n}', generated_text, re.DOTALL)
        if simreviewer_match:
            simreviewer_text = simreviewer_match.group(1).strip()
            # Split into individual reviewer sections
            reviewer_sections = re.split(r'## Reviewer \d+', simreviewer_text)
            # Skip the first empty section if it exists
            if reviewer_sections and not reviewer_sections[0].strip():
                reviewer_sections = reviewer_sections[1:]

            for i, section in enumerate(reviewer_sections):
                review = {
                    "reviewer_id": i + 1,
                    "text": section.strip()
                }

                # Extract summary
                summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if summary_match:
                    review["summary"] = summary_match.group(1).strip()

                # Extract rating
                rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if rating_match:
                    rating_text = rating_match.group(1).strip()
                    # Try to extract a numerical rating (1-10)
                    number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                    if number_match:
                        review["rating"] = float(number_match.group(1))
                    else:
                        review["rating"] = rating_text

                # Extract other sections as needed
                for section_name in ["Soundness", "Presentation", "Contribution",
                                     "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                    section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                    if section_match:
                        review[section_name.lower()] = section_match.group(1).strip()

                result["reviews"].append(review)

        # Extract decision if present
        decision_match = re.search(r'## Decision:\s*\n\s*(\w+)', generated_text)
        if decision_match:
            result["decision"] = decision_match.group(1).strip()

        return result
