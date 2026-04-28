import random

from tqdm import tqdm
import os
import re
import spacy
from src.use_search_apis import search_paper_via_query, retrieve_pes2o_passages

import numpy as np
import os
from nltk import sent_tokenize
#import vllm
import src.instructions as instructions
from FlagEmbedding import FlagReranker

'''
    功能与作用: OpenScholar 核心类。实现完整的 RAG Pipeline：,专门用于科学文献问答和综述生成
        检索与重排序：基于 BGE Cross-Encoder
        对检索到的论文段落重排序，支持引用数加权（norm_cite）。
        生成：构造带引用的 Prompt，调用大模型生成回答。
        反馈迭代：生成自我反馈（Feedback），根据反馈编辑答案，甚至二次检索补充文献。
        事后归因（Post-hoc Attribution）：为生成内容中缺少引用的句子自动补全引用标注。
        支持 run() 单条处理和 run_batch() 批量处理。
'''
nlp = spacy.load('en_core_web_sm')

# To compute API costs based on October 2023 pricing available at https://openai.com/ja-JP/api/pricing/
price_per_million = {"gpt-4o": 2.50, "gpt-4o-2024-08-06": 2.50, "gpt-4o-2024-05-13": 5.00, "gpt-4o-mini": 0.15, "gpt-4o-mini-2024-07-18": 0.15, "gpt-4-turbo": 10.0, "gpt-3.5-turbo-0125": 0.50,'OpenSciLM/Llama-3.1_OpenScholar-8B':1,'Qwen/Qwen3-0.6B':1}
price_per_million_output = {"gpt-4o": 10.00, "gpt-4o-2024-08-06": 10.00,  "gpt-4o-2024-05-13": 15.00, "gpt-4o-mini": 0.600, "gpt-4o-mini-2024-07-18": 0.600, "gpt-4-turbo": 30.0, "gpt-3.5-turbo-0125": 1.50,'OpenSciLM/Llama-3.1_OpenScholar-8B':1,'Qwen/Qwen3-0.6B':1}

def calculate_openai_api_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate OpenAI API cost based on the number of input and output tokens.
    
    Args:
    - input_tokens (int): Number of tokens in the input.
    - output_tokens (int): Estimated number of tokens in the output.
    - price_per_million_tokens (float): Cost per 1 million tokens (e.g., 0.02 for GPT-4).

    Returns:
    - float: The total API cost.
    """
    total_cost_input = (input_tokens / 1000000) * price_per_million[model_name]
    total_cost_output =  (output_tokens / 1000000) * price_per_million_output[model_name]
    total_cost = total_cost_input + total_cost_output
    return round(total_cost, 6)

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
    
def rerank_paragraphs_bge(query, paragraphs, reranker, norm_cite=False, start_index=0, use_abstract=False):
    paragraphs = [p for p in paragraphs if p["text"] is not None]
    if use_abstract is True:
        paragraph_texts = [p["title"] + "\n" + p["abstract"] + "\n" + p["text"] if "title" in p and "abstract" in p else p["text"] for p in paragraphs]
    else:
        paragraph_texts = [p["title"] + " " + p["text"] if "title" in p and p["title"] is not None else p["text"] for p in paragraphs]
    
    print(paragraph_texts[0])
    scores = reranker.compute_score([[query, p] for p in paragraph_texts], batch_size=100)
    if type(scores) is float:
        result_dic = {0: scores}
    else:
        result_dic = {p_id: score for p_id, score in enumerate(scores)}
    if norm_cite is True and len([item["citation_counts"] for item in paragraphs if "citation_counts" in item and item["citation_counts"] is not None]) > 0:
        # add normalized scores
        max_citations = max([item["citation_counts"] for item in paragraphs if "citation_counts" in item and item["citation_counts"] is not None])
        for p_id in result_dic:
            if "citation_counts" in paragraphs[p_id] and paragraphs[p_id]["citation_counts"] is not None:
                result_dic[p_id] = result_dic[p_id] + (paragraphs[p_id]["citation_counts"] / max_citations)
    p_ids = sorted(result_dic.items(), key=lambda x: x[1], reverse=True)
    new_orders = []
    id_mapping = {}
    for i, p_id in enumerate(p_ids):
        new_orders.append(paragraphs[p_id[0]])
        id_mapping[i] = int(p_id[0])
    return new_orders, result_dic, id_mapping

def create_prompt_with_llama3_format(prompt, system_message="You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers."):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(system_message)
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

def load_hf_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
    ):
        from transformers import AutoTokenizer

        # Need to explicitly import the olmo tokenizer.
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token)
        except:
            # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
        # set padding side to left for batch generation
        tokenizer.padding_side = padding_side
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
   
class OpenScholar(object):
    def __init__(self, model, tokenizer, client=None, api_model_name=None, use_contexts=True, top_n=8, reranker=None, min_citation=None, norm_cite=False, ss_retriever=False):
        self.model = model
        self.tokenizer = tokenizer
        self.client = client
        self.model_name = api_model_name
        self.top_n = top_n
        self.no_retrieval = not use_contexts
        self.reranker = reranker
        self.min_citation = min_citation
        self.norm_cite = norm_cite
        self.ss_retriever = ss_retriever
        self.use_contexts = use_contexts

    # Reranking: We rerank passages based on the LMs' predictions on how useful passages are.
    def process_ranking_results(self, result):
        ratings = {int(match.group(1)): int(match.group(2)) for match in re.finditer(r'\[(\d+)\] Rating: (\d)', result)}
        return ratings

    def reranking_passages_cross_encoder2(self, item, batch_size=5, llama3_chat=False, task_name="default", use_abstract=False):
        # 使用最小引用数目进行过滤
        if self.min_citation is not None:
            ctx_above_threshold = [p for p in item["ctxs"] if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
            if len(ctx_above_threshold) > self.top_n:
                item["ctxs"] = ctx_above_threshold
                print("after filtering -- number of ctxs: {0}".format(len(item["ctxs"])))
        # 使用BGE模型进行重排序
        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(item["input"], item["ctxs"], self.reranker, norm_cite=self.norm_cite, use_abstract=use_abstract)
        return reranked_contexts, sorted_results, id_mapping
    
    def reranking_passages_cross_encoder(self, item, batch_size=5, llama3_chat=False, task_name="default", use_abstract=False):
        # 分离title搜索和keyword搜索的论文
        title_papers = [p for p in item["ctxs"] if p.get("title_query", False)]
        keyword_papers = [p for p in item["ctxs"] if not p.get("title_query", False)]
        # 只对keyword搜索的论文应用引用数过滤
        if self.min_citation is not None:
            keyword_papers = [p for p in keyword_papers 
                            if "citation_counts" in p and p["citation_counts"] >= self.min_citation]

        # 合并论文列表，title搜索的论文放在前面
        filtered_papers = title_papers + keyword_papers
        
       # 使用BGE模型进行重排序，但仅对非title搜索的论文重排序
        if filtered_papers:
            # 对所有论文进行排序以获取分数
            reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(
                item["input"], 
                filtered_papers,
                self.reranker, 
                norm_cite=self.norm_cite, 
                use_abstract=use_abstract
            )
            
            # 重新组织结果，确保title论文在前面
            final_contexts = title_papers.copy()  # 首先添加所有title论文
            # 从重排序结果中添加非title论文，直到达到top_n
            remaining_slots = self.top_n - len(final_contexts)
            if remaining_slots > 0:
                non_title_papers = [p for p in reranked_contexts if not p.get("title_query", False)]
                final_contexts.extend(non_title_papers[:remaining_slots])
                
            return final_contexts, sorted_results, id_mapping
        
        return [], {}, {}
    
    def reranking_passages_cross_encoder_supplemental2(self, item, passages, batch_size=5, llama3_chat=False, task_name="default"):
        
        if self.min_citation is not None:
            ctx_above_threshold = [p for p in passages if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
            if len(ctx_above_threshold) > self.top_n:
                passages = ctx_above_threshold
                print("after filtering -- number of ctxs: {0}".format(len(passages)))
                
        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(item["input"], passages, self.reranker, norm_cite=False, start_index=len(item["ctxs"]))
        return reranked_contexts, sorted_results, id_mapping
    
    def reranking_passages_cross_encoder_supplemental(self, item, passages, batch_size=5, llama3_chat=False, task_name="default"):
        # 分离title搜索和keyword搜索的论文
        title_papers = [p for p in passages if p.get("title_query", False)]
        keyword_papers = [p for p in passages if not p.get("title_query", False)]
        
        if self.min_citation is not None:
            keyword_papers = [p for p in keyword_papers 
                            if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
        
        filtered_papers = title_papers + keyword_papers
        
        if filtered_papers:
            reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(
                item["input"],
                filtered_papers,
                self.reranker,
                norm_cite=False,
                start_index=len(item["ctxs"])
            )
            
            final_contexts = title_papers.copy()
            remaining_slots = self.top_n - len(final_contexts)
            if remaining_slots > 0:
                non_title_papers = [p for p in reranked_contexts if not p.get("title_query", False)]
                final_contexts.extend(non_title_papers[:remaining_slots])
                
            return final_contexts, sorted_results, id_mapping
        
        return [], {}, {}
    
    def retrieve_keywords(self, question):
        prompt = [instructions.keyword_extraction_prompt.format_map({"question": question})]
        
        if  self.client is not None:
            result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user",
                            "content": prompt[0]},
                    ],
                    temperature=0.9,
                    max_tokens=1000,
                    timeout=120
                )
            raw_output = result.choices[0].message.content
            outputs = raw_output
        
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.9,  # greedy decoding
                max_tokens=1000,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate(prompt, sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
 
        queries = raw_output.split(", ")[:3]
        queries = [query.replace("Search queries: " , "") for query in queries if len(query) > 0]
        return queries

    # Generation: Generate output based on query, passages
    def generate_response(self, item, max_tokens=3000, llama3_chat=False,  task_name="default", zero_shot=False):
        ranked_results = {}
        print("zero-shot?: {}".format(zero_shot))
        print(item["input"])
        
        if self.use_contexts is False:
            ctxs = []
            # support more task
            if task_name in instructions.task_instructions:
                if zero_shot is True:
                    input_query = instructions.task_instructions[task_name][0] + instructions.task_instructions[task_name][1] + item["input"]
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = instructions.task_instructions[task_name][0] + demonstration + instructions.task_instructions[task_name][1] + item["input"]
            if  task_name == "single_qa":
                input_query = instructions.generation_instance_prompts_w_references_single_paper_no_context.format_map({"input": item["input"]})
        else:
            ctxs = ""
            for doc_idx, doc in enumerate(item["ctxs"][:self.top_n]):
                if "title" in doc and len(doc["title"]) > 0:
                    ctxs += "[{0}] Title: {1} Text: {2}\n".format(doc_idx, doc["title"], doc["text"])
                else:
                    ctxs += "[{0}] {1}\n".format(doc_idx,  doc["text"])
            item["final_passages"] = ctxs
            
            if task_name =="summarization":
                if zero_shot is True:
                    input_query = instructions.prompts_w_references_summarization_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_summarization.format_map({"context": ctxs, "input": item["input"]})
            elif task_name == "single_qa":
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper.format_map({"context": ctxs, "input": item["input"]})
            
            elif task_name in instructions.task_instructions:
                task_instruction = instructions.task_instructions[task_name][0]
                instance_header = instructions.task_instructions[task_name][1]
                if zero_shot is True:
                    input_query = "{0}\nReferences:\n{1}\n{2}{3}".format(task_instruction, ctxs, instance_header, item["input"])
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = "{0}{1}\nReferences:\n{2}\n{3}{4}".format(task_instruction, demonstration, ctxs, instance_header, item["input"])
                    
            else:
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references.format_map({"context": ctxs, "input": item["input"]})

        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
            
        if self.client is not None: 
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=120
            )
            raw_output = result.choices[0].message.content
            outputs = raw_output
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(raw_output.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=max_tokens,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs

        if "References:" in raw_output:
            raw_output = raw_output.split("References:")[0]
        item["output"] = raw_output
        return raw_output, ctxs, cost

    # Feedback: send feedback on model' predictions.
    def process_feedback(self, response):
        feedbacks_and_questions = re.findall(r'Feedback: (.*?)(?:Question: (.*?))?\n', response)
        ratings = [(feedback.strip(), question.strip() if question else "") for feedback, question in feedbacks_and_questions]
        return ratings

    def get_feedback(self, item, llama3_chat):
        input_query = instructions.feedback_example_instance_prompt.format_map({"question": item["input"], "passages": item["final_passages"], "answer": item["output"]})
        # TODO: check if the llama3 chat format is helpful or not. 
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
        
        if self.client is not None:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            outputs = result.choices[0].message.content
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=2000,
                stop_token_ids=[128009]
            )

            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        feedbacks = self.process_feedback(raw_output)
        return feedbacks, cost

    def edit_with_feedback(self, item, feedback, max_tokens=3000, llama3_chat=False):
        input_query = instructions.editing_instance_prompt.format_map({"question": item["input"], "passages": item["final_passages"], "answer": item["output"], "feedback": feedback})
        
        # TODO: check if the llama3 chat format is helpful or not. 
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
        
        if self.client is not None: 
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            raw_output = result.choices[0].message.content
            outputs = raw_output
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=max_tokens,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        print("orig answer: {}".format( item["output"]))
        print("feedback: {}".format(feedback))
        print("updated answer: {}".format(raw_output))
        return raw_output, cost

    def edit_with_feedback_retrieval(self, item, feedback, passages, passage_start_index, max_tokens=2000, llama3_chat=False):
        processed_passages = ""
        for doc_idx, doc in enumerate(passages[:self.top_n]):
            if "title" in doc and len(doc["title"]) > 0:
                processed_passages += "[{0}] Title: {1} Text: {2}\n".format(passage_start_index+doc_idx, doc["title"], doc["text"])
            else:
                processed_passages += "[{0}] {1}\n".format(passage_start_index+doc_idx + len(item["ctxs"]), doc["text"])

        input_query = instructions.editing_with_retrieval_instance_prompt.format_map({"question": item["input"], "retrieved_passages": processed_passages, "answer": item["output"], "feedback": feedback})
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
                
        if self.client is not None: 
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=3000,
            )
            raw_output = result.choices[0].message.content
            outputs = raw_output
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=3000,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0]  for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        return raw_output, cost

    def insert_attributions_posthoc_paragraph(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"] 
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        print(text)
        sentences = text.split("\n")
        print(sentences)
        # post process sentences 
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if len(updated_sentences) > 0 and len(statement) > 0 and statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                # cases where citations are included
                if "[" in statement or (s_index < len(sentences) - 1 and len(sentences[s_index+1]) > 0 and sentences[s_index+1][0] == "["):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
            print("{0} sentences require attributions, e..g, {1}".format(len(post_hoc_sentence), list(post_hoc_sentence.values())[0] ))
            prompts = []
            for s in list(post_hoc_sentence.values()):    
                input_query = instructions.posthoc_attributions_paragraph.format_map({"statement": s, "passages": passages})

                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)
                
                prompts.append(input_query)
            
            if self.client is not None: 
                outputs = []
                for input_query in prompts:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user",
                                "content": input_query},
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    raw_output = result.choices[0].message.content
                    outputs.append(raw_output)
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=0.7,  # greedy decoding
                    max_tokens=2000,
                    stop_token_ids=[128009]
                )
                outputs = self.model.generate(prompts, sampling_params)
                outputs = [it.outputs[0].text for it in outputs]
            
            # Postprocess Output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                    post_hoc_sentence[sentence_key] = processed_output
                
            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs
            
        return "\n".join(updated_sentences)
    
    def insert_attributions_posthoc(self, item, llama3_chat=False):
        text = item["output"]
        passages = item["final_passages"]

        sentences = sent_tokenize(text)
        # post process sentences 
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if statement[0] == "[":
                    updated_sentences[-1]  = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                # cases where citations are included
                if "[" in statement or (s_index < len(sentences) - 1 and sentences[s_index+1][0] =="["):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
                        
            print("{0} sentences require attributions, e..g, {1}".format(len(post_hoc_sentence), list(post_hoc_sentence.values())[0] ))
            prompts = []
            for s in list(post_hoc_sentence.values()):    
                input_query = instructions.posthoc_attributions.format_map({"statement": s, "passages": passages})

                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)
                
                prompts.append(input_query)
            
            if self.client is not None: 
                outputs = []
                for input_query in prompts:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user",
                                "content": input_query},
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    raw_output = result.choices[0].message.content
                    outputs.append(raw_output)
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=0.7,  # greedy decoding
                    max_tokens=2000,
                    stop_token_ids=[128009]
                )
                outputs = self.model.generate(prompts, sampling_params)
                outputs = [it.outputs[0].text for it in outputs]
            
            # process_output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                    post_hoc_sentence[sentence_key] = processed_output
                
            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs
            
        return " ".join(updated_sentences)

    def insert_attributions_posthoc_paragraph_all(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"] 
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        sentences = text.split("\n")
        print(sentences)
        updated_sentences = []
        post_hoc_sentence = {}
        prompts = []

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if len(updated_sentences) > 0 and len(statement) > 0 and statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                updated_sentences.append("[replace_{}]".format(s_index))
                post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        for s in list(post_hoc_sentence.values()):    
            input_query = instructions.posthoc_attributions_paragraph_all.format_map({"statement": s, "passages": passages})

            if llama3_chat is True:
                input_query = create_prompt_with_llama3_format(input_query)
            
            prompts.append(input_query)
        
        if self.client is not None: 
            outputs = []
            cost = 0
            for input_query in prompts:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user",
                            "content": input_query},
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                )
                raw_output = result.choices[0].message.content
                outputs.append(raw_output)
                cost += calculate_openai_api_cost(len(input_query.split(" ")),len(raw_output.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,
                max_tokens=1000,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
            cost = 0
        
        # process_output
        for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
            if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
            else:
                processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                post_hoc_sentence[sentence_key] = processed_output
            
        final_processed_outputs = []
        for item in updated_sentences:
            if item in post_hoc_sentence:
                final_processed_outputs.append(post_hoc_sentence[item])
            else:
                final_processed_outputs.append(item)
        updated_sentences = final_processed_outputs
        
        return "\n".join(updated_sentences), cost

    def run(self, item, ranking_ce=False, use_feedback=False, skip_generation=False, posthoc_at=False, llama3_chat=False, task_name="default", zero_shot=False, max_per_paper=None, use_abstract=False, max_tokens=3000):
        print("llama3 chat format? {0}".format(llama3_chat))
        print("use feedback: {}".format(use_feedback))
        total_cost = 0
            
        if ranking_ce is True:
            # item["ctxs"], ranked_results, id_mapping = self.reranking_passages_cross_encoder(item, batch_size=1, llama3_chat=llama3_chat, task_name=task_name, use_abstract=False)
            item["ctxs"], ranked_results, id_mapping = self.reranking_passages_cross_encoder(
                item, 
                batch_size=1, 
                llama3_chat=llama3_chat, 
                task_name=task_name, 
                use_abstract=use_abstract
            ) # 确保title的论文被保留
            item["ranked_results"] = ranked_results
            item["id_mapping"] = id_mapping
            

        if max_per_paper is not None:
            filtered_ctxs = []
            title_to_count = {}
            for ctx in item["ctxs"]:
                if "title" not in ctx or ctx["title"] is None:
                    ctx["title"] = ""
                title_to_count.setdefault(ctx["title"], 0)
                if title_to_count[ctx["title"]] > max_per_paper:
                    # print("We have already aded the paper {0} {1} times".format(ctx["title"], max_per_paper))
                    continue
                else:
                    filtered_ctxs.append(ctx)
                    title_to_count[ctx["title"]] += 1
                    
            item["ctxs"] = filtered_ctxs
            
        if skip_generation is False:
            generated_result, passages, gen_cost = self.generate_response(item, max_tokens=max_tokens, llama3_chat=llama3_chat, task_name=task_name, zero_shot=zero_shot)
            if "\n\n References":
                generated_result = generated_result.split("\n\n References")[0]
            item["initial_result"] = generated_result
            total_cost += gen_cost

        if use_feedback is True:
            print("generating feedback")
            feedbacks, feedback_cost = self.get_feedback(item, llama3_chat=llama3_chat)[:3]
            total_cost += feedback_cost
            item["feedbacks"] = feedbacks
            for feedback_idx, feedback in tqdm(enumerate(feedbacks[:3])):
                # currently only supports non retrieval feedback
                if len(feedback[1]) == 0:
                    edited_answer, edited_cost = self.edit_with_feedback(item, feedback[0], llama3_chat=llama3_chat)
                    if "Here is the revised answer:\n\n" in edited_answer:
                        edited_answer = edited_answer.split("Here is the revised answer:\n\n")[1]
                    total_cost += edited_cost
                    if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                        item["output"] = edited_answer
                        item["edited_answer_{}".format(feedback_idx)] = edited_answer
                    else:
                        print("skipping as edited answers got too short")
                else:
                    new_papers = []
                    # new_papers = retrieve_pes2o_passages(feedback[1], 20, "pes2o")
                    print("web searched papers: {}".format(len(new_papers)))
                    if self.ss_retriever is True:
                        new_keywords = self.retrieve_keywords(feedback[1])
                        paper_list = {}
                        if len(new_keywords) > 0:
                            for keyword in new_keywords:    
                                top_papers = search_paper_via_query(keyword)
                                print(top_papers)
                                if top_papers is None:
                                    print(keyword)
                                else:
                                    for paper in top_papers:
                                        if paper["paperId"] not in paper_list:
                                            paper["text"] = paper["abstract"]
                                            paper["citation_counts"] = paper["citationCount"]
                                            paper_list[paper["paperId"]] = paper
                            new_papers += list(paper_list.values())
                            # remove duplicarted data 
                    if len(new_papers) > 0:
                        print("before deduplication: {}".format(len(new_papers)))
                        new_papers_dicts = {paper["text"][:100] + paper["title"]: paper for paper in new_papers if paper is not None and type(paper["text"]) is str}
                        new_papers = list(new_papers_dicts.values())
                        print("after deduplication: {}".format(len(new_papers)))
                        # add new papers when and only when we have the new papers. 
                        if len(new_papers) > 0:
                            new_passages_reranked, _ , _  = self.reranking_passages_cross_encoder_supplemental(item, new_papers, batch_size=10, llama3_chat=llama3_chat, task_name=task_name)
                            passages_start_index = len(item["ctxs"])

                            edited_answer, edited_cost = self.edit_with_feedback_retrieval(item, feedback[0], new_passages_reranked, passages_start_index)
                            total_cost += edited_cost
                            if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                                item["ctxs"] += new_passages_reranked[:self.top_n]
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                item["output"] = edited_answer
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                            elif len(item["output"]) == 0 and len(edited_answer) > 0:
                                item["ctxs"] += new_passages_reranked[:self.top_n]
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                item["output"] = edited_answer
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                            else:
                                print("skipping as edited answers got too short")

        if posthoc_at is True:
            # attributed_results = self.insert_attributions_posthoc(item, llama3_chat=llama3_chat)
            # attributed_results = self.insert_attributions_posthoc_paragraph(item, llama3_chat=llama3_chat)
            attributed_results, attributed_cost =  self.insert_attributions_posthoc_paragraph_all(item, llama3_chat=llama3_chat)
            total_cost += attributed_cost
            item["output"] = attributed_results
        
        item["output"] = item["output"].replace("[Response_Start]", "").replace("[Response_End]", "")

        print(item["output"])

        if "\n### References" in item["output"]:
            item["output"] = item["output"].split("\n### References")[0]
        return item, total_cost

    def run_batch(self, items, batch_size=4, ranking_ce=False, use_feedback=False, skip_generation=False, 
            posthoc_at=False, llama3_chat=False, task_name="default", zero_shot=False, 
            max_per_paper=None, use_abstract=False, max_tokens=3000):
        """
        批量处理多个查询项
        
        Args:
            items: List[Dict] - 输入项列表
            batch_size: int - 批处理大小
        """
        print(f"Processing batch of {len(items)} items with batch_size {batch_size}")
        total_costs = []
        processed_items = []

        # 1. 批量进行文档重排序
        if ranking_ce:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_contexts = []
                for item in batch:
                    contexts, ranked_results, id_mapping = self.reranking_passages_cross_encoder(
                        item,
                        batch_size=1,
                        llama3_chat=llama3_chat,
                        task_name=task_name,
                        use_abstract=use_abstract
                    )
                    random.shuffle(contexts)
                    item["ctxs"] = contexts
                    item["ranked_results"] = ranked_results
                    item["id_mapping"] = id_mapping
                    batch_contexts.append(contexts)

        # 2. 对每个项处理最大论文数限制
        if max_per_paper is not None:
            for item in items:
                filtered_ctxs = []
                title_to_count = {}
                for ctx in item["ctxs"]:
                    if "title" not in ctx or ctx["title"] is None:
                        ctx["title"] = ""
                    title_to_count.setdefault(ctx["title"], 0)
                    if title_to_count[ctx["title"]] > max_per_paper:
                        continue
                    else:
                        filtered_ctxs.append(ctx)
                        title_to_count[ctx["title"]] += 1
                item["ctxs"] = filtered_ctxs

        # 3. 批量生成响应
        if not skip_generation:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                prompts = []
                for item in batch:
                    if self.use_contexts is False:
                        ctxs = []
                        if task_name in instructions.task_instructions:
                            if zero_shot:
                                input_query = instructions.task_instructions[task_name][0] + instructions.task_instructions[task_name][1] + item["input"]
                            else:
                                demonstration = instructions.demonstrations[task_name]
                                input_query = instructions.task_instructions[task_name][0] + demonstration + instructions.task_instructions[task_name][1] + item["input"]
                        if task_name == "single_qa":
                            input_query = instructions.generation_instance_prompts_w_references_single_paper_no_context.format_map({"input": item["input"]})
                    else:
                        ctxs = ""
                        for doc_idx, doc in enumerate(item["ctxs"][:self.top_n]):
                            if "title" in doc and len(doc["title"]) > 0:
                                ctxs += "[{0}] Title: {1} Text: {2}\n".format(doc_idx, doc["title"], doc["text"])
                            else:
                                ctxs += "[{0}] {1}\n".format(doc_idx, doc["text"])
                        item["final_passages"] = ctxs
                        
                        if task_name == "summarization":
                            if zero_shot:
                                input_query = instructions.prompts_w_references_summarization_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                            else:
                                input_query = instructions.generation_instance_prompts_summarization.format_map({"context": ctxs, "input": item["input"]})
                        elif task_name == "single_qa":
                            if zero_shot:
                                input_query = instructions.generation_instance_prompts_w_references_single_paper_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                            else:
                                input_query = instructions.generation_instance_prompts_w_references_single_paper.format_map({"context": ctxs, "input": item["input"]})
                        elif task_name in instructions.task_instructions:
                            task_instruction = instructions.task_instructions[task_name][0]
                            instance_header = instructions.task_instructions[task_name][1]
                            if zero_shot:
                                input_query = "{0}\nReferences:\n{1}\n{2}{3}".format(task_instruction, ctxs, instance_header, item["input"])
                            else:
                                demonstration = instructions.demonstrations[task_name]
                                input_query = "{0}{1}\nReferences:\n{2}\n{3}{4}".format(task_instruction, demonstration, ctxs, instance_header, item["input"])
                        else:
                            if zero_shot:
                                input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                            else:
                                input_query = instructions.generation_instance_prompts_w_references.format_map({"context": ctxs, "input": item["input"]})

                    if llama3_chat:
                        input_query = create_prompt_with_llama3_format(input_query)
                    prompts.append(input_query)

                # 批量生成
                if self.client is not None:
                    # 处理OpenAI API
                    outputs = []
                    gen_costs = []
                    for prompt in prompts:

                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=max_tokens,
                            stream=True,
                            timeout=300
                        )

                        full_response = ""
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content

                        outputs.append(full_response)
                        gen_costs.append(calculate_openai_api_cost(len(prompt.split()), len(outputs[-1].split()), self.model_name))
                else:
                    # 使用vLLM批量处理
                    sampling_params = vllm.SamplingParams(
                        temperature=0.4,
                        max_tokens=max_tokens,
                        stop_token_ids=[128009]
                    )
                    outputs = self.model.generate(prompts, sampling_params)
                    outputs = [it.outputs[0].text for it in outputs]
                    gen_costs = [0] * len(outputs)

                # 处理输出
                for item, output, cost in zip(batch, outputs, gen_costs):
                    raw_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in output else output
                    if "References:" in raw_output:
                        raw_output = raw_output.split("References:")[0]
                    item["initial_result"] = raw_output
                    item["output"] = raw_output
                    item["total_cost"] = cost

        # 4. 批量处理反馈（如果需要）
        if use_feedback:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                for item in batch:
                    feedbacks, feedback_cost = self.get_feedback(item, llama3_chat=llama3_chat)[:3]
                    item["total_cost"] += feedback_cost
                    item["feedbacks"] = feedbacks
                    
                    for feedback_idx, feedback in enumerate(feedbacks[:3]):
                        if not feedback[1]:  # 无需检索的反馈
                            edited_answer, edited_cost = self.edit_with_feedback(item, feedback[0], llama3_chat=llama3_chat)
                            item["total_cost"] += edited_cost
                            if "Here is the revised answer:\n\n" in edited_answer:
                                edited_answer = edited_answer.split("Here is the revised answer:\n\n")[1]
                            
                            if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                                item["output"] = edited_answer
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                            else:
                                print("跳过由于编辑后答案太短")

        # 5. 批量处理后处理属性（如果需要）
        if posthoc_at:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                prompts = []
                for item in batch:
                    attributed_results, attributed_cost = self.insert_attributions_posthoc_paragraph_all(item, llama3_chat=llama3_chat)
                    item["total_cost"] += attributed_cost
                    item["output"] = attributed_results.replace("[Response_Start]", "").replace("[Response_End]", "")
                    if "\n### References" in item["output"]:
                        item["output"] = item["output"].split("\n### References")[0]

        # 返回处理后的结果
        return items, [item.get("total_cost", 0) for item in items]

def process_paragraph(text):
    text = text.replace("<cit.>", "")
    text = remove_citations(text)
    return text

def process_input_data2(data, use_contexts=True):
    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]

        new_ctxs = []
        if use_contexts is True:
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs

            # remove duplicated contexts
            processed_paras = []
            for ctx in tqdm(item["ctxs"]):
                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]
                if ctx["text"] is None or len(ctx["text"]) ==0:
                    continue
                if type(ctx["text"]) != str:
                    ctx["text"] = " ".join(ctx["text"]["contexts"])
                ctx["text"] = process_paragraph(ctx["text"])
                if "title" not in ctx:
                    ctx["title"] = ""
                processed_paras.append(ctx)

            processed_paras_dicts = {paper["text"][:100] + paper["title"]: paper for paper in processed_paras}
            processed_paras = list(processed_paras_dicts.values())

            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
        processed_data.append(item)
    return processed_data

def process_input_data(data, use_contexts=True):
    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]
                
        if use_contexts is True:
            # 分离title查询和keyword查询的论文
            title_papers = []
            keyword_papers = []
            new_ctxs = []
            
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs
            
            # 分别处理两类论文
            for ctx in tqdm(item["ctxs"]):
                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]
                if ctx["text"] is None or len(ctx["text"]) == 0:
                    continue
                if type(ctx["text"]) != str:
                    ctx["text"] = " ".join(ctx["text"]["contexts"])
                ctx["text"] = process_paragraph(ctx["text"])
                if "title" not in ctx:
                    ctx["title"] = ""
                
                # 根据标记分类
                if ctx.get("title_query", False):
                    title_papers.append(ctx)
                else:
                    keyword_papers.append(ctx)
            
            # 分别去重
            # 处理title论文
            title_papers_dict = {
                paper["text"][:100] + paper["title"]: paper 
                for paper in title_papers
            }
            processed_title_papers = list(title_papers_dict.values())
            
            # 处理keyword论文
            keyword_papers_dict = {
                paper["text"][:100] + paper["title"]: paper 
                for paper in keyword_papers
            }
            processed_keyword_papers = list(keyword_papers_dict.values())
            
            # 合并结果，确保title论文在前
            processed_paras = processed_title_papers + processed_keyword_papers
            
            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
            
        processed_data.append(item)
    return processed_data