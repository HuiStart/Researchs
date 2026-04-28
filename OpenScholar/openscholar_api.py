import multiprocessing
import sys
import os
import argparse
import torch
from FlagEmbedding import FlagReranker
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from openai import OpenAI

multiprocessing.set_start_method('spawn')
from src.open_scholar import OpenScholar
from src.open_scholar import process_input_data
from src.use_search_apis import search_semantic_scholar

'''
    功能与作用: Flask API 服务入口。
    对外暴露batch_ask接口，接收问题列表
    内部流程
        1.调用小模型，生成 Semantic Scholar 搜索关键词
        2.调用 Semantic Scholar API
        3.通过 OpenScholar，llm（Llama-3.1_OpenScholar-8B）生成带引用的学术回答
        4.返回结果，关键词 和 费用
    
'''
class OpenScholarAPI:
    def __init__(self, config):
        self.config = config
        self.app = Flask(__name__)
        self.client = None
        self.client2 = None
        self.open_scholar = None
        self.reranker = None
        
        self.initialize_models()
        self.setup_routes()
    
        def initialize_models(self):        """Initialize all required models"""
        # Initialize OpenAI clients for model inference
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=f"http://127.0.0.1:{self.config.small_model_port}/v1"
        )
        
        self.client2 = OpenAI(
            api_key=self.config.api_key,
            base_url=f"http://127.0.0.1:{self.config.large_model_port}/v1",
            timeout=120
        )
        
        # Initialize reranker
        self.reranker = FlagReranker(self.config.reranker_path, use_fp16=True)
        
        # Initialize OpenScholar
        self.open_scholar = OpenScholar(
            model=None,
            tokenizer=None,
            client=self.client2,
            api_model_name=self.config.large_model_name,
            use_contexts=True,
            top_n=self.config.top_n,
            reranker=self.reranker,
            min_citation=None,
            norm_cite=False,
            ss_retriever=True
        )
    
        def process_batch(self, questions, titles, batch_size):        """Batch process questions and generate search keywords"""
        all_prompts = []
        for question in questions:
            search_prompt = """
                Suggest semantic scholar search APIs to retrieve relevant papers to answer the following question related to the most recent NLP research. The search queries must be short, and commma separated. Here's an example. I'll show one example and the test instance you should suggest the search queries. \n
                ##\n
                Question: How have prior work incorporated personality attributes to train personalized dialogue generation models?\n
                Search queries: personalized dialogue generation, personalized language models, personalized dialogue\n
                ##\n
                Question: How do retrieval-augmented LMs perform well in knowledge-intensive tasks?\n
                Search queries: retrieval-augmented LMs, knowledge-intensive tasks, large language models for knowledge-intensive tasks, retrieval-augmented generation
                ##\n
                Question: {question}\n
                Search queries:"""
            all_prompts.append(search_prompt.format(question=question))
        
        all_keywords = []
        for i in range(0, len(all_prompts), 1):
            outputs = self.client.completions.create(
                model=self.config.small_model_name,
                prompt=all_prompts[i],
                n=4,
                temperature=0.6,
                max_tokens=1000,
                stop=['\n']
            )
            for idx in range(len(outputs.choices)):
                chosen = outputs.choices[idx].text
                if 'Search queries:' in chosen:
                    chosen = chosen.split('Search queries:')[1]
                elif 'Search queries' in chosen:
                    chosen = chosen.split('Search queries')[1]
                elif 'search queries' in chosen:
                    chosen = chosen.split('search queries')[1]
                if len(chosen.split(',')) > 3:
                    break
            batch_keywords = [chosen.split(',')]
            print('key words: ', batch_keywords)
            all_keywords.extend(batch_keywords)
        
        return all_keywords
    
    def setup_routes(self):
        @self.app.route('/batch_ask', methods=['POST'])
        def batch_ask_questions():
            import time
            start = time.time()
            data = request.json
            questions = data.get('questions', [])
            titles = data.get('titles', [])
            all_keywords = data.get('keywords', [])
            
            if not questions:
                return jsonify({"error": "No questions provided"}), 400
            
                        if not titles:                titles = [None] * len(questions)                        # 1. Batch generate search keywords            all_keywords = self.process_batch(questions, titles, self.config.search_batch_size)                        # 2. Prepare OpenScholar input data            input_items = []            for question, keywords, title_list in zip(questions, all_keywords, titles):                # Search papers                keyword_papers, title_papers, _ = search_semantic_scholar(                    question,                    new_keywords=[keywords],                    new_titles=title_list,                    s2_api_key=self.config.s2_api_key                )                                # Merge and mark paper sources                retrieved_papers = []                for paper in title_papers:                    paper["title_query"] = True                    retrieved_papers.append(paper)                for paper in keyword_papers:                    paper["title_query"] = False                    retrieved_papers.append(paper)                                item = {                    "input": question,                    "ctxs": retrieved_papers                }                input_items.append(item)                        # 3. Process input data            processed_data = process_input_data(input_items, use_contexts=True)                        # 4. Batch process requests            response_items, total_costs = self.open_scholar.run_batch(                processed_data,                batch_size=self.config.scholar_batch_size,                ranking_ce=True,                use_feedback=False,                skip_generation=False,                posthoc_at=False,                llama3_chat=True,                task_name="default",                zero_shot=True,                max_tokens=self.config.max_tokens            )                        # 5. Prepare return results
            results = []
            for item, cost, keywords in zip(response_items, total_costs, all_keywords):
                result = {
                    "final_passages": item.get("final_passages", ""),
                    "output": item.get("output", ""),
                    "total_cost": cost,
                    "keywords": keywords
                }
                results.append(result)
            
            return jsonify({"results": results})
    
    def run(self):
        multiprocessing.freeze_support()
        self.app.run(host='0.0.0.0', port=self.config.api_port)

class Config:
    def __init__(self):
        # ===== API Key Configuration =====
        # Semantic Scholar API Key (Required)
        # Apply at: https://www.semanticscholar.org/product/api
        self.S2_API_KEY = "YOUR_SEMANTIC_SCHOLAR_API_KEY"
        
        # vLLM API Key (can be any string, used for API calls)
        self.API_KEY = "sk-your-api-key-here"
        
        # ===== Model Configuration =====
        # Large model name (for main inference)
        self.LARGE_MODEL_NAME = "OpenSciLM/Llama-3.1_OpenScholar-8B"
        
        # Small model name (for search keyword generation)
        self.SMALL_MODEL_NAME = "Qwen/Qwen3-0.6B"
        
        # Reranker model path (please replace with your actual path)
        self.RERANKER_PATH = "OpenSciLM/OpenScholar_Reranker"
        
        # ===== Port Configuration =====
        # Large model service port
        self.LARGE_MODEL_PORT = 38011
        
        # Small model service port
        self.SMALL_MODEL_PORT = 38014
        
        # API service port
        self.API_PORT = 38015
        
        # ===== Processing Parameters =====
        # Search generation batch size
        self.SEARCH_BATCH_SIZE = 100
        
        # OpenScholar processing batch size
        self.SCHOLAR_BATCH_SIZE = 100
        
        # Number of papers to retrieve
        self.TOP_N = 10
        
        # Maximum tokens for generation
        self.MAX_TOKENS = 3000
        
        # ===== GPU Configuration =====
        # CUDA device setting (optional)
        self.CUDA_VISIBLE_DEVICES = "0"
        
        # Large model GPU memory utilization
        self.LARGE_MODEL_GPU_MEMORY_UTILIZATION = 0.8
        
        # Small model GPU memory utilization
        self.SMALL_MODEL_GPU_MEMORY_UTILIZATION = 0.05
        
        # Large model maximum length
        self.LARGE_MODEL_MAX_LEN = 70000
        
        # Small model maximum length
        self.SMALL_MODEL_MAX_LEN = 10000

def parse_args():
    parser = argparse.ArgumentParser(description='OpenScholar API Server')
    parser.add_argument('--api_key', type=str, default='YOUR_API_KEY_HERE', 
                        help='API key for model inference')
    parser.add_argument('--s2_api_key', type=str, default='YOUR_SEMANTIC_SCHOLAR_API_KEY',
                        help='Semantic Scholar API key')
    parser.add_argument('--large_model_port', type=int, default=38011,
                        help='Port for large model server')
    parser.add_argument('--small_model_port', type=int, default=38014,
                        help='Port for small model server')
    parser.add_argument('--api_port', type=int, default=38015,
                        help='Port for API server')
    parser.add_argument('--reranker_path', type=str, default='OpenSciLM/OpenScholar_Reranker',
                        help='Path to reranker model')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Top N papers to retrieve')
    parser.add_argument('--max_tokens', type=int, default=3000,
                        help='Maximum tokens for generation')
    parser.add_argument('--search_batch_size', type=int, default=100,
                        help='Batch size for search generation')
    parser.add_argument('--scholar_batch_size', type=int, default=100,
                        help='Batch size for OpenScholar processing')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 创建配置对象
    config = Config()
    config.api_key = args.api_key
    config.s2_api_key = args.s2_api_key
    config.large_model_port = args.large_model_port
    config.small_model_port = args.small_model_port
    config.api_port = args.api_port
    config.reranker_path = args.reranker_path
    config.top_n = args.top_n
    config.max_tokens = args.max_tokens
    config.search_batch_size = args.search_batch_size
    config.scholar_batch_size = args.scholar_batch_size
    
    # 启动API服务
    api_server = OpenScholarAPI(config)
    api_server.run() 
