import argparse
import multiprocessing
import time

from FlagEmbedding import FlagReranker
from flask import Flask, jsonify, request
from openai import OpenAI

from src import use_search_apis
from src.open_scholar import OpenScholar, process_input_data


try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # The start method may already be set when the module is imported by tests.
    pass


class OpenScholarAPI:
    """Flask API wrapper for OpenScholar batch academic Q&A."""

    def __init__(self, config):
        self.config = config
        self.app = Flask(__name__)
        self.client = None
        self.client2 = None
        self.open_scholar = None
        self.reranker = None

        self.initialize_models()
        self.setup_routes()

    def initialize_models(self):
        """Initialize local OpenAI-compatible model clients and reranker."""
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=f"http://127.0.0.1:{self.config.small_model_port}/v1",
        )

        self.client2 = OpenAI(
            api_key=self.config.api_key,
            base_url=f"http://127.0.0.1:{self.config.large_model_port}/v1",
            timeout=120,
        )

        self.reranker = FlagReranker(self.config.reranker_path, use_fp16=True)
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
            ss_retriever=True,
        )

    def process_batch(self, questions, batch_size):
        """Generate Semantic Scholar keyword queries for each question."""
        all_keywords = []
        for i in range(0, len(questions), batch_size):
            for question in questions[i : i + batch_size]:
                search_prompt = """
Suggest semantic scholar search APIs to retrieve relevant papers to answer the following question related to the most recent NLP research. The search queries must be short, and commma separated. Here's an example. I'll show one example and the test instance you should suggest the search queries.
##
Question: How have prior work incorporated personality attributes to train personalized dialogue generation models?
Search queries: personalized dialogue generation, personalized language models, personalized dialogue
##
Question: How do retrieval-augmented LMs perform well in knowledge-intensive tasks?
Search queries: retrieval-augmented LMs, knowledge-intensive tasks, large language models for knowledge-intensive tasks, retrieval-augmented generation
##
Question: {question}
Search queries:""".format(
                    question=question
                )

                outputs = self.client.completions.create(
                    model=self.config.small_model_name,
                    prompt=search_prompt,
                    n=4,
                    temperature=0.6,
                    max_tokens=1000,
                    stop=["\n"],
                )

                chosen = outputs.choices[0].text
                for choice in outputs.choices:
                    candidate = choice.text
                    for marker in ["Search queries:", "Search queries", "search queries"]:
                        if marker in candidate:
                            candidate = candidate.split(marker, 1)[1]
                            break
                    if len([part for part in candidate.split(",") if part.strip()]) >= 3:
                        chosen = candidate
                        break

                keywords = [part.strip() for part in chosen.split(",") if part.strip()]
                all_keywords.append(keywords[:5])
                print("keywords:", keywords[:5])

        return all_keywords

    @staticmethod
    def _normalize_titles(titles, question_count):
        if not titles:
            return [None] * question_count

        if question_count == 1 and all(isinstance(item, str) for item in titles):
            return [titles]

        normalized = []
        for item in titles[:question_count]:
            if item is None:
                normalized.append(None)
            elif isinstance(item, str):
                normalized.append([item])
            else:
                normalized.append(item)

        while len(normalized) < question_count:
            normalized.append(None)
        return normalized

    @staticmethod
    def _normalize_keywords(keywords, question_count):
        if not keywords:
            return None
        if question_count == 1 and all(isinstance(item, str) for item in keywords):
            return [keywords]
        return keywords

    def setup_routes(self):
        @self.app.route("/", methods=["GET"])
        def health_check():
            return jsonify({"status": "ok"})

        @self.app.route("/batch_ask", methods=["POST"])
        def batch_ask_questions():
            start = time.time()
            data = request.get_json(force=True) or {}
            questions = data.get("questions", [])
            titles = self._normalize_titles(data.get("titles", []), len(questions))
            provided_keywords = self._normalize_keywords(data.get("keywords", []), len(questions))

            if not questions:
                return jsonify({"error": "No questions provided"}), 400

            all_keywords = provided_keywords or self.process_batch(
                questions, self.config.search_batch_size
            )

            # Ensure the configured key is used instead of the module placeholder.
            use_search_apis.S2_API_KEY = self.config.s2_api_key

            input_items = []
            for question, keywords, title_list in zip(questions, all_keywords, titles):
                keyword_papers, title_papers, _ = use_search_apis.search_semantic_scholar(
                    question,
                    new_keywords=keywords,
                    new_titles=title_list,
                )

                retrieved_papers = []
                for paper in title_papers:
                    paper["title_query"] = True
                    retrieved_papers.append(paper)
                for paper in keyword_papers:
                    paper["title_query"] = False
                    retrieved_papers.append(paper)

                input_items.append({"input": question, "ctxs": retrieved_papers})

            processed_data = process_input_data(input_items, use_contexts=True)
            response_items, total_costs = self.open_scholar.run_batch(
                processed_data,
                batch_size=self.config.scholar_batch_size,
                ranking_ce=True,
                use_feedback=False,
                skip_generation=False,
                posthoc_at=False,
                llama3_chat=True,
                task_name="default",
                zero_shot=True,
                max_tokens=self.config.max_tokens,
            )

            results = []
            for item, cost, keywords in zip(response_items, total_costs, all_keywords):
                results.append(
                    {
                        "final_passages": item.get("final_passages", ""),
                        "output": item.get("output", ""),
                        "total_cost": cost,
                        "keywords": keywords,
                    }
                )

            return jsonify({"elapsed_seconds": time.time() - start, "results": results})

    def run(self):
        multiprocessing.freeze_support()
        self.app.run(host="0.0.0.0", port=self.config.api_port)


class Config:
    def __init__(self):
        self.api_key = "sk-your-api-key-here"
        self.s2_api_key = "YOUR_SEMANTIC_SCHOLAR_API_KEY"

        self.large_model_name = "OpenSciLM/Llama-3.1_OpenScholar-8B"
        self.small_model_name = "Qwen/Qwen3-0.6B"
        self.reranker_path = "OpenSciLM/OpenScholar_Reranker"

        self.large_model_port = 38011
        self.small_model_port = 38014
        self.api_port = 38015

        self.search_batch_size = 100
        self.scholar_batch_size = 100
        self.top_n = 10
        self.max_tokens = 3000


def parse_args():
    parser = argparse.ArgumentParser(description="OpenScholar API Server")
    parser.add_argument("--api_key", type=str, default="YOUR_API_KEY_HERE")
    parser.add_argument("--s2_api_key", type=str, default="YOUR_SEMANTIC_SCHOLAR_API_KEY")
    parser.add_argument("--large_model_port", type=int, default=38011)
    parser.add_argument("--small_model_port", type=int, default=38014)
    parser.add_argument("--api_port", type=int, default=38015)
    parser.add_argument("--reranker_path", type=str, default="OpenSciLM/OpenScholar_Reranker")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--search_batch_size", type=int, default=100)
    parser.add_argument("--scholar_batch_size", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    OpenScholarAPI(config).run()
