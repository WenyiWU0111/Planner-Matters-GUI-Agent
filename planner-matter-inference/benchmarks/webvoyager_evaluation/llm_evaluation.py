import argparse
import json
import os
import pathlib
import sys

path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))

from agent.llm_config import create_direct_vllm_model

from evaluator import LLMEvaluator


class LLMEvaluation:
    def __init__(self, html_folder: str, args: argparse.Namespace):
        self.llm_client = create_direct_vllm_model(args, model_name=args.model)
        self.html_folder = html_folder
        self.config_folder = args.config_folder
        self.evaluator = LLMEvaluator(self.llm_client)
        llm_eval_path = os.path.join(self.html_folder, "llm_evaluation.json")
        if os.path.exists(llm_eval_path):
            with open(llm_eval_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                self.seen_task_ids = {item["task_id"] for item in results}
        else:
            self.seen_task_ids = set()

    def evaluate(self) -> None:
        results = []
        all_scores = []
        llm_eval_path = os.path.join(self.html_folder, "llm_evaluation.json")
        if self.seen_task_ids:
            with open(llm_eval_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                all_scores = [item["score"] for item in results]

        for config_file in os.listdir(self.config_folder):
            config_path = os.path.join(self.config_folder, config_file)
            if not os.path.isfile(config_path) or not config_path.endswith(".json"):
                continue
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            task_id = config.get("task_id", "")
            if task_id in self.seen_task_ids:
                print(f"Task {task_id} already evaluated")
                continue
            self.seen_task_ids.add(task_id)
            html_file = os.path.join(self.html_folder, f"render_{task_id}.html")
            print(f"Evaluating task {task_id} with html file {html_file}")
            if not os.path.exists(html_file):
                continue
            score, answer_text, ori_answer = self.evaluator(config_path, self.html_folder)
            all_scores.append(score)
            results.append({
                "task_id": task_id,
                "score": score,
                "answer_text": answer_text,
                "ori_answer": ori_answer,
            })
            print(f"Task {task_id} score: {score}")
            with open(llm_eval_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

        print("html_folder:", self.html_folder)
        if all_scores:
            print(f"Average score: {sum(all_scores) / len(all_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument(
        "--html_folder",
        type=str,
        default=os.environ.get("HTML_FOLDER", "results/webvoyager/qwen2.5-vl/test/test"),
    )
    parser.add_argument("--model", type=str, default="qwen2.5-vl-32b")
    parser.add_argument(
        "--config_folder",
        type=str,
        default=os.environ.get("CONFIG_FOLDER", "webvoyager_evaluation/data/test"),
    )
    args = parser.parse_args()
    llm_evaluation = LLMEvaluation(args.html_folder, args)
    llm_evaluation.evaluate()
