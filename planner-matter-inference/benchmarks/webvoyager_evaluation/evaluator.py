"""Simplified evaluation system for GUI Agent"""
import base64
import json
import os
import re
import sys
from pathlib import Path
from io import BytesIO

from PIL import Image

_project_root = os.environ.get(
    "INFERENCE_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from browser_env import Action, Trajectory, StateInfo


class Evaluator:
    """Base class for evaluation."""

    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page,
        client,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        if not trajectory or not isinstance(trajectory[-1], dict):
            raise ValueError("The last element of trajectory should be an action")
        return trajectory[-1]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        if len(trajectory) < 2 or not isinstance(trajectory[-2], dict):
            raise ValueError("The second last element of trajectory should be a state")
        return trajectory[-2]


class LLMEvaluator(Evaluator):
    """Determine task success/failure from render HTML (last 5 screenshots + answer)."""

    def __init__(self, vllm_client=None):
        super().__init__()
        self.vllm_client = vllm_client

    def __call__(
        self,
        config_file: Path | str,
        html_folder: Path | str,
    ) -> tuple[float, str, str]:
        with open(config_file, "r", encoding="utf-8") as f:
            configs = json.load(f)
        task_id = configs.get("task_id", "")
        intent = configs.get("intent", "")
        print(f"task_id: {task_id}, intent: {intent}")

        html_file = os.path.join(html_folder, f"render_{task_id}.html")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        image_bs64s = self.extract_and_validate_images(html_content)
        answer = self.extract_answer(html_content)
        print(f"answer: {answer}")

        SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the last 5 screens showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'.
You should provide the 'SUCCESS' or 'NOT SUCCESS' between <result> and </result>."""

        USER_PROMPT = """TASK: {task}
        Result Response: {answer}"""
        user_prompt = USER_PROMPT.format(task=intent, answer=answer)
        print(f"user_prompt: {user_prompt}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        image_contents = []
        for image_bs64 in image_bs64s[-5:]:
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_bs64['base64_data']}"},
            })
        messages.append({"role": "user", "content": image_contents})

        response, _, _ = self.vllm_client.chat(
            messages=messages,
            temperature=0.8,
            max_tokens=1024,
        )
        print("response", response)
        answer_text = response.content.strip().lower()
        try:
            extracted_answer = re.search(r"<result>(.*?)</result>", answer_text).group(1)
            if not extracted_answer:
                extracted_answer = answer_text
        except Exception:
            extracted_answer = answer_text

        if "not success" in extracted_answer:
            return 0.0, answer_text, answer
        return 1.0, answer_text, answer

    def extract_and_validate_images(self, html_content: str) -> list[dict]:
        match_pattern = r"<img\s+[^>]*src=['\"]data:image/[^;]+;base64,([^'\"]+)['\"][^>]*>"
        matches = re.findall(match_pattern, html_content)
        valid_images = []
        for base64_data in matches:
            try:
                image_data = base64.b64decode(base64_data)
                img = Image.open(BytesIO(image_data))
                valid_images.append({
                    "base64_data": base64_data,
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode,
                })
            except Exception as e:
                print(f"Invalid or corrupted image data: {e}")
                continue
        return valid_images

    def extract_answer(self, html_content: str) -> str:
        match_pattern = r"finished\(answer=(.*?)\)"
        answer = re.search(match_pattern, html_content)
        return answer.group(1) if answer else ""


if __name__ == "__main__":
    from config.argument_parser import config
    from agent.llm_config import load_tool_llm

    args = config()
    evaluate_model = load_tool_llm(args, "qwen2.5-vl-32b")
    evaluator = LLMEvaluator(vllm_client=evaluate_model)
    config_file = os.path.join(
        os.environ.get("INFERENCE_PROJECT_ROOT", "."),
        "webvoyager_evaluation/data/test/test_ESPN--4.json",
    )
    html_folder = os.path.join(
        os.environ.get("INFERENCE_PROJECT_ROOT", "."),
        "results/webvoyager/test/qwen2.5-vl",
    )
    score, answer_text, ori_answer = evaluator(config_file, html_folder)
    print(score, answer_text, ori_answer)
