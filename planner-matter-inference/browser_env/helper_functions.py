import html
import json
import re
from pathlib import Path
from typing import Any

from browser_env import Action, StateInfo

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


def get_render_action(action: Action) -> str:
    """Format predicted actions for HTML rendering (raw text + parsed description)."""
    from .action_parser_ground import get_action_description as ground_get_action_description

    if not isinstance(action, list):
        action = [action]
    action_str = ""
    for per_action in action:
        raw_text = per_action.get("text") or ""
        raw_answer = per_action.get("answer") or ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)
        full_text = html.escape(raw_text + raw_answer)
        parsed = html.escape(ground_get_action_description(per_action))
        action_str += f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{full_text}</pre></div>"
        action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{parsed}</pre></div>"
    return action_str


def get_action_description(action: Action) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    from .action_parser_ground import get_action_description as ground_get_action_description
    if isinstance(action, list):
        action_str = '; '.join(ground_get_action_description(per_action) for per_action in action)
    else:
        action_str = ground_get_action_description(action)

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, config_file: str, result_dir: str
    ) -> None:
        with open(config_file, "r", encoding="utf-8") as f:
            _config = json.load(f)

        _config_str = "".join(f"{k}: {v}\n" for k, v in _config.items())
        _config_str = f"<pre>{html.escape(_config_str)}</pre>\n"
        task_id = _config.get("task_id", _config.get("id", "unknown"))

        Path(result_dir).mkdir(parents=True, exist_ok=True)
        self.render_file = open(
            Path(result_dir) / f"render_{task_id}.html", "a+", encoding="utf-8"
        )
        self.render_file.truncate(0)
        self.render_file.write(HTML_TEMPLATE.format(body=_config_str))
        self.render_file.flush()
        # Track if augmented_intent has been shown (only show once at the beginning)
        self._augmented_intent_shown = False

    def _get_body_from_html(self) -> str:
        """Read current HTML and return the body content; fallback to empty string if parsing fails."""
        self.render_file.seek(0)
        html_content = self.render_file.read()
        match = re.search(r"<body>(.*?)</body>", html_content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Append one step (observation + meta + action) to the trajectory HTML."""
        observation = state_info.get("observation") or {}
        text_obs = observation.get("text", "")
        if not isinstance(text_obs, str):
            text_obs = str(text_obs)
        text_obs = html.escape(text_obs)
        info = state_info.get("info") or {}
        page = info.get("page")
        url = getattr(page, "url", "") if page else ""
        url_esc = html.escape(url)

        new_content = "<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href=\"{url_esc}\">URL: {url_esc}</a></h3>\n"
        new_content += f"<div class='state_obv'><pre>{text_obs}</pre></div>\n"

        if render_screenshot:
            img_obs = observation.get("image_for_render") or observation.get("image", "")
            if img_obs:
                new_content += f"<img src='data:image/png;base64,{img_obs}' style='width:50vw; height:auto;'/>\n"
        # Augmented intent (from planner) — show only once at the beginning
        if not self._augmented_intent_shown and meta_data and meta_data.get("augmented_intent"):
            augmented_intent = "**Augmented Intent (Planner):** " + str(meta_data["augmented_intent"])
            new_content += f"<div class='augmented_intent' style='background-color:#E6F3FF; padding:10px; margin:10px 0; border-radius:5px;'><pre>{html.escape(augmented_intent)}</pre></div>\n"
            self._augmented_intent_shown = True  # Mark as shown so it won't appear in subsequent steps
        # verifier feedback
        if meta_data and 'verifier_feedback' in meta_data and meta_data['verifier_feedback']:
            verifier = meta_data['verifier_feedback']
            verifier_content = "**Verifier Feedback:**\n"
            if isinstance(verifier, dict):
                effectiveness_reason = verifier.get('effectiveness_reason', '')
                reminder = verifier.get('reminder', '')
                if effectiveness_reason:
                    verifier_content += f"*Action Analysis*: {effectiveness_reason}\n"
                if reminder:
                    verifier_content += f"*Reminder*: {reminder}\n"
            else:
                verifier_content += str(verifier)
            new_content += f"<div class='verifier_feedback' style='background-color:#FFF4E6; padding:10px; margin:10px 0; border-radius:5px;'><pre>{verifier_content}</pre></div>\n"
        # trajectory verifier feedback
        if meta_data and 'trajectory_verifier_feedback' in meta_data:
            trajectory_verifier_feedback = meta_data['trajectory_verifier_feedback']
            new_content += f"<div class='trajectory_verifier_feedback' style='background-color:#E6F3FF; padding:10px; margin:10px 0; border-radius:5px;'><pre>{trajectory_verifier_feedback}</pre></div>\n"
        # plan
        if meta_data and 'task_plan' in meta_data:
            task_plan = meta_data['task_plan']['plan']
            new_content += f"<div class='task_plan'><pre>{task_plan}</pre></div>\n"
        # step history reflection
        if meta_data and 'step_history_reflection' in meta_data:
            step_history_reflection = meta_data['step_history_reflection']
            new_content += f"<div class='step_history_reflection'><pre>{step_history_reflection}</pre></div>\n"
        # experience action suggestions
        if meta_data and 'experience_action_suggestions' in meta_data:
            experience_action_suggestions = meta_data['experience_action_suggestions']
            new_content += f"<div class='experience_action_suggestions'><pre>{experience_action_suggestions}</pre></div>\n"
        # candidate actions
        if meta_data and 'candidate_actions_scores' in meta_data:
            candidate_actions_scores = meta_data['candidate_actions_scores']
            new_content += f"<div class='candidate_actions_scores'><pre>{candidate_actions_scores}</pre></div>\n"
        # response
        if meta_data and 'response_history' in meta_data:
            response = meta_data['response_history'][-1]
            new_content += f"<div class='response_history'><pre>{response}</pre></div>\n"
        # action
        action_str = get_render_action(action)
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        html_body = self._get_body_from_html() + new_content
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(HTML_TEMPLATE.format(body=html_body))
        self.render_file.flush()

    def close(self, score: float = None, answer_text: str = None, ori_answer: str = None) -> None:
        """Close the render file, optionally adding evaluation results"""
        if score is not None or answer_text is not None or ori_answer is not None:
            # Add evaluation results section before closing
            eval_content = "<h2>Evaluation Results</h2>\n"
            
            if score is not None:
                result_text = "PASS" if score == 1.0 else "FAIL"
                score_color = "#4CAF50" if score == 1.0 else "#F44336"
                eval_content += f"<div class='evaluation_score' style='background-color:{score_color}; color:white; padding:15px; margin:10px 0; border-radius:5px; font-size:18px; font-weight:bold;'>"
                eval_content += f"<strong>Score: {score} ({result_text})</strong></div>\n"
            
            if answer_text:
                eval_content += f"<div class='evaluation_answer' style='background-color:#E3F2FD; padding:15px; margin:10px 0; border-radius:5px;'>"
                eval_content += f"<strong>Evaluator Response:</strong><pre style='white-space:pre-wrap;'>{answer_text}</pre></div>\n"
            
            if ori_answer:
                eval_content += f"<div class='evaluation_ori_answer' style='background-color:#F5F5F5; padding:15px; margin:10px 0; border-radius:5px;'>"
                eval_content += f"<strong>Original Answer:</strong><pre style='white-space:pre-wrap;'>{ori_answer}</pre></div>\n"
            
            html_body = self._get_body_from_html() + eval_content
            self.render_file.seek(0)
            self.render_file.truncate()
            self.render_file.write(HTML_TEMPLATE.format(body=html_body))
            self.render_file.flush()
        
        self.render_file.close()
