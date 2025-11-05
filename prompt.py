# prompt.py

import os
import re
import json
from openai import OpenAI
from config import OPENAI_API_KEY, USE_PROXY, PROXY_URL


# 代理，不需要代理请勿运行
if USE_PROXY:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

client = OpenAI(api_key=OPENAI_API_KEY)

PREFIX = "colored illustration of"
STYLE_SUFFIX = "clean outlines, vibrant colors, natural shading, soft cartoon style, high quality"

SYSTEM_PROMPT = (
    "You turn a scene graph (frames with nodes/edges/attributes) into ONE English drawing prompt for each frame.\n"
    "Rules:\n"
    "1) Output exactly ONE line without quotes or code fences for each frame.\n"
    f"2) The line MUST start with: `{PREFIX}`\n"
    f"3) The line MUST end with: `{STYLE_SUFFIX}`\n"
    "4) Keep facts consistent with the JSON; do not contradict.\n"
)

USER_PROMPT = (
    "Given this scene graph (JSON below), produce ONE single-line prompt in English for each frame.\n"
    "Start EXACTLY with: colored illustration of\n"
    "End EXACTLY with: clean outlines, vibrant colors, natural shading, soft cartoon style, high quality\n\n"
    "Scene Graph JSON:\n{sg_json}"
)


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _strip_code_fences(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"^```(?:json|txt)?", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"```$", "", s.strip(), flags=re.IGNORECASE)
    return s.strip()


def _ensure_prefix_suffix(s: str) -> str:
    s = _norm_spaces(_strip_code_fences(s)).rstrip(",.")
    # 前缀
    if not s.lower().startswith(PREFIX):
        s = f"{PREFIX} {s}"
    # 后缀
    if not s.lower().endswith(STYLE_SUFFIX):
        s = f"{s}, {STYLE_SUFFIX}"
    return _norm_spaces(s)


def _coerce_single_line(s: str) -> str:
    """Only take the non-empty text on the first line."""
    s = _strip_code_fences(s)
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _call_one_frame(frame, model="gpt-5-mini", max_retries=3):
    sg_json = json.dumps([frame], ensure_ascii=False, indent=2)
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT.format(
                        sg_json=sg_json)},
                ]
            )
            content = (resp.choices[0].message.content or "").strip()
            line = _coerce_single_line(content)
            line = _ensure_prefix_suffix(line)
            if not line.lower().startswith(PREFIX) or not line.lower().endswith(STYLE_SUFFIX):
                raise ValueError("prefix/suffix constraint not satisfied")
            return line
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(
        "promptify (per-frame) failed without exception")


def generate_prompt(frames, model="gpt-5-mini", max_retries=3, return_list=False):
    if isinstance(frames, dict) and "time" in frames:
        frames = [frames]

    prompts = []
    for fr in frames or []:
        line = _call_one_frame(fr, model=model, max_retries=max_retries)
        prompts.append(line)

    return prompts if return_list else "\n".join(prompts)
