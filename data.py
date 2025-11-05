# data.py

import os
import re
import json
import hashlib
import argparse
from openai import OpenAI
from tqdm import tqdm

from config import OPENAI_API_KEY, MODEL_NAME, TEXTS_FILE, SCENE_GRAPHS_FILE, USE_PROXY, PROXY_URL


# 代理，不需要代理请勿运行
if USE_PROXY:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

# 定期缓存
FLUSH_EVERY = 50

SYSTEM_PROMPT = """You are a smart intelligent assistant who can help me generate short story (2-4 sentences) and its corresponding scene graph.

Story requirements:
- Diverse characters, environments and actions.
- Should naturally evolve into 2-5 time frames

Scene Graph requirements:
- Strict JSON format (list of frames)
- Each frame: {"time": "T#", "nodes": [...], "edges": [...]}
- Nodes: {"id": "", "attributes": ["", ...]}
- Edges: ["subject", "predicate", "object"]
- Only include explicit elements from the text (no inferred elements)
- Weather/setting should be attributes on existing nodes if and only if explicitly mentioned (e.g., park: sunny).
- Keep frames minimal: 2-5 frames.
- Do not repeat scenes between each Story.

Your output format (exactly):
Story: <one paragraph with several sentences>
Scene Graph:
<JSON array>

For example :
Story: A young boy in a red shirt waves at a brown dog in a sunny park. The dog runs toward the boy wagging its tail, and the boy kneels down to pet it.
Scene Graph:
[
  {
    "time": "T1",
    "nodes": [
      { "id": "boy",  "attributes": ["young", "red_shirt"] },
      { "id": "dog",  "attributes": ["brown"             ] },
      { "id": "park", "attributes": ["sunny"             ] }
    ],
    "edges": [ ["boy", "in", "park"], ["dog", "in", "park"], ["boy", "waves_at", "dog"] ]
  },
  {
    "time": "T2",
    "nodes": [
      { "id": "boy",  "attributes": ["young", "red_shirt"                ] },
      { "id": "dog",  "attributes": ["brown", "running",   "wagging_tail"] },
      { "id": "park", "attributes": ["sunny"                             ] }
    ],
    "edges": [ ["boy", "in", "park"], ["dog", "in", "park"], ["dog", "runs_toward", "boy"] ]
  },
  {
    "time": "T3",
    "nodes": [
      { "id": "boy",  "attributes": ["young", "red_shirt", "kneeling"] },
      { "id": "dog",  "attributes": ["brown"                         ] },
      { "id": "park", "attributes": ["sunny"                         ] }
    ],
    "edges": [ ["boy", "in", "park"], ["dog", "in", "park"], ["boy", "pets", "dog"] ]
  }
]
"""


class SceneGraphGenerator:
    def __init__(self, num_data):
        self.num_data = num_data
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.texts_data = self._load_data(TEXTS_FILE)
        self.graphs_data = self._load_data(SCENE_GRAPHS_FILE)
        self.existing_text_hashes = {self._hash_text(
            item["text"]) for item in self.texts_data if isinstance(item, dict) and "text" in item}
        self.flush_every = FLUSH_EVERY
        self.system_prompt = SYSTEM_PROMPT

    def _load_data(self, file_path):
        """Load data from a JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _hash_text(self, text):
        """Generate a hash for the given text."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _snake(self, s):
        """Convert string to snake_case."""
        return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")

    def _text_vocab(self, text):
        """Return a set of words (vocabulary) in the text."""
        WORD_RE = re.compile(r"[a-z]+")
        return set(WORD_RE.findall(text.lower()))

    def _lint_and_fix(self, text, graph):
        """Fix graph and check errors."""
        errors = []
        if not isinstance(graph, list) or not graph:
            return graph, ["graph_not_list"]

        vocab = self._text_vocab(text)
        fixed = []

        for i, fr in enumerate(graph, 1):
            nodes = fr.get("nodes", [])
            edges = fr.get("edges", [])

            nid_set = set()
            new_nodes = []
            for nd in nodes:
                nid = self._snake(nd.get("id", ""))
                attrs = [self._snake(a) for a in nd.get("attributes", [])]
                if not re.match(r"^[a-z0-9_]+$", nid):
                    errors.append(f"node_not_snake:{nid}")
                parts = [p for p in nid.split("_") if p]
                if parts and not all(p in vocab for p in parts):
                    errors.append(f"node_not_in_text:{nid}")
                nid_set.add(nid)
                new_nodes.append({"id": nid, "attributes": attrs})

            new_edges = []
            seen = set()
            for e in edges:
                if not isinstance(e, list) or len(e) != 3:
                    errors.append("edge_bad_shape")
                    continue
                s, p, o = self._snake(e[0]), self._snake(
                    e[1]), self._snake(e[2])
                if s in nid_set and o in nid_set:
                    key = (s, p, o)
                    if key not in seen:
                        new_edges.append([s, p, o])
                        seen.add(key)
                else:
                    errors.append(f"edge_dangling:{[s, p, o]}")

            fixed.append({
                "time": f"T{i}",
                "nodes": new_nodes,
                "edges": new_edges
            })

        return fixed, errors

    def _save_data(self):
        """Save updated texts and graphs data to the respective files."""
        with open(TEXTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.texts_data, f, indent=4, ensure_ascii=False)

        with open(SCENE_GRAPHS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.graphs_data, f, indent=4, ensure_ascii=False)

    def generate_data(self):
        """Generate and enrich scene graph data."""
        for i in tqdm(range(self.num_data)):  # 根据命令行参数控制生成数据的数量
            try:
                # 调用 OpenAI API 获取数据
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system",
                            "content": "You are a smart intelligent assistant who can help me generate short story (2-4 sentences) and its corresponding scene graph."},
                        {"role": "user", "content": self.system_prompt}
                    ]
                )
                output = response.choices[0].message.content.strip()

                # 处理生成的文本和图形
                if "Story:" in output and "Scene Graph:" in output:
                    parts = output.split("Scene Graph:")
                    text = parts[0].replace("Story:", "").strip()
                    text_hash = self._hash_text(text)

                    # 检查文本是否重复
                    if text_hash in self.existing_text_hashes:
                        continue

                    graph_raw = parts[1].strip()
                    try:
                        graph = json.loads(graph_raw)
                    except json.JSONDecodeError:
                        continue

                    fixed_graph, errs = self._lint_and_fix(text, graph)
                    if "graph_not_list" in errs or "edge_bad_shape" in errs:
                        continue

                    # 更新数据
                    self.texts_data.append({"text": text})
                    self.graphs_data.append(fixed_graph)
                    self.existing_text_hashes.add(text_hash)

                    # 定期缓存
                    if len(self.existing_text_hashes) % self.flush_every == 0:
                        self._save_data()

            except Exception:
                continue

        self._save_data()  # 保存最终数据


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scene graphs and short stories.")
    parser.add_argument(
        "--num_data", type=int, default=2, help="Number of data to generate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = SceneGraphGenerator(num_data=args.num_data)
    generator.generate_data()
