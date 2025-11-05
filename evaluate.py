# evaluate.py

import json
import pandas as pd
import argparse
from pipeline import build_scene_graph
from metric import spice_on_files
from enrich import enrich
from prompt import generate_prompt
from config import TEXTS_FILE, SCENE_GRAPHS_FILE, OUTPUT_PATH, EVALUATION_PATH

import warnings
warnings.filterwarnings('ignore')


class SceneGraphEvaluator:
    def __init__(self, texts_path, graphs_path, output_path, evaluation_path, evaluate=False, enrich=False, prompt=False, enrich_path=None, prompt_path=None):
        self.texts_path = texts_path
        self.graphs_path = graphs_path
        self.output_path = output_path
        self.evaluation_path = evaluation_path
        self.evaluate = evaluate
        self.enrich = enrich
        self.prompt = prompt
        self.enrich_path = enrich_path
        self.prompt_path = prompt_path

        self.texts = self.read_texts()
        self.scene_graphs = []

    def read_texts(self):
        data = json.load(open(self.texts_path, "r", encoding="utf-8"))
        texts = [str(item["text"]) if isinstance(item, dict)
                 and "text" in item else item for item in data]
        return texts

    def predict_scene_graphs(self):
        for _, t in enumerate(self.texts, 1):
            res = build_scene_graph(t, debug=False)
            frames = res[0] if isinstance(res, tuple) else res
            clean = []
            for j, fr in enumerate(frames or [], 1):
                time = fr.get("time") or f"T{j}"
                nodes = fr.get("nodes") or []
                edges = fr.get("edges") or []

                # 去编号
                id_map = {}
                merged = {}
                for nd in nodes:
                    old = nd.get("id")
                    if not old:
                        continue

                    base = str(old).split('#', 1)[0]
                    id_map[old] = base
                    attrs = nd.get("attributes") or []
                    if base not in merged:
                        item = {"id": base, "attributes": list(
                            dict.fromkeys(attrs))}
                        if "count" in nd:
                            item["count"] = nd.get("count")
                        merged[base] = item
                    else:
                        merged[base]["attributes"] = list(dict.fromkeys(
                            (merged[base].get("attributes") or []) + attrs))

                nodes2 = list(merged.values())

                seen = set()
                edges2 = []
                for e in edges:
                    if isinstance(e, (list, tuple)) and len(e) == 3:
                        s, p, o = e
                        s2 = id_map.get(s, str(s).split('#', 1)[0])
                        o2 = id_map.get(o, str(o).split('#', 1)[0])
                        if s2 in merged and o2 in merged:
                            tup = (s2, p, o2)
                            if tup not in seen:
                                seen.add(tup)
                                edges2.append([s2, p, o2])

                clean.append({
                    "time": time,
                    "scene": (fr.get("scene") or "").strip(),
                    "nodes": nodes2,
                    "edges": edges2
                })
            self.scene_graphs.append(clean)

    def dump_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def evaluate_results(self, gold_path, pred_path, report_json,
                         mode='synonym', by_frame=False, weights=(1, 1, 1)):
        report = spice_on_files(gold_path, pred_path,
                                mode=mode,
                                by_frame=by_frame,
                                weights=weights,
                                allow_multi_ref=True,
                                save_report_path=report_json)
        df = self.build_dataframe(report)

        return df, report

    def build_dataframe(self, report):
        micro = self.aggregate_micro(report.get("samples", []))
        rows = []
        for name in ("objects", "attributes", "relations"):
            m = micro[name]
            rows.append({
                "Metric": name,
                "Precision": m["precision"],
                "Recall":    m["recall"],
                "F1":        m["f1"],
                "TP":        m["tp"],
                "Pred":      m["pred"],
                "Gold":      m["gold"],
            })
        rows.append({"Metric": "SPICE_macro",
                     "Precision": None,
                     "Recall": None,
                     "F1": report.get("dataset_SPICE_macro", 0.0),
                     "TP": None, "Pred": None, "Gold": None})
        rows.append({"Metric": "SPICE_micro",
                     "Precision": None, "Recall": None,
                     "F1": report.get("dataset_SPICE_micro", 0.0),
                     "TP": None, "Pred": None, "Gold": None})

        df = pd.DataFrame(
            rows, columns=["Metric", "Precision", "Recall",
                           "F1", "TP", "Pred", "Gold"])

        mask = df["Metric"].str.startswith("SPICE")
        for col in ["Precision", "Recall", "TP", "Pred", "Gold"]:
            df.loc[mask, col] = "-"

        for col in ("Precision", "Recall", "F1"):
            df[col] = df[col].apply(lambda x: round(
                float(x), 4) if isinstance(x, (int, float)) else x)
        return df

    def aggregate_micro(self, samples):
        acc = {
            "objects":    {"tp": 0, "pred": 0, "gold": 0},
            "attributes": {"tp": 0, "pred": 0, "gold": 0},
            "relations":  {"tp": 0, "pred": 0, "gold": 0},
        }
        for s in samples:
            for k in ("objects", "attributes", "relations"):
                block = s.get(k) if isinstance(s.get(k), dict) else None
                if block and all(x in block for x in ("tp", "pred", "gold")):
                    acc[k]["tp"] += int(block.get("tp", 0))
                    acc[k]["pred"] += int(block.get("pred", 0))
                    acc[k]["gold"] += int(block.get("gold", 0))

        def prf(b):
            P = b["tp"] / (b["pred"] or 1.0)
            R = b["tp"] / (b["gold"] or 1.0)
            F = 2 * P * R / (P + R) if (P + R) else 0.0
            return {
                "precision": round(P, 4),
                "recall":    round(R, 4),
                "f1":        round(F, 4),
                "tp":        b["tp"],
                "pred":      b["pred"],
                "gold":      b["gold"],
            }

        return {k: prf(v) for k, v in acc.items()}

    def enrich_scene_graphs(self):
        enriched_graphs = []
        for graph in self.scene_graphs:
            enriched = enrich(graph)
            enriched_graphs.append(enriched)
        self.dump_json(enriched_graphs, self.enrich_path)

    def generate_prompts(self):
        prompts = []
        for graph in self.scene_graphs:
            prompts.append(generate_prompt(graph))
        self.dump_json(prompts, self.prompt_path)

    def run(self):
        self.predict_scene_graphs()
        self.dump_json(self.scene_graphs, self.output_path)

        if self.enrich:
            self.enrich_scene_graphs()
        if self.prompt:
            self.generate_prompts()
        if self.evaluate:
            df, report = self.evaluate_results(self.graphs_path,
                                               self.output_path,
                                               self.evaluation_path)
            print(df)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scene Graph Generation and Evaluation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation")
    parser.add_argument("--enrich", action="store_true",
                        help="Run scene graph enrichment")
    parser.add_argument("--prompt", action="store_true",
                        help="Generate prompts for scene graphs")
    parser.add_argument("--enrich-path", type=str,
                        help="Path to save enriched scene graphs")
    parser.add_argument("--prompt-path", type=str,
                        help="Path to save generated prompts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = SceneGraphEvaluator(TEXTS_FILE, SCENE_GRAPHS_FILE, OUTPUT_PATH,
                                    EVALUATION_PATH, args.evaluate, args.enrich,
                                    args.prompt, args.enrich_path, args.prompt_path)
    evaluator.run()
