# metric.py

import re
import json
from nltk.corpus import wordnet as wn


_TIME = re.compile(r"^T(\d+)$")
_HAS_WN = True
LEM_CACHE, SYN_CACHE = {}, {}


def snake(s):
    return re.sub(r"[^a-z0-9_]+", "_", (s or "").lower()).strip("_")


def lemmatize_tok(tok):
    t = (tok or "").lower()
    if t in LEM_CACHE:
        return LEM_CACHE[t]
    if _HAS_WN:
        parts = t.split("_")
        head = parts[0]
        cands = [head]
        for pos in ["n", "v", "a", "r"]:
            for syn in wn.synsets(head, pos=pos):
                for l in syn.lemmas():
                    cands.append(l.name().lower().replace(" ", "_"))
        head = sorted(set(cands), key=len)[0]
        out = "_".join([head] + parts[1:])
    else:
        out = re.sub(r"(ing|ed|es|s)$", "", t)
    LEM_CACHE[t] = out
    return out


def synset(tok):
    t = (tok or "").lower()
    if t in SYN_CACHE:
        return SYN_CACHE[t]
    if not _HAS_WN:
        SYN_CACHE[t] = {t}
        return SYN_CACHE[t]
    out = {t}
    parts = t.split("_")
    head = parts[0]
    for pos in ["n", "v", "a", "r"]:
        for syn in wn.synsets(head, pos=pos):
            for l in syn.lemmas():
                out.add(l.name().lower().replace(" ", "_"))
    SYN_CACHE[t] = {"_".join([w] + parts[1:]) for w in out}
    return SYN_CACHE[t]


def match_token(a, b, mode="synonym"):
    if mode == "exact":
        return snake(a) == snake(b)
    if mode == "lemma":
        return lemmatize_tok(a) == lemmatize_tok(b)
    aa, bb = lemmatize_tok(a), lemmatize_tok(b)
    if aa == bb:
        return True
    return len(synset(aa) & synset(bb)) > 0


def obj_set(frame):
    out = set()
    for nd in frame.get("nodes", []):
        out.add(snake(nd.get("id", "")))
    return out


def attr_set(frame):
    out = set()
    for nd in frame.get("nodes", []):
        nid = snake(nd.get("id", ""))
        for a in nd.get("attributes", []):
            out.add((nid, snake(a)))
    return out


def rel_set(frame):
    out = set()
    nodes = obj_set(frame)
    for e in frame.get("edges", []):
        if not isinstance(e, (list, tuple)) or len(e) < 3:
            continue
        s, p, o = e[0], e[1], e[2]
        s, p, o = snake(s), snake(p), snake(o)
        if s in nodes and o in nodes:
            out.add((s, p, o))
    return out


def merge_frames(frames):
    O, A, R = set(), set(), set()
    for fr in frames:
        O |= obj_set(fr)
        A |= attr_set(fr)
        R |= rel_set(fr)
    return O, A, R


def f1_match(gold_set, pred_set, mode, kind):

    gold_list = list(gold_set)
    used = set()
    tp = 0

    def ok(x, y):
        if kind == "obj":
            return match_token(x, y, mode)
        if kind == "attr":
            return (x[0] == y[0]) and match_token(x[1], y[1], mode)
        if kind == "rel":
            return (x[0] == y[0]) and (x[2] == y[2]) and match_token(x[1], y[1], mode)
        return x == y

    for y in pred_set:
        for i, x in enumerate(gold_list):
            if i in used:
                continue
            if ok(x, y):
                used.add(i)
                tp += 1
                break

    P = tp / (len(pred_set) or 1)
    R = tp / (len(gold_set) or 1)
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return {
        "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
        "tp": tp, "pred": len(pred_set), "gold": len(gold_set)
    }


def frame_map(frames):
    m = {}
    for i, fr in enumerate(frames, 1):
        t = fr.get("time")
        if not isinstance(t, str) or not _TIME.match(t):
            t = f"T{i}"
        m[t] = fr
    return m


def spice_for_pair(gold_frames, pred_frames, *, mode="synonym", by_frame=False, weights=(1, 1, 1)):
    wO, wA, wR = weights
    if not by_frame:
        gO, gA, gR = merge_frames(gold_frames)
        pO, pA, pR = merge_frames(pred_frames)
        so = f1_match(gO, pO, mode, "obj")
        sa = f1_match(gA, pA, mode, "attr")
        sr = f1_match(gR, pR, mode, "rel")
        denom = (wO + wA + wR) or 1
        spice = round((wO*so["f1"] + wA*sa["f1"] + wR*sr["f1"]) / denom, 4)
        return {"SPICE": spice, "objects": so, "attributes": sa, "relations": sr}
    else:
        gm, pm = frame_map(gold_frames), frame_map(pred_frames)
        keys = sorted(set(gm.keys()) | set(pm.keys()), key=lambda k: int(
            _TIME.match(k).group(1)) if _TIME.match(k) else 10**9)
        so = sa = sr = 0.0
        per = {}
        for t in keys:
            g = gm.get(t, {"nodes": [], "edges": []})
            p = pm.get(t, {"nodes": [], "edges": []})
            _so = f1_match(obj_set(g), obj_set(p), mode, "obj")
            _sa = f1_match(attr_set(g), attr_set(p), mode, "attr")
            _sr = f1_match(rel_set(g), rel_set(p), mode, "rel")
            denom = (wO + wA + wR) or 1
            per[t] = {
                "objects": _so, "attributes": _sa, "relations": _sr,
                "SPICE": round((wO*_so["f1"] + wA*_sa["f1"] + wR*_sr["f1"]) / denom, 4)
            }
            so += _so["f1"]
            sa += _sa["f1"]
            sr += _sr["f1"]
        denom = (wO + wA + wR) or 1
        spice = round(((wO*(so/len(keys) if keys else 0.0)) +
                       (wA*(sa/len(keys) if keys else 0.0)) +
                       (wR*(sr/len(keys) if keys else 0.0))) / denom, 4)
        return {"SPICE": spice,
                "objects": {"f1": round(so/len(keys), 4) if keys else 0.0},
                "attributes": {"f1": round(sa/len(keys), 4) if keys else 0.0},
                "relations": {"f1": round(sr/len(keys), 4) if keys else 0.0},
                "per_frame": per}


def spice_for_pair_multi(gold_refs, pred_frames, *, mode="synonym", by_frame=False, weights=(1, 1, 1)):
    best = None
    for ref in gold_refs:
        cur = spice_for_pair(ref, pred_frames, mode=mode,
                             by_frame=by_frame, weights=weights)
        if (best is None) or (cur["SPICE"] > best["SPICE"]):
            best = cur
    return best or {"SPICE": 0.0}


def spice_on_files(gold_path, pred_path, *, mode="synonym", by_frame=False, weights=(1, 1, 1), allow_multi_ref=True, save_report_path=None):
    gold = json.load(open(gold_path, "r", encoding="utf-8"))
    pred = json.load(open(pred_path, "r", encoding="utf-8"))

    def frames_of(x):
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            for k in ("frames", "scene_graph", "gold_frames", "pred_frames"):
                if k in x:
                    return x[k]
        return x

    out = {"dataset_SPICE_macro": 0.0, "dataset_SPICE_micro": 0.0,
           "samples": [], "mode": mode, "by_frame": by_frame}
    if not gold:
        return out

    micro = {"obj": {"tp": 0, "pred": 0, "gold": 0},
             "attr": {"tp": 0, "pred": 0, "gold": 0},
             "rel": {"tp": 0, "pred": 0, "gold": 0}}
    S = 0.0

    for g, p in zip(gold, pred):
        G = frames_of(g)
        P = frames_of(p)
        if allow_multi_ref and isinstance(G, list) and G and isinstance(G[0], list):
            res = spice_for_pair_multi(
                G, P, mode=mode, by_frame=by_frame, weights=weights)
        else:
            res = spice_for_pair(
                G, P, mode=mode, by_frame=by_frame, weights=weights)
        out["samples"].append(res)
        S += res["SPICE"]

        if "objects" in res and isinstance(res["objects"], dict) and "tp" in res["objects"]:
            micro["obj"]["tp"] += res["objects"]["tp"]
            micro["obj"]["pred"] += res["objects"]["pred"]
            micro["obj"]["gold"] += res["objects"]["gold"]
        if "attributes" in res and isinstance(res["attributes"], dict) and "tp" in res["attributes"]:
            micro["attr"]["tp"] += res["attributes"]["tp"]
            micro["attr"]["pred"] += res["attributes"]["pred"]
            micro["attr"]["gold"] += res["attributes"]["gold"]
        if "relations" in res and isinstance(res["relations"], dict) and "tp" in res["relations"]:
            micro["rel"]["tp"] += res["relations"]["tp"]
            micro["rel"]["pred"] += res["relations"]["pred"]
            micro["rel"]["gold"] += res["relations"]["gold"]

    out["dataset_SPICE_macro"] = round(S / len(out["samples"]), 4)

    def micro_f1(block):
        P = block["tp"]/(block["pred"] or 1)
        R = block["tp"]/(block["gold"] or 1)
        return 2*P*R/(P+R) if (P+R) else 0.0
    mo, ma, mr = micro_f1(micro["obj"]), micro_f1(
        micro["attr"]), micro_f1(micro["rel"])
    denom = sum(weights) or 1
    out["dataset_SPICE_micro"] = round(
        (weights[0]*mo + weights[1]*ma + weights[2]*mr)/denom, 4)
    out["micro_breakdown"] = {"objects": round(
        mo, 4), "attributes": round(ma, 4), "relations": round(mr, 4)}

    if save_report_path:
        json.dump(out, open(save_report_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    return out


def spice_delta_on_files(raw_pred_file: str, post_pred_file: str, gold_file: str):
    if "spice_on_files" not in globals():
        raise RuntimeError("metric.py need spice_on_files(...)")
    raw = spice_on_files(raw_pred_file, gold_file)
    post = spice_on_files(post_pred_file, gold_file)
    return {"raw": raw, "post": post}
