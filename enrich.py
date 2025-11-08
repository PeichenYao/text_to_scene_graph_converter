# enrich.py

import os
import re
import json
from openai import OpenAI
from config import OPENAI_API_KEY, USE_PROXY, PROXY_URL


# Proxy settings; enable only if a proxy is required
if USE_PROXY:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a smart intelligent assistant who enriches a base scene graph.

OBJECTIVE
- Input:
  - TEXT_ORIGINAL = the raw story text (full original text).
  - BASE_SCENE_GRAPH = a JSON array of frames. Each frame has {"time", "scene", "nodes", "edges"}.
- Output: a FULL ENRICHED scene graph **as a JSON ARRAY of frames** with the **same times**.
- Keep every original node/edge **untouched**; only **append** new nodes/edges and **append** attributes to existing nodes.

STRICT FORMAT
- Return **ONLY JSON** (no prose, no markdown). The top-level must be a JSON **array** of frames.
- Each frame object must have **exactly**: "time", "scene", "nodes", "edges".
- Every node object may contain **only**:
  - "id"  (snake_case string)
  - "attributes" (array of snake_case strings)
  - If a node in the input already has "count", keep it unchanged; do **not** invent new "count".
- Do **not** introduce any other fields. All semantics must be encoded as **attributes**.

HOW TO ENRICH (put everything into attributes)
- Prefer explicit facts from TEXT_ORIGINAL and BASE_SCENE_GRAPH; never contradict TEXT_ORIGINAL.
- Colors: add tokens like **<color>** (e.g., blue, brown, yellow).
- Materials/texture: **<mat>** (e.g., wood, metal, grass).
- Size hints: **<size>** (e.g. small, big).
- Scene background: if reasonable, add common environment entities (e.g., sky, sun, clouds, grass, trees, road, sidewalk, water, sand, buildings, benches, path, cabinets, stove, sink, etc.), each as a node with an "id" and an "attributes" list.
- Spatial relations: add conventional edges only (e.g., in/on/above/under/next_to/near/front_of/behind/toward/touches) and **never contradict** existing edges.

PRIORITY & INFERENCE
- Prefer explicit textual facts from TEXT_ORIGINAL/base graph; infer **minimal** plausible defaults only when helpful.
- Use short, compositional, snake_case attributes; avoid long sentences.

STABILITY
- For the same node id across frames, keep inferred attributes (especially colors/materials/size) **consistent** unless the text clearly implies a change.

RETURN
- Return the **complete enriched scene graph** (all frames) as a single JSON array that preserves the original content and only adds nodes/edges/attributes under the rules above."""

USER_PROMPT = """You are given TEXT_ORIGINAL and a BASE_SCENE_GRAPH. Enrich the graph according to the system rules.

TEXT_ORIGINAL:
{0}

BASE_SCENE_GRAPH:
{1}

OUTPUT REQUIREMENTS
- Return ONLY the FULL enriched scene graph as a JSON ARRAY of frames (no prose, no markdown).
- Preserve every original frame/time, scene, node, edge. Do NOT remove or rename anything; only APPEND.
- Nodes may have only: "id", "attributes" (and keep existing "count" if present; never invent new count).
- All semantics must be encoded as short snake_case attributes (e.g., blue, brown, wood, metal, small, large).
- Add reasonable background nodes (e.g., sky, sun, clouds, grass, trees, road, sidewalk, water, sand, buildings, bench, path, cabinets, stove, sink, etc.) when helpful.
- Add only conventional spatial relations (in/on/above/under/next_to/near/front_of/behind/toward/touches).
- Keep inferred attributes consistent for the same node id across frames unless the text implies a change.
- Output must be a single JSON array of frames—nothing else.
"""


def _candidate_json_arrays(s: str):
    """Extract fragments from the text that look like JSON arrays (longest first)."""
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    cands = re.findall(r"\[[\s\S]*\]", s)
    return sorted(cands, key=len, reverse=True)


def _strip_trailing_commas(txt: str) -> str:
    """Fault tolerance for common tail comma issues."""
    return re.sub(r",\s*([\]}])", r"\1", txt)


def _try_parse_json(txt: str):
    """
    Try to parse the JSON. If it fails, 
    fix the comma at the end and try a few more times.
    """
    import json
    for _ in range(3):
        try:
            return json.loads(txt)
        except Exception:
            txt = _strip_trailing_commas(txt)
    raise


def _extract_json_array_or_raise(text: str):
    """
    Try to parse the candidate fragments one by one
    and return the first valid JSON array.
    """
    for cand in _candidate_json_arrays(text):
        try:
            obj = _try_parse_json(cand)
            if isinstance(obj, list):
                return obj
        except Exception:
            continue
    raise ValueError("No valid JSON array found in completion")


def safe_parse_json_array(text: str):
    """
    First, go through the candidate array.
    If it fails, perform a coarse slice as a fallback
    """
    try:
        return _extract_json_array_or_raise(text)
    except Exception:
        pass
    i = text.find("[")
    j = text.rfind("]")
    if i != -1 and j != -1 and i < j:
        snippet = text[i:j+1]
        for attempt in (snippet, _strip_trailing_commas(snippet)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, list):
                    return obj
            except Exception:
                continue
    raise ValueError("safe_parse_json_array: cannot recover a JSON array")


def _frame_map(frames):
    """Index with time to facilitate alignment between base and enriched."""
    return {f.get("time"): f for f in frames if isinstance(f, dict) and "time" in f}


def _validate_node_shape(base_nodes_by_id, nd, tlabel):
    """Node structure and field legitimacy verification."""
    if not isinstance(nd, dict):
        raise ValueError(f"[{tlabel}] node must be object, got {type(nd)}")
    keys = set(nd.keys())
    allowed = {"id", "attributes"}
    if nd.get("id") in base_nodes_by_id and ("count" in base_nodes_by_id[nd["id"]]):
        allowed.add("count")
    extra = keys - allowed
    if extra:
        raise ValueError(
            f"[{tlabel}] node {nd.get('id')} has extra keys: {extra}")
    if "attributes" in nd and not isinstance(nd["attributes"], list):
        raise ValueError(
            f"[{tlabel}] node {nd.get('id')} attributes must be a list")
    if ("count" in nd) and (nd["id"] in base_nodes_by_id) and ("count" not in base_nodes_by_id[nd["id"]]):
        raise ValueError(f"[{tlabel}] node {nd['id']} invented 'count'")


def diff_guard(base_frames, enriched_frames):
    """
    Only add, no rename/no delete.
    """
    if not isinstance(enriched_frames, list):
        raise ValueError("enriched result must be a JSON array")

    for fr in enriched_frames:
        if not isinstance(fr, dict):
            raise ValueError("each frame must be an object")
        if set(fr.keys()) != {"time", "scene", "nodes", "edges"}:
            raise ValueError(
                f"[{fr.get('time')}] frame keys must be exactly time/scene/nodes/edges")

    bmap, emap = _frame_map(base_frames), _frame_map(enriched_frames)
    if set(bmap.keys()) != set(emap.keys()):
        raise ValueError(
            f"frame times mismatch: base={sorted(bmap)} vs enriched={sorted(emap)}")

    for t in sorted(bmap.keys()):
        b, e = bmap[t], emap[t]
        base_nodes_by_id = {n["id"]: n for n in b.get(
            "nodes", []) if isinstance(n, dict) and "id" in n}
        e_nodes_by_id = {n["id"]: n for n in e.get(
            "nodes", []) if isinstance(n, dict) and "id" in n}

        # Node detection
        for nid, bn in base_nodes_by_id.items():
            if nid not in e_nodes_by_id:
                raise ValueError(f"[{t}] missing original node id: {nid}")
            en = e_nodes_by_id[nid]
            _validate_node_shape(base_nodes_by_id, en, tlabel=t)
            battrs = set(bn.get("attributes", []) or [])
            eattrs = set(en.get("attributes", []) or [])
            if not battrs.issubset(eattrs):
                raise ValueError(
                    f"[{t}] node {nid} lost attributes: {sorted(battrs - eattrs)}")
            if "count" in bn and en.get("count") != bn["count"]:
                raise ValueError(
                    f"[{t}] node {nid} changed count: {bn['count']} -> {en.get('count')}")
            if "count" not in bn and "count" in en:
                raise ValueError(
                    f"[{t}] node {nid} invented count in enriched")

        # Edge detection
        base_edges = {tuple(x) for x in (b.get("edges") or [])}
        enr_edges = {tuple(x) for x in (e.get("edges") or [])}
        for trip in base_edges:
            if trip not in enr_edges:
                raise ValueError(f"[{t}] missing original edge: {trip}")


def enrich(sg, text=None, model="gpt-5-mini", max_retries=3):
    """
    Enrich a base scene graph with inferred attributes/relations.
    """
    sg_json = json.dumps(sg, ensure_ascii=False, indent=2)
    text_raw = (text or "").strip()

    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT.format(
                        text_raw, sg_json)},
                ]
            )
            content = (resp.choices[0].message.content or "").strip()

            enriched = safe_parse_json_array(content)
            diff_guard(sg, enriched)
            return enriched

        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err else RuntimeError(
        "enrich failed without exception object")
