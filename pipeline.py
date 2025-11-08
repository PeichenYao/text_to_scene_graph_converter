# pipeline.py

import os
import re
import csv
import json
import spacy


def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def _left_finite_subject(v):
    sent = v.sent
    left_verbs = [
        t for t in sent
        if t.i < v.i and (t.pos_ in {"VERB", "AUX"})
        and (t.tag_ in {"VBZ", "VBP", "VBD"} or "Fin" in t.morph.get("VerbForm"))
    ]
    for cand in reversed(left_verbs):
        s = next((c for c in cand.children if c.dep_ in {
                 "nsubj", "nsubjpass"}), None)
        if s is not None:
            return s
    return None


def sg_rules():
    """
    Three types of predicates and dynamic attributes are identified through regular expressions.
    """
    rules = {
        "locative_predicate_regex":
            r"(?:^|_)(in|on|under|over|above|below|near|behind|front(?:_of)?|inside|outside|within|at|around|between|across|through|onto|into)$",
        "motion_from_predicate_regex":
            r"(?:^|_)(down|off|from|away)$|^(leave|descend|dismount|get(?:s|ting)?_off)(?:_|$)",
        "motion_toward_predicate_regex":
            r"(?:^|_)(to|toward|towards)$|^(approach|head|move|walk|run|go)(?:s|ed|ing)?(?:_|$)",
        "dynamic_attribute_regex":
            r"^(?:[a-z]+ing|[a-z]+(?:s|es))(?:_|$)",
        "normalize_mapping": {
            "towards": "toward", "onto": "on", "upon": "on",
            "into": "in", "inside": "in", "beside": "next_to", "nearby": "near"
        },
        "mutex_groups": [["sitting", "standing", "walking", "running", "jumping"]],
    }
    rules["_compiled"] = {k: re.compile(
        v) for k, v in rules.items() if k.endswith("_regex")}
    return rules


SG_RULES = sg_rules()


nlp = spacy.load("en_core_web_trf")


VERB_TAGS = {"VB", "VBD", "VBP", "VBZ", "VBG", "VBN"}
EXCLUDE_LEMMAS = {"be", "have", "do"}
EXCLUDE_TAGS = {"VBG", "VBN"}

ADV_PARTICLES = {"back", "down", "up", "away", "out", "off", "over", "around"}

TEMPORAL_CUES = {
    "suddenly", "then", "after", "before", "once", "later", "soon",
    "immediately", "meanwhile", "eventually", "next"
}

EVENT_CLUSTER_MAX_GAP = 12      # The maximum allowed token distance between verbs within the same sentence
EVENT_CLUSTER_SOFT_MAX = 2      # How many heads can a cluster retain at most?
EVENT_SHARE_PREPS = {"to", "toward", "towards", "into", "in", "on", "at", "inside",
                     "outside", "onto", "over", "under", "near", "behind", "front_of", "around"}


def _subject_lemma(v):
    s = get_subject(v)
    if s is None:
        s = _left_finite_subject(v)
    return s.lemma_.lower() if s is not None else None


def _pp_pairs(v):
    pairs = set()
    for p in (c for c in v.children if c.dep_ == "prep"):
        pobj = next((x for x in p.children if x.dep_ == "pobj"), None)
        if pobj is not None:
            pairs.add((p.lemma_.lower(), pobj.lemma_.lower()))
    return pairs


def _cluster_heads_by_event(heads, sent, max_gap=EVENT_CLUSTER_MAX_GAP, soft_max=EVENT_CLUSTER_SOFT_MAX):
    """
    Greedy clustering is performed from left to right. Entities that are the same or share the same location/goal are regarded as belonging to the same event cluster.
    Additional constraints are imposed that the distance between adjacent heads does not exceed max_gap, and there are no significant temporal prompt words crossing within the cluster.
    """
    doc = sent.doc
    clusters, cur = [], []
    last = None
    for h in heads:
        if not cur:
            cur = [h]
            last = h
            continue

        same_subj = (_subject_lemma(h) == _subject_lemma(last))
        share_goal = bool(_pp_pairs(h) & _pp_pairs(last) & {
                          (p, o) for p, o in _pp_pairs(last) if p in EVENT_SHARE_PREPS})
        L1, R1 = span_LR_for_head(last)
        L2, R2 = span_LR_for_head(h)
        near_ok = (h.i - last.i) <= max_gap
        cue_break = (first_temporal_cue_index(
            doc, min(L1, L2), max(R1, R2)) is not None)

        if (same_subj or share_goal) and near_ok and not cue_break:
            cur.append(h)
            last = h
            if len(cur) >= soft_max:
                clusters.append(cur)
                cur = []
                last = None
        else:
            clusters.append(cur)
            cur = [h]
            last = h

    if cur:
        clusters.append(cur)
    return clusters


def phrasal(v):
    base = v.lemma_.lower()
    parts_prt = [c.text.lower() for c in v.children if c.dep_ == "prt"]
    parts_adv = [c.lemma_.lower() for c in v.children if c.dep_ ==
                 "advmod" and c.lemma_.lower() in ADV_PARTICLES]
    parts = parts_adv + parts_prt
    return base if not parts else f"{base}_{'_'.join(parts)}"


def is_finite_verb(tok):
    if tok.lemma_.lower() in EXCLUDE_LEMMAS:
        return False
    if tok.tag_ in EXCLUDE_TAGS:
        return False
    is_verbish = (tok.pos_ in {"VERB", "AUX"}) or (tok.tag_ in VERB_TAGS)
    if not is_verbish:
        return False
    is_finite = (tok.tag_ in {"VBZ", "VBP", "VBD"}) or (
        "Fin" in tok.morph.get("VerbForm"))
    return is_finite


def pick_heads(sent):
    cands = []
    for t in sent:
        if not is_finite_verb(t):
            continue
        has_subject = any(ch.dep_ in {"nsubj", "nsubjpass"}
                          for ch in t.children)
        if t.dep_ in {"ROOT", "conj", "parataxis"} or has_subject:
            cands.append(t)
    if not cands:
        r = sent.root
        if is_finite_verb(r):
            cands = [r]
        else:
            first_vb = next((t for t in sent if t.tag_ in VERB_TAGS and t.lemma_.lower(
            ) not in EXCLUDE_LEMMAS), None)
            if first_vb:
                cands = [first_vb]
    return sorted(cands, key=lambda x: x.i)


def span_LR_for_head(head):
    toks = set(t.i for t in head.subtree)
    for c in head.children:
        if c.dep_ in {"xcomp", "ccomp"} and c.pos_ == "VERB":
            toks |= set(t.i for t in c.subtree)
    return min(toks), max(toks)


def first_cc_index(head):
    ccs = [c.i for c in head.children if c.dep_ == "cc"]
    return min(ccs) if ccs else None


def first_cc_index_anywhere(head):
    ccs = [t.i for t in head.subtree if t.dep_ ==
           "cc" and t.lemma_.lower() == "and"]
    return min(ccs) if ccs else None


def first_temporal_cue_index(doc, L, R):
    idx = None
    for t in doc[L+1:R+1]:
        if t.lemma_.lower() in TEMPORAL_CUES and t.pos_ in {"ADV", "SCONJ", "ADP"}:
            idx = t.i
            break
    return idx


def _ensure_sentence_frames(doc, frames):
    out = []
    for sent in doc.sents:
        has = any((fr["span_start"] >= sent.start and fr["span_end"]
                  < sent.end) for fr in frames)
        if has:
            for fr in frames:
                if fr["span_start"] >= sent.start and fr["span_end"] < sent.end:
                    out.append(fr)
        else:
            out.append({
                "time": "T?",
                "triggers": [],
                "span_start": sent.start,
                "span_end":   sent.end - 1,
                "span_text":  sent.text,
                "sentence":   sent.text,
                "cuts":       [],
                "head_index": sent.root.i,
                "temporal_cues": []
            })

    out.sort(key=lambda fr: fr["span_start"])
    for i, fr in enumerate(out, 1):
        fr["time"] = f"T{i}"
    return out


def segment_frames(text):
    doc = nlp(text)
    frames = []
    t = 1
    for sent in doc.sents:
        heads = pick_heads(sent)
        if not heads:
            continue

        clusters = _cluster_heads_by_event(heads, sent,
                                           max_gap=EVENT_CLUSTER_MAX_GAP,
                                           soft_max=EVENT_CLUSTER_SOFT_MAX)

        for ci, hs in enumerate(clusters):
            LR = [span_LR_for_head(h) for h in hs]
            L = min(x for x, _ in LR)
            R = max(y for _, y in LR)
            cut_marks = []

            if ci + 1 < len(clusters):
                next_L = min(span_LR_for_head(h)[0] for h in clusters[ci+1])
                if next_L - 1 < R:
                    R = next_L - 1
                    cut_marks.append("next_head")

            cue_i = first_temporal_cue_index(doc, L, R)
            if cue_i is not None and cue_i - 1 < R:
                R = cue_i - 1
                cut_marks.append("temporal_cue")

            if R < L:
                L = R = min(h.i for h in hs)

            span_text = doc[L:R+1].text
            triggers = [phrasal(h) for h in hs]
            head_idx = min(h.i for h in hs)

            frames.append({
                "time": f"T{t}",
                "triggers": triggers,
                "span_start": L,
                "span_end":   R,
                "span_text":  span_text,
                "sentence":   sent.text,
                "cuts":       cut_marks,
                "head_index": head_idx,
                "temporal_cues": [w.text for w in doc[L:R+1] if w.lemma_.lower() in TEMPORAL_CUES]
            })
            t += 1

    return frames


def lem(tok):
    return tok.lemma_.lower()


def pred(v, prep=None):
    base = v.text.lower()
    prts = [c.text.lower() for c in v.children if c.dep_ == "prt"]
    if prts:
        base = f"{base}_{'_'.join(prts)}"
    if prep is not None:
        base = f"{base}_{prep.lemma_.lower()}"
    return base


def get_subject(v):
    for c in v.children:
        if c.dep_ in {"nsubj", "nsubjpass"}:
            return c
    if v.dep_ in {"conj", "xcomp", "ccomp"} and v.head.pos_ in {"VERB", "AUX"}:
        return get_subject(v.head)
    return None


def compose_intransitive_pred(v):
    base = v.text.lower()
    parts = [c.text.lower() for c in v.children if c.dep_ == "prt"]
    parts += [
        c.lemma_.lower()
        for c in v.children
        if c.dep_ == "advmod" and c.tag_ == "RB" and not c.lemma_.lower().endswith("ly")
    ]
    empty_preps = [
        p for p in v.children
        if p.dep_ == "prep" and not any(ch.dep_ in {"pobj", "pcomp"} for ch in p.children)
    ]
    parts += [p.lemma_.lower() for p in empty_preps]
    if parts:
        base = f"{base}_{'_'.join(parts)}"
    return base


def _resolve_pronoun_index(doc, idx):
    if idx is None or idx < 0 or idx >= len(doc):
        return idx
    tok = doc[idx]
    if tok.pos_ != "PRON" or tok.lower_ not in {"it", "them", "him", "her"}:
        return idx
    sent = tok.sent
    lefts = [t for t in sent if t.i < tok.i and t.pos_ in {"NOUN", "PROPN"}]
    if lefts:
        return lefts[-1].i
    # Previous sentence ends with a backward step
    try:
        sents = list(doc.sents)
        k = sents.index(sent)
        if k > 0:
            prev = sents[k-1]
            cand = [t for t in prev if t.pos_ in {"NOUN", "PROPN"}]
            if cand:
                return cand[-1].i
    except Exception:
        pass
    return idx


def postproc_add_xcomp_triples(doc, triples):
    existing = {(t["s"], t["p"], t["o"], t["head"]) for t in triples}
    out = list(triples)

    def _push(s_tok, v_tok, o_tok, pnode=None):
        s, o = s_tok.lemma_.lower(), o_tok.lemma_.lower()
        p = pred(v_tok, pnode)
        key = (s, p, o, v_tok.i)
        if key not in existing:
            out.append(dict(s=s, p=p, o=o, head=v_tok.i,
                            s_i=s_tok.i, o_i=o_tok.i, sent_i=v_tok.sent.start))
            existing.add(key)

    for sent in doc.sents:
        for v in [t for t in sent if t.pos_ == "VERB"]:
            xcs = [c for c in v.children if c.dep_ ==
                   "xcomp" and c.pos_ == "VERB"]
            if not xcs:
                continue
            # The subject of "v" functions as the controller.
            subj = next((c for c in v.children if c.dep_ in {
                        "nsubj", "nsubjpass"}), None)
            if subj is None and v.dep_ in {"conj", "xcomp", "ccomp"} and v.head.pos_ in {"VERB", "AUX"}:
                # Upward inheritance
                h = v.head
                subj = next((c for c in h.children if c.dep_ in {
                            "nsubj", "nsubjpass"}), None)
            if subj is None:
                continue

            for xc in xcs:
                # Direct Object
                dobjs = [c for c in xc.children if c.dep_ in {
                    "dobj", "obj", "attr", "dative"}]
                for o in dobjs:
                    _push(subj, xc, o, None)
                # Prepositional Object
                for pnode in [c for c in xc.children if c.dep_ == "prep"]:
                    for o in [x for x in pnode.children if x.dep_ == "pobj"]:
                        _push(subj, xc, o, pnode)
    return out


def postproc_resolve_and_disambiguate(doc, triples):

    # Anaphora of pronouns
    for t in triples:
        if t.get("o_i") is not None:
            new_idx = _resolve_pronoun_index(doc, t["o_i"])
            if new_idx != t["o_i"]:
                t["o_i"] = new_idx
                t["o"] = doc[new_idx].lemma_.lower()

    by_key = {}
    for t in triples:
        if t["o"] == "__null__":
            continue
        key = (t["sent_i"], t["s"], t["o"])
        by_key.setdefault(key, []).append(t)

    keep = set()
    for key, cand in by_key.items():
        def _dist(tt):
            if tt.get("o_i") is None or tt.get("head") is None:
                return 10**9
            return abs(tt["head"] - tt["o_i"])
        cand.sort(key=_dist)
        best = cand[0]
        best_d = _dist(best)
        for t in cand:
            if _dist(t) == best_d:
                keep.add((t["s"], t["p"], t["o"], t["head"]))

    out = []
    seen = set()
    for t in triples:
        if t["o"] != "__null__":
            key = (t["s"], t["p"], t["o"], t["head"])
            if key not in keep:
                continue
        k2 = (t["s"], t["p"], t["o"], t["head"])
        if k2 in seen:
            continue
        seen.add(k2)
        out.append(t)
    return out


def extract_predicates_with_pos(doc):
    """
    Extract the relation sentence by sentence.
    """
    triples = []

    for sent in doc.sents:
        sent_cache = []

        verbs = [t for t in sent if t.pos_ == "VERB" and t.dep_ in {
            "ROOT", "conj", "xcomp", "ccomp"}]
        for v in verbs:
            # Subject
            subj = get_subject(v)

            # Passive
            agent = None
            if any(c.dep_ == "nsubjpass" for c in v.children):
                by = next((c for c in v.children if c.dep_ == "agent"), None)
                if by:
                    agent = next(
                        (x for x in by.children if x.dep_ == "pobj"), None)
            if agent:
                patient = next(
                    (c for c in v.children if c.dep_ == "nsubjpass"), None)
                if patient:
                    sent_cache.append(dict(
                        s=lem(agent), p=pred(v), o=lem(patient),
                        head=v.i, s_i=agent.i, o_i=patient.i, sent_i=sent.start
                    ))

            # Subject-Verb-Object
            objs = [c for c in v.children if c.dep_ in {
                "dobj", "obj", "attr", "dative"}]
            if subj and objs:
                for o in objs:
                    sent_cache.append(dict(
                        s=lem(subj), p=pred(v), o=lem(o),
                        head=v.i, s_i=subj.i, o_i=o.i, sent_i=sent.start
                    ))

            # Prepositional complement
            for pnode in [c for c in v.children if c.dep_ == "prep"]:
                pobjs = [x for x in pnode.children if x.dep_ == "pobj"]
                for o in pobjs:
                    if subj:
                        sent_cache.append(dict(
                            s=lem(subj), p=pred(v, pnode), o=lem(o),
                            head=v.i, s_i=subj.i, o_i=o.i, sent_i=sent.start
                        ))
                for dobj in objs:
                    if pnode.lemma_.lower() != "of":
                        sent_cache.append(dict(
                            s=lem(dobj), p=pnode.lemma_.lower(), o=lem(pobjs[0]) if pobjs else "__null__",
                            head=v.i, s_i=dobj.i, o_i=(pobjs[0].i if pobjs else None), sent_i=sent.start
                        ))

            # Indefinite candidate
            has_obj = bool(objs)
            has_pobj = any(p.dep_ == "prep" and any(
                x.dep_ == "pobj" for x in p.children) for p in v.children)
            if subj and (not has_obj) and (not has_pobj):
                sent_cache.append(dict(
                    s=lem(subj), p=compose_intransitive_pred(v), o="__null__",
                    head=v.i, s_i=subj.i, o_i=None, sent_i=sent.start
                ))

        # Noun + Preposition
        for np in [t for t in sent if t.pos_ in {"NOUN", "PROPN"}]:
            for pnode in [c for c in np.children if c.dep_ == "prep" and c.lemma_.lower() != "of"]:
                pobj = next(
                    (x for x in pnode.children if x.dep_ == "pobj"), None)
                if pobj:
                    sent_cache.append(dict(
                        s=lem(np), p=pnode.lemma_.lower(), o=lem(pobj),
                        head=pnode.i, s_i=np.i, o_i=pobj.i, sent_i=sent.start
                    ))

        # remove __null__
        have_obj = {(t["head"], t["s"]) for t in sent_cache if t["o"]
                    != "__null__" and t["s"] is not None}
        tmp1 = [t for t in sent_cache if not (
            t["o"] == "__null__" and (t["head"], t["s"]) in have_obj)]

        vprep_keys = set()
        for t in tmp1:
            if "_" in t["p"]:
                prep_tail = t["p"].split("_")[-1]
                if prep_tail:
                    vprep_keys.add((t["s"], prep_tail, t["o"], t["sent_i"]))
        tmp2 = []
        for t in tmp1:
            if "_" not in t["p"] and (t["s"], t["p"], t["o"], t["sent_i"]) in vprep_keys:
                continue
            tmp2.append(t)

        seen = set()
        for t in tmp2:
            key = (t["s"], t["p"], t["o"], t["head"])
            if key in seen:
                continue
            seen.add(key)
            triples.append(t)

    return triples


LABEL2TYPE = {
    "PERSON": "character",
    "GPE": "location", "LOC": "location", "FAC": "location"
}
PRONOUNS = {"it", "he", "she", "they", "him",
            "her", "them", "its", "his", "their", "theirs"}


def coarse_type(ent_label):
    return LABEL2TYPE.get(ent_label, "object")


def canon_id_from_mentions(mentions):
    propns = [m for m in mentions if any(t.pos_ == "PROPN" for t in m)]
    if propns:
        m = max(propns, key=lambda s: len(s.text))
        return _snake(m.text)
    heads = [m.root.lemma_.lower() for m in mentions]
    return _snake(max(set(heads), key=heads.count) if heads else "obj")


def collect_mentions(doc):
    mentions = []
    # NER
    for ent in doc.ents:
        if ent.root.pos_ == "PRON":
            continue
        mentions.append((ent, coarse_type(ent.label_)))
    taken = {(s.start, s.end) for s, _ in mentions}
    for nc in doc.noun_chunks:
        if (nc.start, nc.end) in taken:
            continue
        if nc.root.pos_ == "PRON":
            continue
        mentions.append((nc, "object"))
    return mentions


def _premerge_same_sentence(doc, mentions):
    groups = {}
    for i, (span, _t) in enumerate(mentions):
        key = (span.root.lemma_.lower(), span.root.head.sent.start)
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def link_definite_nps(doc, mentions):
    idx_by_start = sorted(range(len(mentions)),
                          key=lambda i: mentions[i][0].start)
    lemma_positions = {}
    link = {}

    for i in idx_by_start:
        span, _ = mentions[i]
        head = span.root
        det = next((c for c in head.children if c.dep_ == "det"), None)
        lemma = head.lemma_.lower()

        if det is not None and det.lemma_.lower() in {"the", "this", "that", "these", "those"}:
            cands = [j for j in (lemma_positions.get(
                lemma, []) or []) if mentions[j][0].end <= span.start]
            if cands:
                def _rank(j):
                    spj = mentions[j][0]
                    same_sent = 0 if spj.root.sent.start == span.root.sent.start else 1
                    dist = abs(span.start - spj.end)
                    return (same_sent, dist)
                cands.sort(key=_rank)
                link[i] = cands[0]

        lemma_positions.setdefault(lemma, []).append(i)

    return link


def _is_person_span(span):
    if any(t.ent_type_ == "PERSON" for t in span):
        return True
    if any(t.pos_ == "PROPN" for t in span) and not any(t.lower_ in {"it", "its"} for t in span):
        return True
    nouns = {t.lemma_.lower() for t in span if t.pos_ in {"NOUN", "PROPN"}}
    return bool(nouns & {"man", "woman", "boy", "girl", "person", "people", "child", "kid", "lady", "guy"})


def _np_is_plural(span):
    head = span.root
    if "Plur" in head.morph.get("Number"):
        return True
    if any(t.dep_ == "cc" and t.lemma_.lower() == "and" for t in span.subtree):
        return True
    lem = head.lemma_.lower()
    return lem in {"people", "children", "men", "women", "kids", "boys", "girls"}


def _pron_base_and_number(tok):
    w = tok.lemma_.lower()
    base = {"its": "it", "his": "he", "her": "she",
            "their": "they", "theirs": "they"}.get(w, w)
    num = "plur" if base in {"they"} else "sing"
    return base, num


def heuristic_coref(doc, mentions):
    mention_spans = [m for m, _ in mentions]
    idx_by_start = sorted(range(len(mention_spans)),
                          key=lambda i: mention_spans[i].start)

    def _covering_mention(tok_i):
        for i, (sp, _t) in enumerate(mentions):
            if sp.start <= tok_i < sp.end:
                return i
        return None

    def _mention_by_root_tok(tok):
        cand = [i for i, (sp, _t) in enumerate(mentions) if sp.root.i == tok.i]
        if cand:
            return cand[0]
        same_sent = [i for i, (sp, _t) in enumerate(mentions)
                     if sp.root.lemma_.lower() == tok.lemma_.lower() and sp.root.sent.start == tok.sent.start]
        if same_sent:
            return same_sent[0]
        any_lemma = [i for i, (sp, _t) in enumerate(
            mentions) if sp.root.lemma_.lower() == tok.lemma_.lower()]
        return any_lemma[0] if any_lemma else None

    def _nearest_left_finite_subject_in_sentence(pron):
        sent = pron.sent
        for v in reversed([t for t in sent
                           if t.i < pron.i and (t.pos_ in {"VERB", "AUX"})
                           and (t.tag_ in {"VBZ", "VBP", "VBD"} or "Fin" in t.morph.get("VerbForm"))]):
            subj = next((c for c in v.children if c.dep_ in {
                        "nsubj", "nsubjpass"}), None)
            if subj is not None and subj.pos_ in {"NOUN", "PROPN"}:
                return subj
        return None

    def _compatible(pron_tok, mspan):
        pbase, pnum = _pron_base_and_number(pron_tok)
        if pbase in {"he", "she"}:
            return _is_person_span(mspan)
        if pbase == "it":
            return not _is_person_span(mspan)
        if pbase == "they":
            return _np_is_plural(mspan)
        return True

    def _rank_key(pron, mspan):
        same_sent = (mspan.root.sent.start == pron.sent.start)
        is_subj = (mspan.root.dep_ in {"nsubj", "nsubjpass"})
        dist = abs(pron.i - mspan.end)
        pbase, _ = _pron_base_and_number(pron)
        prop_bonus = (1 if (pbase in {"he", "she"} and any(
            t.pos_ == "PROPN" for t in mspan)) else 0)
        plural_ok = (1 if (pbase == "they" and _np_is_plural(mspan)) else 0)
        return (0 if same_sent else 1, 0 if is_subj else 1, -(prop_bonus+plural_ok), dist)

    def find_antecedent(pron):
        w_base, _ = _pron_base_and_number(pron)

        cands = [i for i in idx_by_start if mention_spans[i].end <= pron.i]
        if not cands:
            return None

        if w_base in {"it", "he", "she", "they"} and (pron.dep_ == "poss" or pron.text.lower() in {"its", "his", "her", "their", "theirs"}):
            subj_tok = _nearest_left_finite_subject_in_sentence(pron)
            if subj_tok is not None:
                idx = _covering_mention(
                    subj_tok.i) or _mention_by_root_tok(subj_tok)
                if idx is not None:
                    return idx

        filt = [i for i in cands if _compatible(
            pron, mention_spans[i])] or cands
        filt.sort(key=lambda i: _rank_key(pron, mention_spans[i]))
        return filt[0]

    links = {}
    for tok in doc:
        if tok.lemma_.lower() in PRONOUNS:
            idx = find_antecedent(tok)
            if idx is not None:
                links[tok.i] = idx
    return links


def extract_entities(text):
    doc = nlp(text)
    mentions = collect_mentions(doc)

    pre_clusters = _premerge_same_sentence(doc, mentions)
    clusters, mention_to_cluster = [], {}

    for s in pre_clusters:
        cid = len(clusters)
        clusters.append(set(s))
        for i in s:
            mention_to_cluster[i] = cid

    if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
        span2idx = {(s.start, s.end): i for i, (s, _) in enumerate(mentions)}
        for c in doc._.coref_clusters:
            idxs = set()
            for m in c.mentions:
                k = (m.start, m.end)
                if k in span2idx:
                    idxs.add(span2idx[k])
            if idxs:
                target = None
                for i in idxs:
                    if i in mention_to_cluster:
                        target = mention_to_cluster[i]
                        break
                if target is None:
                    target = len(clusters)
                    clusters.append(set())
                for i in idxs:
                    clusters[target].add(i)
                    mention_to_cluster[i] = target

    # Proper noun heuristic
    pron_links = heuristic_coref(doc, mentions)
    for pron_i, ant_idx in pron_links.items():
        cid = mention_to_cluster.get(ant_idx, None)
        if cid is None:
            cid = len(clusters)
            clusters.append({ant_idx})
            mention_to_cluster[ant_idx] = cid
        clusters[cid].add(("PRON", pron_i))

    # definite NP Linking
    np_links = link_definite_nps(doc, mentions)
    for cur_idx, ant_idx in np_links.items():
        cid = mention_to_cluster.get(ant_idx, None)
        if cid is None:
            cid = len(clusters)
            clusters.append({ant_idx})
            mention_to_cluster[ant_idx] = cid
        clusters[cid].add(cur_idx)
        mention_to_cluster[cur_idx] = cid

    # The lone one's addition
    for i in range(len(mentions)):
        if i not in mention_to_cluster:
            clusters.append({i})

    # Generate nodes
    base_counter = {}
    nodes = []
    for cl in clusters:
        real_idxs = [i for i in cl if isinstance(i, int)]
        real_mentions = [mentions[i][0] for i in real_idxs]
        if not real_mentions:
            continue

        node_type = "object"
        for i in real_idxs:
            ent_lbls = {t.ent_type_ for t in mentions[i][0]}
            if "PERSON" in ent_lbls:
                node_type = "character"
                break
            if ent_lbls & {"GPE", "LOC", "FAC"}:
                node_type = "location"

        base = canon_id_from_mentions(real_mentions)
        base_counter[base] = base_counter.get(base, 0) + 1
        node_id = f"{base}#{base_counter[base]}"

        alias_texts = [m.text for m in real_mentions]
        pron_alias = []
        for item in cl:
            if isinstance(item, tuple) and item[0] == "PRON":
                pron_alias.append(doc[item[1]].text)
        alias_texts.extend(pron_alias)

        uniq_alias, seen = [], set()
        for a in alias_texts:
            s = _snake(a)
            if s not in seen:
                uniq_alias.append(s)
                seen.add(s)

        nodes.append(
            {"id": node_id, "type": node_type, "mentions": uniq_alias})

    return {"nodes": nodes}


# Attribute Mounting Window
ATTR_WINDOW_K = 3


def _attr_attach_with_window(head, mod, K=ATTR_WINDOW_K):
    """
    Only allow the modifiers to be attached to the left side of the same head noun within a close range (<= K)
    """
    # Only accept left-side modification
    if mod.i > head.i:
        return False
    if head.i - mod.i > K:
        return False
    doc = head.doc
    for j in range(mod.i + 1, head.i):
        t = doc[j]
        if t.dep_ == "cc" or t.text in {",", ";", ":"}:
            return False
    return True


def verb_ing(lemma):
    w = lemma.lower()
    if w.endswith("ie"):
        return w[:-2] + "ying"
    vowels = "aeiou"
    if len(w) >= 3 and w[-1] not in vowels+"wyx" and w[-2] in vowels and w[-3] not in vowels:
        return w + w[-1] + "ing"
    if w.endswith("e") and w not in {"be", "see", "flee"}:
        return w[:-1] + "ing"
    return w + "ing"


def phrase_from_np(head):
    adjs = [t.lemma_.lower() for t in head.lefts if t.dep_ == "amod"]
    core = head.lemma_.lower()
    return _snake(" ".join(adjs + [core]) if adjs else core)


def inherit_subject(v):
    s = next((c for c in v.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
    if s:
        return s
    if v.dep_ in {"conj", "xcomp", "advcl", "ccomp"} and v.head.pos_ in {"VERB", "AUX"}:
        return inherit_subject(v.head)
    return None


def extract_attributes_with_pos(doc):
    attrs = []
    def push(tgt, a, p): attrs.append(dict(target=tgt, attr=a, pos=p))

    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            for m in tok.lefts:
                if m.dep_ == "amod" and m.pos_ == "ADJ" and _attr_attach_with_window(tok, m, K=ATTR_WINDOW_K):
                    push(tok.lemma_.lower(), _snake(m.lemma_), m.i)

    for v in [t for t in doc if t.pos_ == "VERB" and t.lemma_.lower() != "be"]:
        subj = inherit_subject(v)
        if not subj or subj.pos_ not in {"NOUN", "PROPN"}:
            continue
        has_dobj = any(c.dep_ in {"dobj", "obj", "attr", "dative"}
                       for c in v.children)
        if not has_dobj:
            push(subj.lemma_.lower(), _snake(verb_ing(v.lemma_.lower())), v.i)

    for v in [t for t in doc if t.pos_ == "VERB" and t.lemma_.lower() != "be"]:
        subj = inherit_subject(v)
        left_subj = _left_finite_subject(v)
        if v.tag_ == "VBG" and left_subj is not None:
            subj = left_subj
        if not subj or subj.pos_ not in {"NOUN", "PROPN"}:
            continue

        obj = next((c for c in v.children if c.dep_ in {
                   "dobj", "obj"} and c.pos_ in {"NOUN", "PROPN"}), None)
        if not obj:
            for p in [c for c in v.children if c.dep_ == "prep"]:
                obj = next((y for y in p.children if y.dep_ ==
                           "pobj" and y.pos_ in {"NOUN", "PROPN"}), None) or obj
        if not obj:
            continue

        poss = next((ch for ch in obj.children if ch.dep_ == "poss"), None)

        if poss is not None:
            target_lemma = poss.lemma_.lower()
        else:
            target_lemma = subj.lemma_.lower()

        attrs.append(dict(
            target=target_lemma,
            attr=_snake(f"{verb_ing(v.lemma_.lower())}_{obj.lemma_.lower()}"),
            pos=v.i,
        ))

    for v in [t for t in doc if t.pos_ == "VERB" and t.tag_ == "VBG"]:
        obj = next((c for c in v.children if c.dep_ in {
                   "dobj", "obj"} and c.pos_ in {"NOUN", "PROPN"}), None)
        if not obj:
            for p in [c for c in v.children if c.dep_ == "prep"]:
                obj = next((y for y in p.children if y.dep_ ==
                           "pobj" and y.pos_ in {"NOUN", "PROPN"}), None) or obj

        target = None
        poss = next((ch for ch in (obj.children if obj else [])
                    if ch.dep_ == "poss"), None)

        if poss is None and obj is None:
            for j in range(v.i+1, min(len(doc), v.i+6)):
                cand = doc[j]
                if cand.pos_ in {"NOUN", "PROPN"}:
                    poss2 = next(
                        (ch for ch in cand.children if ch.dep_ == "poss"), None)
                    if poss2 is not None:
                        poss = poss2
                        obj = cand
                        break

        if poss is not None:
            target = poss.lemma_.lower()
        else:
            subj = next((c for c in v.children if c.dep_ in {
                        "nsubj", "nsubjpass"}), None) or inherit_subject(v)
            left_subj = _left_finite_subject(v)
            if left_subj is not None and (subj is None or subj.i > v.i):
                subj = left_subj
            if not subj or subj.pos_ not in {"NOUN", "PROPN"}:
                continue
            target = subj.lemma_.lower()

        if obj:
            push(target, _snake(
                f"{verb_ing(v.lemma_.lower())}_{obj.lemma_.lower()}"), v.i)
        else:
            push(target, _snake(verb_ing(v.lemma_.lower())), v.i)

    for v in [t for t in doc if t.pos_ == "VERB"]:
        subj = inherit_subject(v)
        if not subj or subj.pos_ not in {"NOUN", "PROPN"}:
            continue
        for p in [c for c in v.children if c.dep_ == "prep" and c.lemma_.lower() == "with"]:
            pobj = next((y for y in p.children if y.dep_ ==
                        "pobj" and y.pos_ in {"NOUN", "PROPN"}), None)
            if pobj:
                push(subj.lemma_.lower(), phrase_from_np(pobj), pobj.i)

    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            for p in tok.children:
                if p.dep_ == "prep" and p.lemma_.lower() == "of":
                    pobj = next((y for y in p.children if y.dep_ ==
                                "pobj" and y.pos_ in {"NOUN", "PROPN"}), None)
                    if pobj:
                        push(tok.lemma_.lower(),
                             f"of_{pobj.lemma_.lower()}".replace(" ", "_"), p.i)
                if p.dep_ == "prep" and p.lemma_.lower() == "with":
                    pobj = next((y for y in p.children if y.dep_ ==
                                "pobj" and y.pos_ in {"NOUN", "PROPN"}), None)
                    if pobj:
                        push(tok.lemma_.lower(), phrase_from_np(pobj), pobj.i)

    return attrs


def extract_quantities_with_pos(doc):
    out = []

    def set_count(head, count_str, pos):
        out.append(dict(target=head.lemma_.lower(), count=count_str, pos=pos))
    _NUM_WORDS = {"a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                  "dozen", "dozens", "several", "many", "few", "couple", "some", "each", "every", "all",
                  "both", "numerous", "multiple", "hundreds", "thousands", "lots"}
    for tok in doc:
        if tok.dep_ in {"nummod", "quantmod"} and tok.head.pos_ in {"NOUN", "PROPN"}:
            set_count(tok.head, tok.text.lower(), tok.i)
    for tok in doc:
        if tok.dep_ == "det" and tok.head.pos_ in {"NOUN", "PROPN"}:
            w = tok.text.lower()
            if w in _NUM_WORDS or w.isdigit():
                set_count(tok.head, w, tok.i)
    for i, tok in enumerate(doc):
        if tok.text.isdigit():
            nxt = doc[i+1] if i+1 < len(doc) else None
            if nxt is not None and nxt.pos_ in {"NOUN", "PROPN"}:
                set_count(nxt, tok.text, tok.i)
    for tok in doc:
        if tok.lemma_.lower() in _NUM_WORDS and tok.dep_ in {"amod", "advmod", "det"}:
            head = tok.head
            if head.pos_ in {"NOUN", "PROPN"}:
                set_count(head, tok.text.lower(), tok.i)
    return out


def _prune_orphan_nodes(nodes, edges):
    used = set()
    for s, p, o in edges:
        used.add(s)
        if o != "__null__":
            used.add(o)
    return [n for n in nodes if (n["id"] in used) or n.get("attributes")]


def _propagate_static_attributes_inplace(out_frames):
    mem = {}

    def is_static(a: str) -> bool:
        return re.search(r"\b\w+ing\b", a) is None
    for fr in out_frames:
        for nd in fr.get("nodes", []):
            if nd["id"] in mem:
                for a in mem[nd["id"]]:
                    if a not in nd["attributes"]:
                        nd["attributes"].append(a)
        for nd in fr.get("nodes", []):
            statics = [a for a in nd.get("attributes", []) if is_static(a)]
            if statics:
                mem.setdefault(nd["id"], set()).update(statics)

# The positional relations of propagation between frames
def _propagate_location_edges(out_frames):
    last_loc = {}
    for fr in out_frames:
        have_in = {(s, o) for (s, p, o) in fr["edges"] if p == "in"}
        for nd in fr["nodes"]:
            s = nd["id"]
            if s in last_loc and (s, last_loc[s]) not in have_in:
                fr["edges"].append([s, "in", last_loc[s]])
        for (s, p, o) in fr["edges"]:
            if p == "in":
                last_loc[s] = o


def _assign_to_frames(frames, alias2id, triples, attrs, counts):
    def norm(x): return alias2id.get(_snake(x), _snake(x))
    out_frames = []
    for fr in frames:
        L, R = fr["span_start"], fr["span_end"]
        node_attrs = {}
        node_count = {}
        edges, seen_e = [], set()

        # relation
        for t in triples:
            if L <= t["head"] <= R:
                s = norm(t["s"])
                o = "__null__" if t["o"] == "__null__" else norm(t["o"])
                key = (s, t["p"], o)
                if key not in seen_e:
                    edges.append([s, t["p"], o])
                    seen_e.add(key)
                node_attrs.setdefault(s, set())
                if o != "__null__":
                    node_attrs.setdefault(o, set())

        # Attributes and quantities
        for a in attrs:
            if L <= a["pos"] <= R:
                eid = norm(a["target"])
                node_attrs.setdefault(eid, set()).add(a["attr"])
        for c in counts:
            if L <= c["pos"] <= R:
                eid = norm(c["target"])
                node_count[eid] = node_count.get(eid, c["count"])

        # Assembly
        nid_set = set(node_attrs.keys())
        for s, p, o in edges:
            nid_set.add(s)
            if o != "__null__":
                nid_set.add(o)
        nodes = []
        for nid in sorted(nid_set):
            nd = {"id": nid, "attributes": sorted(node_attrs.get(nid, set()))}
            if nid in node_count:
                nd["count"] = node_count[nid]
            nodes.append(nd)

        nodes = _prune_orphan_nodes(nodes, edges)
        out_frames.append({
            "time": fr["time"],
            "scene": (fr.get("span_text") or fr.get("sentence") or "").strip(),
            "nodes": nodes,
            "edges": edges
        })
    return out_frames


def _token_table(doc):
    rows = []
    for i, t in enumerate(doc):
        rows.append(dict(i=i, text=t.text, lemma=t.lemma_, pos=t.pos_, tag=t.tag_,
                         dep=t.dep_, head=t.head.i, head_text=t.head.text))
    return rows


def _inspect_vbg(doc, alias2id):
    out = []
    def norm(x): return alias2id.get(_snake(x), _snake(x))
    for v in [t for t in doc if t.pos_ == "VERB" and t.tag_ == "VBG"]:
        obj = next((c for c in v.children if c.dep_ in {
                   "dobj", "obj"} and c.pos_ in {"NOUN", "PROPN"}), None)
        if not obj:
            for p in [c for c in v.children if c.dep_ == "prep"]:
                obj = next((y for y in p.children if y.dep_ ==
                           "pobj" and y.pos_ in {"NOUN", "PROPN"}), None) or obj
        poss = None
        if obj:
            poss = next((ch for ch in obj.children if ch.dep_ == "poss"), None)
        subj = next((c for c in v.children if c.dep_ in {
                    "nsubj", "nsubjpass"}), None) or inherit_subject(v)
        left_subj = _left_finite_subject(v)
        if left_subj is not None:
            subj = left_subj
        chosen = (poss.lemma_.lower() if poss is not None else (
            subj.lemma_.lower() if subj else None))
        out.append({
            "v_text": v.text, "v_i": v.i,
            "subj": (subj.text, subj.i) if subj else None,
            "obj": (obj.text, obj.i) if obj else None,
            "poss": (poss.text, poss.i) if poss else None,
            "chosen_raw": chosen, "chosen_norm": norm(chosen) if chosen else None
        })
    return out


def _build_alias_map(text):
    g = extract_entities(text)
    alias2id = {}
    for nd in g["nodes"]:
        base = nd["id"].split("#")[0]
        for alias in nd.get("mentions", []):
            alias2id[_snake(alias)] = base
        alias2id[base] = base
    return alias2id


def repair_triples_pp_objects(doc, triples, rules=None):
    rules = rules or SG_RULES
    sent_by_start = {s.start: s for s in doc.sents}

    def _is_pron(tok):
        return tok.pos_ == "PRON"

    def _pos_score(tok):
        return 1 if tok.pos_ in {"NOUN", "PROPN"} else 0

    fixed = []
    for t in triples:
        if t.get("o") != "__null__" or not _sg_is_toward(t.get("p", ""), rules):
            fixed.append(t)
            continue

        sent = sent_by_start.get(t.get("sent_i"))
        head_i = t.get("head")
        if sent is None or head_i is None or head_i < 0 or head_i >= len(doc):
            fixed.append(t)
            continue

        head_tok = doc[head_i]
        tail = (t.get("p") or "").split("_")[-1].lower()
        s_lem = (t.get("s") or "").lower()

        best = None
        cands = []

        # prep -> pobj
        for pnode in (c for c in head_tok.children if c.dep_ == "prep"):
            pobj = next((x for x in pnode.children if x.dep_ == "pobj"), None)
            if pobj is None:
                continue
            match = 1 if pnode.lemma_.lower() == tail else 0
            dist = abs(pobj.i - head_tok.i)
            cands.append(("prep", match, dist, pobj))

        # Recent Nouns
        for tok in sent:
            if tok.pos_ in {"NOUN", "PROPN"} and tok.lemma_.lower() != s_lem:
                dist = abs(tok.i - head_tok.i)
                cands.append(("near", 0, dist, tok))

        if cands:
            def _rank(z):
                origin, match, dist, tok = z
                pref = 0 if origin == "prep" else 1
                pron_penalty = 1 if _is_pron(tok) else 0
                return (pref, -match, pron_penalty, dist, -_pos_score(tok))
            cands.sort(key=_rank)
            best = cands[0][3]

        if best is not None:
            t2 = dict(t)
            t2["o"] = best.lemma_.lower()
            t2["o_i"] = best.i
            fixed.append(t2)
        else:
            fixed.append(t)

    return fixed


def postproc_polarity(doc, triples, drop_neg=True, modal_suffix="_maybe"):

    neg_heads, modal_heads = set(), set()

    for v in (t for t in doc if t.pos_ == "VERB"):
        has_neg = any(ch.dep_ == "neg" for ch in v.children)
        if not has_neg:
            for aux in (ch for ch in v.children if ch.dep_ in {"aux", "auxpass"}):
                if any(gg.dep_ == "neg" for gg in aux.children):
                    has_neg = True
                    break

        has_modal = any(aux.tag_ == "MD" for aux in v.children if aux.dep_ in {
                        "aux", "auxpass"})

        if has_neg:
            neg_heads.add(v.i)
        if has_modal:
            modal_heads.add(v.i)

    out = []
    for t in triples:
        h = t.get("head")
        if h is None:
            out.append(t)
            continue

        if drop_neg and h in neg_heads:
            continue

        if h in modal_heads and isinstance(t.get("p"), str):
            p = t["p"]
            if not p.endswith(modal_suffix):
                t = dict(t)
                t["p"] = f"{p}{modal_suffix}"

        out.append(t)

    return out


def _sg_norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def _sg_is_locative(pred: str, rules) -> bool:
    return bool(rules["_compiled"]["locative_predicate_regex"].search(_sg_norm(pred)))


def _sg_is_from(pred: str, rules) -> bool:
    return bool(rules["_compiled"]["motion_from_predicate_regex"].search(_sg_norm(pred)))


def _sg_is_toward(pred: str, rules) -> bool:
    return bool(rules["_compiled"]["motion_toward_predicate_regex"].search(_sg_norm(pred)))


def _pred_family(pred, rules):
    if _sg_is_from(pred, rules):
        return "from"
    if _sg_is_toward(pred, rules):
        return "toward"
    if _sg_is_locative(pred, rules):
        return "loc"
    return "other"


def _sgpost_backfill_last_object(frames, rules):

    last_obj = {}

    def _ensure_node(fr, nid):
        if nid == "__null__":
            return
        nodes = fr.get("nodes") or []
        if not any(isinstance(n, dict) and n.get("id") == nid for n in nodes):
            nodes.append({"id": nid, "attributes": []})
            fr["nodes"] = nodes

    for fi, fr in enumerate(frames):
        edges = fr.get("edges") or []
        before = sum(1 for _e in edges if isinstance(
            _e, (list, tuple)) and len(_e) == 3 and _e[2] == "__null__")

        # First, review the existing valid objects of the sentence, and update the memory.
        for s, p, o in edges:
            fam = _pred_family(p, rules)
            if o != "__null__":
                last_obj.setdefault(s, {})[fam] = o

        # Backfilling
        changed = 0
        for e in edges:
            if not (isinstance(e, (list, tuple)) and len(e) == 3):
                continue
            s, p, o = e
            if o != "__null__":
                continue
            fam = _pred_family(p, rules)
            if fam in {"toward", "loc"}:
                cand = None
                cand = (last_obj.get(s, {}) or {}).get(fam)
                if cand is None and fam == "loc":
                    cand = (last_obj.get(s, {}) or {}).get("loc")
                if cand:
                    e[2] = cand
                    _ensure_node(fr, cand)
                    changed += 1

# The object is missing/there is no object. Use other nodes in this frame to complete it.


def _sgpost_repair_pp_edges_in_frame(frame: dict, rules) -> None:
    ids = [n.get("id")
           for n in (frame.get("nodes") or []) if isinstance(n, dict)]
    for e in (frame.get("edges") or []):
        if not (isinstance(e, (list, tuple)) and len(e) == 3):
            continue
        s, p, o = e
        if o == "__null__" and _sg_is_toward(p, rules):
            cand = [nid for nid in ids if nid not in (s, "__null__")]
            if cand:
                e[2] = cand[0]

# The "cross-frame leaving" action lacks a subject, so the previous location is filled in as the subject.


def _sgpost_backfill_from_location(frames: list, rules) -> None:
    last_loc = {}
    for fr in frames:
        ids = {n.get("id")
               for n in (fr.get("nodes") or []) if isinstance(n, dict)}
        for e in (fr.get("edges") or []):
            if not (isinstance(e, (list, tuple)) and len(e) == 3):
                continue
            s, p, o = e
            if _sg_is_from(p, rules) and (o == "__null__" or o not in ids) and s in last_loc:
                e[2] = last_loc[s]
                e[1] = "leaves"
        for e in (fr.get("edges") or []):
            if not (isinstance(e, (list, tuple)) and len(e) == 3):
                continue
            s, p, o = e
            if _sg_is_locative(p, rules):
                last_loc[s] = o

# The morphological standardization of the verb in predicates


def _sg_stem_verb_token(w: str) -> str:
    w = (w or "").strip().lower()
    if not w:
        return ""

    # Minimum context
    doc_ctx = nlp(f"She {w}")
    tok = doc_ctx[1] if len(doc_ctx) > 1 else doc_ctx[0]
    lemma = tok.lemma_.lower()

    # Original context
    if lemma == w:
        doc_alone = nlp(w)
        if len(doc_alone):
            lemma = doc_alone[0].lemma_.lower()

    return lemma


def _sg_canon_multiword_prep(q: str) -> str:
    q = q.replace("in_the_front_of", "front_of").replace(
        "in_front_of", "front_of")
    q = q.replace("next to", "next_to")
    q = q.replace(" ", "_")
    return q


_SYMMETRIC_PREDS = {"next_to", "near"}


def _sgpost_symmetry_canonicalize_inplace(frame: dict):
    if not frame or "edges" not in frame:
        return
    new_edges, seen = [], set()
    for e in (frame.get("edges") or []):
        if not (isinstance(e, (list, tuple)) and len(e) == 3):
            continue
        s, p, o = e
        p_norm = (p or "").strip().lower().replace(" ", "_")
        if p_norm in _SYMMETRIC_PREDS:
            s2, o2 = (s, o) if s <= o else (o, s)
            key = (s2, p_norm, o2)
            if key in seen:
                continue
            seen.add(key)
            new_edges.append([s2, p_norm, o2])
        else:
            new_edges.append([s, p_norm, o])
    frame["edges"] = new_edges


def _sgpost_normalize_edges(scene_graph: list, rules=None):
    rules = rules or SG_RULES
    mapping = rules.get("normalize_mapping", {}) or {}

    def _canon_pred(p: str) -> str:
        q = _sg_canon_multiword_prep(_sg_norm(p))
        parts = q.split("_")
        if parts:
            parts[0] = _sg_stem_verb_token(parts[0])
        q = "_".join(parts)
        return mapping.get(q, q)

    for frame in scene_graph or []:
        edges = frame.get("edges") or []
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                e[1] = _canon_pred(e[1])
        _sgpost_symmetry_canonicalize_inplace(frame)
    return scene_graph

# Dynamic attribute determination


def _sg_is_dynamic_attr(attr: str, rules) -> bool:
    return bool(rules["_compiled"]["dynamic_attribute_regex"].match(_sg_norm(attr)))

# Static attribute inheritance across frames


def _sgpost_propagate_static_attrs(frames: list, rules) -> None:
    mem = {}
    for fr in frames:
        for n in (fr.get("nodes") or []):
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            attrs = set(n.get("attributes") or [])
            prev = mem.get(nid, set())
            keep = {a for a in prev if not _sg_is_dynamic_attr(a, rules)}
            n["attributes"] = sorted(attrs | keep)
            mem[nid] = {a for a in n["attributes"]
                        if not _sg_is_dynamic_attr(a, rules)}

# Extract a single frame state


def _sg_extract_frame_state(frame, rules):
    # location: extract locative border o from (s, p, o) 
    loc = {}
    for e in (frame.get("edges") or []):
        if not (isinstance(e, (list, tuple)) and len(e) == 3):
            continue
        s, p, o = e
        if o != "__null__" and _sg_is_locative(p, rules):
            loc[s] = o

    # actions: The dynamic attributes in the node attributes (such as "ing", "past participle", etc.) are used for mutual exclusion checks.
    actions = {}
    for n in (frame.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        aset = set(a for a in (n.get("attributes") or [])
                   if _sg_is_dynamic_attr(a, rules))
        if aset:
            actions[nid] = aset

    return {"location": loc, "actions": actions}

# Consistency Audit


def _sg_audit_consistency(frames, rules, tag="audit"):

    os.makedirs("artifacts", exist_ok=True)

    viol_rows = []
    traces = {}

    last_loc = {}     # nid -> loc_id
    prev_edges = []   # The previous frame of edges is used to assist in determining whether there is a movement transition.

    mutex_groups = [tuple(g) for g in (rules.get("mutex_groups") or [])]

    def _has_motion_transition(edges, s):
        for ss, p, o in edges:
            if ss == s and (_sg_is_toward(p, rules) or _sg_is_from(p, rules) or p == "leaves"):
                return True
        return False

    for i, fr in enumerate(frames or []):
        st = _sg_extract_frame_state(fr, rules)
        loc_now = st["location"]
        act_now = st["actions"]
        time_label = fr.get("time")

        # Frame-level mutually exclusive actions
        for nid, aset in act_now.items():
            for group in mutex_groups:
                present = [g for g in group if g in aset]
                if len(present) > 1:
                    viol_rows.append({
                        "frame_index": i, "time": time_label, "subject": nid,
                        "violation": "mutex_conflict",
                        "details": " / ".join(sorted(present)),
                        "prev_state": "", "curr_state": ",".join(sorted(aset))
                    })

        # Multiple positions in the same frame
        loc_bucket = {}
        for s, p, o in (fr.get("edges") or []):
            if _sg_is_locative(p, rules) and o != "__null__":
                loc_bucket.setdefault(s, set()).add(o)

        for s, targets in loc_bucket.items():
            if len(targets) > 1:
                viol_rows.append({
                    "frame_index": i, "time": time_label, "subject": s,
                    "violation": "multiple_locations_same_frame",
                    "details": " -> ".join(sorted(targets)),
                    "prev_state": last_loc.get(s, ""),
                    "curr_state": loc_now.get(s, "")
                })

        # Position jump but no transition signal
        for s, curr_loc in loc_now.items():
            prev_loc = last_loc.get(s)
            if prev_loc is not None and curr_loc != prev_loc:
                has_transition = _has_motion_transition(
                    prev_edges, s) or _has_motion_transition(fr.get("edges") or [], s)
                if not has_transition:
                    viol_rows.append({
                        "frame_index": i, "time": time_label, "subject": s,
                        "violation": "teleport_no_transition",
                        "details": "%s -> %s" % (prev_loc, curr_loc),
                        "prev_state": prev_loc, "curr_state": curr_loc
                    })

        # Record the trajectory
        all_ids = set(list(loc_now.keys()) + list(act_now.keys()))
        for nid in all_ids:
            traces.setdefault(nid, []).append({
                "frame_index": i,
                "time": time_label,
                "location": loc_now.get(nid),
                "actions": sorted(list(act_now.get(nid, [])))
            })

        # updated last_loc / prev_edges
        for s, v in loc_now.items():
            last_loc[s] = v
        prev_edges = fr.get("edges") or []

    # input
    with open(os.path.join("artifacts", "consistency_violations_%s.csv" % tag), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
                           "frame_index", "time", "subject", "violation", "details", "prev_state", "curr_state"])
        w.writeheader()
        w.writerows(viol_rows)

    with open(os.path.join("artifacts", "state_traces_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

# Dynamic attribute


def _sgpost_decay_dynamic_attrs(frames: list, rules, ttl: int = 1) -> None:
    dyn = {}
    mutex_groups = [tuple(g) for g in (rules.get("mutex_groups") or [])]

    def _apply_mutex(attrs: set[str]) -> set[str]:
        if not mutex_groups:
            return attrs
        A = set(attrs)
        for group in mutex_groups:
            present = [g for g in group if g in A]
            if len(present) > 1:
                keep = present[-1]
                A -= set(group)
                A.add(keep)
        return A

    for fr in frames:
        for nid, table in list(dyn.items()):
            for a in list(table.keys()):
                table[a] -= 1
                if table[a] < 0:
                    del table[a]
            if not table:
                del dyn[nid]

        for n in (fr.get("nodes") or []):
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            attrs = list(n.get("attributes") or [])
            keep, tab = [], dyn.setdefault(nid, {})
            for a in attrs:
                if _sg_is_dynamic_attr(a, rules):
                    tab[a] = ttl
                else:
                    keep.append(a)
            alive = {a for a, t in tab.items() if t >= 0}
            n["attributes"] = sorted(_apply_mutex(set(keep) | alive))


def _sgpost_process_scene_graph(frames: list, rules=None, ttl_dynamic: int = 1):
    rules = rules or SG_RULES
    if not isinstance(frames, list):
        return frames
    for fr in frames:
        _sgpost_repair_pp_edges_in_frame(fr, rules)
    _sgpost_backfill_from_location(frames, rules)
    _sgpost_backfill_last_object(frames, rules)
    _sgpost_propagate_static_attrs(frames, rules)
    _sgpost_decay_dynamic_attrs(frames, rules, ttl=ttl_dynamic)
    return frames


def _assign_persistent_ids_inplace(frames):
    """
    Assign the base id that first appears to this type of entity, and then reuse it for stability across frames.
    """

    base2pid = {}
    counter = {}

    for fr in frames or []:
        nodes = fr.get("nodes") or []
        edges = fr.get("edges") or []

        # local frame
        local = {}

        # Assign/reuse persistent IDs for nodes
        new_nodes = []
        for nd in nodes:
            base = nd.get("id")
            if base not in base2pid:
                counter[base] = counter.get(base, 0) + 1
                base2pid[base] = f"{base}#{counter[base]}"
            pid = base2pid[base]
            local[base] = pid

            attrs = nd.get("attributes") or []
            new_nd = {"id": pid, "attributes": list(attrs)}
            if "count" in nd:
                new_nd["count"] = nd["count"]
            new_nodes.append(new_nd)

        # Rewrite the sentence.
        new_edges = []
        for s, p, o in edges:
            s2 = local.get(s, s)
            o2 = o if o == "__null__" else local.get(o, o)
            new_edges.append([s2, p, o2])

        fr["nodes"] = new_nodes
        fr["edges"] = new_edges


def build_scene_graph(text, debug=False):
    doc = nlp(text)
    frames = segment_frames(text)
    frames = _ensure_sentence_frames(doc, frames)
    alias2id = _build_alias_map(text)

    triples = extract_predicates_with_pos(doc)
    attrs = extract_attributes_with_pos(doc)
    counts = extract_quantities_with_pos(doc)

    triples = postproc_add_xcomp_triples(doc, triples)
    triples = postproc_resolve_and_disambiguate(doc, triples)

    triples = repair_triples_pp_objects(doc, triples, rules=SG_RULES)
    triples = postproc_polarity(
        doc, triples, drop_neg=True, modal_suffix="_maybe")

    out_frames = _assign_to_frames(frames, alias2id, triples, attrs, counts)
    _assign_persistent_ids_inplace(out_frames)
    _propagate_static_attributes_inplace(out_frames)
    _propagate_location_edges(out_frames)

    _sg_audit_consistency(out_frames, SG_RULES, tag="pre")

    if not debug:
        return out_frames
    dbg = {
        "tokens": _token_table(doc),
        "frames_raw": frames,
        "alias2id": alias2id,
        "triples_with_pos": triples,
        "attrs_with_pos": attrs,
        "counts_with_pos": counts,
        "vbg_inspect": _inspect_vbg(doc, alias2id),
    }

    out_frames = _sgpost_process_scene_graph(
        out_frames, rules=SG_RULES, ttl_dynamic=1)
    out_frames = _sgpost_normalize_edges(out_frames, rules=SG_RULES)

    return out_frames, dbg

