from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PREDICATES = {
    "belongsToTopic": ("Article", "Topic"),
    "belongsToSubject": ("Article", "Subject"),
    "belongsToChapter": ("Article", "Chapter"),
    "relatedTo": ("Article", "Article"),
    "hasSource": ("Article", "SourceArticle"),
    "sourceArticleOf": ("SourceArticle", "SourceDocument"),
    "amendedBy": ("SourceArticle", "SourceArticle"),
    "supplementedBy": ("SourceArticle", "SourceArticle"),
    "repealedBy": ("SourceArticle", "SourceArticle"),
}
EXCLUDED_OFFICIAL_PREDICATES = {"amendedBy", "supplementedBy", "repealedBy"}
ENTITY_TYPES = ("Article", "Topic", "Subject", "Chapter", "SourceDocument", "SourceArticle")
ID_FIELDS = ("record_id", "subject_id", "topic_id", "article_id")


@dataclass(frozen=True)
class Record:
    values: Mapping[str, Any]

    def get(self, name: str) -> str:
        value = self.values.get(name, "")
        return "" if value is None or isinstance(value, (list, dict)) else str(value).strip()

    def raw(self, name: str) -> Any:
        return self.values.get(name, "")


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str
    evidence: Tuple[Tuple[str, str, str], ...]
    confidence: float = 1.0

    @property
    def fact_id(self) -> str:
        raw = "\t".join((self.subject, self.predicate, self.object))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def record_id(self) -> str:
        return self.evidence[0][0]

    @property
    def field(self) -> str:
        return self.evidence[0][1]


def _entity_id(kind: str, value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip())
    encoded = urllib.parse.quote(normalized, safe=":_-.")
    return f"{kind.lower()}:{encoded}" if encoded else ""


def _stable_id(kind: str, value: str) -> str:
    return _entity_id(kind, value)


def _text(record: Record, *fields: str) -> str:
    return " ".join(record.get(field) for field in fields if record.get(field)).strip()


def _record_id(record: Record, index: int) -> str:
    return record.get("record_id") or f"row-{index}"


def normalize_citation(value: str) -> str:
    """Canonicalize article aliases without discarding compound components."""
    normalized = urllib.parse.unquote(str(value)).strip().lower()
    normalized = re.sub(r"^(?:điều|dieu)[\s_:-]*", "", normalized)
    normalized = normalized.replace("_", ".")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^0-9a-z.]+$", "", normalized)
    return normalized


def load_records(input_path: Path, limit: Optional[int] = None) -> List[Record]:
    """Load JSONL, CSV, or a Hugging Face dataset directory/export."""
    records: List[Record] = []
    if input_path.suffix.lower() in {".jsonl", ".json"}:
        with input_path.open(encoding="utf-8") as handle:
            if input_path.suffix.lower() == ".jsonl":
                source = (json.loads(line) for line in handle if line.strip())
            else:
                parsed = json.load(handle)
                source = parsed if isinstance(parsed, list) else parsed.get("train", [])
            for item in source:
                records.append(Record({str(k): v for k, v in item.items()}))
                if limit and len(records) >= limit:
                    break
        return records
    if input_path.suffix.lower() == ".csv":
        with input_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                records.append(Record(row))
                if limit and len(records) >= limit:
                    break
        return records
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "A .csv/.json/.jsonl input is required unless the optional 'datasets' package is installed."
        ) from exc
    dataset = load_dataset("tmquan/phapdien-moj-gov-vn", "documents", split="train")
    for row in dataset:
        records.append(Record({str(k): v for k, v in row.items()}))
        if limit and len(records) >= limit:
            break
    return records


def _source_token(text: str) -> str:
    url = next(iter(re.findall(r"https?://\S+", text)), "")
    if url:
        parsed = urllib.parse.urlsplit(url.rstrip(".,)"))
        query = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)))
        fragment = urllib.parse.quote(urllib.parse.unquote(parsed.fragment), safe="_-.")
        token = f"{parsed.netloc}{parsed.path}"
        if query:
            token += f"?{query}"
        if fragment:
            token += f"#{fragment}"
        return token
    compact = re.sub(r"\s+", " ", text).strip(" .,:;")
    return compact[:180]


def _article_candidates(text: str) -> Iterator[str]:
    pattern = r"(?i)\b(?:điều|dieu)[\s_:-]*([0-9]+(?:[._][0-9A-Za-z]+)+|[0-9]+[a-z]?)\b"
    for match in re.finditer(pattern, text):
        yield normalize_citation(match.group(1))


def _source_entries(value: Any) -> Iterator[Tuple[str, str]]:
    """Yield (href, associated text) from structured or legacy source_links."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                href = str(item.get("href") or item.get("url") or "").strip()
                label = str(item.get("text") or item.get("title") or "").strip()
                if href:
                    yield href, label
            elif item:
                yield from _source_entries(item)
        return
    if isinstance(value, Mapping):
        href = str(value.get("href") or value.get("url") or "").strip()
        if href:
            yield href, str(value.get("text") or value.get("title") or "").strip()
        return
    text = "" if value is None else str(value)
    for href in re.findall(r"https?://[^\s<>\]\"']+", text):
        yield href.rstrip(".,);"), ""


def _impact_targets(text: str, phrase: str) -> Iterator[str]:
    pattern = re.compile(
        rf"(?i){re.escape(phrase)}[^.;]{{0,120}}?(?:bởi|theo)\s+([^.;]+)"
    )
    for match in pattern.finditer(text):
        for segment in re.split(r"\s*(?:;|,\s*(?=Điều\b))\s*", match.group(1)):
            segment = segment.strip(" ,")
            if segment:
                yield segment


def extract_graph(records: Sequence[Record]) -> Tuple[
    List[Dict[str, str]], List[Fact], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]
]:
    entities: Dict[str, Dict[str, str]] = {}
    facts: Dict[Tuple[str, str, str], Fact] = {}
    unresolved: List[Dict[str, str]] = []
    aliases: List[Dict[str, str]] = []
    article_lookup: Dict[str, List[str]] = defaultdict(list)
    seen_ids: Dict[str, int] = {}
    accepted_indices = set()
    missing_record_ids = 0
    citation_stats = Counter()
    citation_formats = Counter()

    for index, record in enumerate(records):
        explicit_id = record.get("record_id")
        record_id = _record_id(record, index)
        if not explicit_id:
            missing_record_ids += 1
        if record_id in seen_ids:
            unresolved.append({"source_record_id": record_id, "reference_text": "",
                               "reason": f"duplicate record_id; first seen at row {seen_ids[record_id]}",
                               "field": "record_id"})
            continue
        seen_ids[record_id] = index
        accepted_indices.add(index)
        article = _stable_id("Article", record_id)
        for alias in (record.get("article_id"), record.get("article_anchor"), record.get("article_title")):
            if alias:
                normalized = normalize_citation(alias)
                article_lookup[normalized].append(article)
                aliases.append({"alias": alias, "normalized_alias": normalized,
                                "entity_id": article, "source_record_id": record_id})

    def add_entity(kind: str, value: str, record_id: str) -> str:
        entity_id = _stable_id(kind, value)
        if not entity_id:
            return ""
        entities.setdefault(entity_id, {
            "entity_id": entity_id,
            "entity_type": kind,
            "label": value.strip(),
            "source_record_id": record_id,
        })
        return entity_id

    def add_fact(subject: str, predicate: str, obj: str, provenance_id: str, field: str, evidence: str,
                 confidence: float = 1.0) -> None:
        if not subject or not obj or predicate not in PREDICATES:
            return
        key = (subject, predicate, obj)
        current = facts.get(key)
        item = (provenance_id, field, evidence)
        if current:
            facts[key] = Fact(subject, predicate, obj, current.evidence + (item,), confidence)
        else:
            facts[key] = Fact(subject, predicate, obj, (item,), confidence)

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        if index not in accepted_indices:
            continue
        article_key = record.get("article_id") or record.get("article_anchor") or record_id
        article = add_entity("Article", record_id, record_id)
        subject = add_entity("Subject", f"{record.get('subject_id')}:{record.get('subject_title')}", record_id) \
            if record.get("subject_id") and record.get("subject_title") else ""
        topic = add_entity("Topic", f"{record.get('topic_id')}:{record.get('topic_title')}", record_id) \
            if record.get("topic_id") and record.get("topic_title") else ""
        chapter = add_entity(
            "Chapter",
            f"{record.get('subject_id')}:{record.get('topic_id')}:{record.get('chapter_title')}",
            record_id,
        ) if record.get("subject_id") and record.get("topic_id") and record.get("chapter_title") else ""
        if not article:
            unresolved.append({"source_record_id": record_id, "reference_text": "", "reason": "missing article identity",
                               "field": "article_id"})
            continue
        evidence = record.get("article_title") or record.get("content_text") or article_key
        add_fact(article, "belongsToSubject", subject, record_id, "subject_title", record.get("subject_title"))
        add_fact(article, "belongsToTopic", topic, record_id, "topic_title", record.get("topic_title"))
        add_fact(article, "belongsToChapter", chapter, record_id, "chapter_title", record.get("chapter_title"))

        source_note = record.get("source_note_text")
        source_entries = list(_source_entries(record.raw("source_links")))
        if not source_entries and source_note:
            source_entries = [("", source_note)]
        for href, label in source_entries:
            source_text = " ".join(part for part in (label, href, source_note) if part)
            source_token = _source_token(href or source_note)
            source_article = add_entity("SourceArticle", source_token, record_id) if source_token else ""
            source_document = add_entity("SourceDocument", source_note or label, record_id)
            if source_article:
                add_fact(article, "hasSource", source_article, record_id, "source_links", source_text)
                add_fact(source_article, "sourceArticleOf", source_document, record_id, "source_links", source_text)
            else:
                unresolved.append({"source_record_id": record_id, "reference_text": source_text,
                                   "reason": "source link is missing or malformed", "field": "source_links"})

        related = record.get("related_note_text")
        for citation in _article_candidates(related):
            candidates = sorted(set(article_lookup.get(citation, [])))
            citation_stats["extracted"] += 1
            citation_formats["compound" if "." in citation else "simple"] += 1
            if len(candidates) == 1 and candidates[0] != article:
                add_fact(article, "relatedTo", candidates[0], record_id, "related_note_text", related, 0.8)
                citation_stats["resolved"] += 1
            elif len(candidates) > 1:
                citation_stats["ambiguous"] += 1
                unresolved.append({"source_record_id": record_id, "reference_text": f"Điều {citation}",
                                   "reason": "article citation is ambiguous across records", "field": "related_note_text"})
            else:
                citation_stats["unresolved"] += 1
                unresolved.append({"source_record_id": record_id, "reference_text": f"Điều {citation}",
                                   "reason": "article citation did not resolve in corpus", "field": "related_note_text"})

        clauses = [clause.strip() for clause in re.split(r"[;\n]", source_note) if clause.strip()]
        for phrase, predicate in (
            ("được sửa đổi", "amendedBy"),
            ("được bổ sung", "supplementedBy"),
            ("bị bãi bỏ", "repealedBy"),
        ):
            for clause in clauses:
                if phrase not in clause.lower():
                    continue
                targets = list(_impact_targets(clause, phrase))
                for target_text in targets:
                    target_token = _source_token(target_text)
                    target = add_entity("SourceArticle", target_token, record_id) if target_token else ""
                    current_source = add_entity("SourceArticle", _source_token(clause), record_id)
                    if current_source and target and target != current_source:
                        add_fact(current_source, predicate, target, record_id, "source_note_text", target_text, 0.7)
                    else:
                        unresolved.append({"source_record_id": record_id, "reference_text": target_text or source_note,
                                           "reason": f"{phrase} annotation lacks a resolvable target", "field": "source_note_text"})
                if not targets:
                    unresolved.append({"source_record_id": record_id, "reference_text": source_note,
                                       "reason": f"{phrase} annotation lacks a resolvable target", "field": "source_note_text"})

    metrics = {
        "missing_record_ids": missing_record_ids,
        "duplicate_record_ids": sum(1 for row in unresolved if row["field"] == "record_id"),
        "citations_extracted": citation_stats["extracted"],
        "citations_resolved": citation_stats["resolved"],
        "citations_ambiguous": citation_stats["ambiguous"],
        "citations_unresolved": citation_stats["unresolved"],
        "citation_resolution_rate": (
            citation_stats["resolved"] / citation_stats["extracted"]
            if citation_stats["extracted"] else 0.0
        ),
        "citation_formats": dict(citation_formats),
    }
    return list(entities.values()), list(facts.values()), unresolved, aliases, metrics


def _split_facts(facts: Sequence[Fact], seed: int, validation_ratio: float, test_ratio: float) -> Dict[str, List[Fact]]:
    target = [fact for fact in facts if fact.predicate == "relatedTo"]
    background = [fact for fact in facts if fact.predicate != "relatedTo"]
    groups: Dict[Tuple[str, str], List[Fact]] = defaultdict(list)
    for fact in target:
        groups[tuple(sorted((fact.subject, fact.object)))].append(fact)
    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)
    test_target = int(len(target) * test_ratio)
    valid_target = int(len(target) * validation_ratio)
    test: List[Fact] = []
    valid: List[Fact] = []
    train: List[Fact] = []
    for group in grouped:
        if len(test) < test_target:
            test.extend(group)
        elif len(valid) < valid_target:
            valid.extend(group)
        else:
            train.extend(group)
    return {
        "background": background,
        "train": train,
        "valid": valid,
        "test": test,
    }


def _write_rows(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rules(path: Path) -> None:
    # Soft topical coherence is an empirical heuristic, not a legal implication.
    path.write_text(
        "0.5 !belongsToTopic(A, T) v !belongsToTopic(B, T) v relatedTo(A, B)\n"
        "0.5 !belongsToSubject(A, S) v !belongsToSubject(B, S) v relatedTo(A, B)\n",
        encoding="utf-8",
    )


def build_bundle(records: Sequence[Record], output_dir: Path, seed: int = 42,
                 validation_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entities, facts, unresolved, aliases, resolution_stats = extract_graph(records)
    official_facts = [fact for fact in facts if fact.predicate not in EXCLUDED_OFFICIAL_PREDICATES]
    splits = _split_facts(official_facts, seed, validation_ratio, test_ratio)
    entity_ids = {row["entity_id"] for row in entities}
    for split_name in ("background", "train", "valid", "test"):
        for fact in splits[split_name]:
            if fact.subject not in entity_ids or fact.object not in entity_ids:
                raise ValueError(f"Fact has an unknown entity: {fact}")

    _write_rows(output_dir / "entities.csv", entities, ("entity_id", "entity_type", "label", "source_record_id"))
    evidence_rows = [{
        "fact_id": fact.fact_id, "source_record_id": record_id, "field": field,
        "original_text": evidence, "evidence_type": "extracted"
    } for fact in facts for record_id, field, evidence in fact.evidence]
    _write_rows(output_dir / "facts.csv", [{
        "fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate,
        "object": fact.object, "source_record_id": fact.record_id, "confidence": fact.confidence
    } for fact in facts], ("fact_id", "subject", "predicate", "object", "source_record_id", "confidence"))
    _write_rows(output_dir / "exported_facts.csv", [{
        "fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate,
        "object": fact.object, "source_record_id": fact.record_id, "confidence": fact.confidence
    } for fact in official_facts], ("fact_id", "subject", "predicate", "object", "source_record_id", "confidence"))
    _write_rows(output_dir / "evidence.csv", evidence_rows,
                ("fact_id", "source_record_id", "field", "original_text", "evidence_type"))
    _write_rows(output_dir / "unresolved.csv", unresolved,
                ("source_record_id", "reference_text", "reason", "field"))
    _write_rows(output_dir / "aliases.csv", aliases,
                ("alias", "normalized_alias", "entity_id", "source_record_id"))
    positives = {(fact.subject, fact.object) for fact in splits["train"] + splits["valid"] + splits["test"]}
    candidates = sorted(row["entity_id"] for row in entities if row["entity_type"] == "Article")
    negative_rows = []
    for split_name in ("train", "valid", "test"):
        for fact in splits[split_name]:
            seed_value = int(hashlib.sha1(f"{seed}:{split_name}:{fact.subject}:{fact.object}".encode()).hexdigest()[:8], 16)
            candidate = candidates[seed_value % len(candidates)] if candidates else ""
            for _ in range(len(candidates)):
                if candidate != fact.subject and (fact.subject, candidate) not in positives:
                    negative_rows.append({
                        "subject": fact.subject, "predicate": "relatedTo", "object": candidate,
                        "source_fact_id": fact.fact_id, "split": split_name,
                        "sampling": "deterministic-one-corruption",
                    })
                    break
                candidate = candidates[(candidates.index(candidate) + 1) % len(candidates)]
    _write_rows(output_dir / "negative_samples.csv", negative_rows,
                ("subject", "predicate", "object", "source_fact_id", "split", "sampling"))

    express_dir = output_dir / "expressgnn"
    express_dir.mkdir(exist_ok=True)
    (express_dir / "entities.txt").write_text(
        "\n".join(sorted(entity_ids)) + "\n", encoding="utf-8"
    )
    relations = sorted(name for name in PREDICATES if name not in EXCLUDED_OFFICIAL_PREDICATES)
    _write_rows(output_dir / "relations.csv",
                [{"predicate": name, "domain": PREDICATES[name][0], "range": PREDICATES[name][1]}
                 for name in relations],
                ("predicate", "domain", "range"))
    (express_dir / "relations.txt").write_text("\n".join(relations) + "\n", encoding="utf-8")
    def fact_lines(items: Iterable[Fact]) -> str:
        return "".join(f"{fact.subject}\t{fact.predicate}\t{fact.object}\n" for fact in items)
    (express_dir / "facts.txt").write_text(fact_lines(splits["background"]), encoding="utf-8")
    (express_dir / "train.txt").write_text(fact_lines(splits["train"]), encoding="utf-8")
    (express_dir / "valid.txt").write_text(fact_lines(splits["valid"]), encoding="utf-8")
    (express_dir / "test.txt").write_text(fact_lines(splits["test"]), encoding="utf-8")
    _rules(express_dir / "rules.txt")
    (express_dir / "cleaned_rules_weight_larger_than_0.9.txt").write_text(
        (express_dir / "rules.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    stats = {
        "records": len(records),
        "entities": len(entities),
        "facts": len(facts),
        "official_export_facts": len(official_facts),
        "excluded_official_predicates": sorted(EXCLUDED_OFFICIAL_PREDICATES),
        "unresolved": len(unresolved),
        "entities_by_type": dict(Counter(row["entity_type"] for row in entities)),
        "facts_by_predicate": dict(Counter(fact.predicate for fact in facts)),
        "official_facts_by_predicate": dict(Counter(fact.predicate for fact in official_facts)),
        "splits": {name: len(items) for name, items in splits.items()},
        "split_ratios": {
            name: (len(splits[name]) / len(facts) if facts else 0.0)
            for name in ("background", "train", "valid", "test")
        },
        "target_split_ratios": {
            name: (len(splits[name]) / sum(len(splits[key]) for key in ("train", "valid", "test"))
                   if sum(len(splits[key]) for key in ("train", "valid", "test")) else 0.0)
            for name in ("train", "valid", "test")
        },
        "evaluation": {
            "target_relation": "relatedTo",
            "test_query_count": len(splits["test"]),
            "article_candidate_count": sum(row["entity_type"] == "Article" for row in entities),
            "candidate_directions_per_query": 2,
            "negative_samples_are_pipeline_only": True,
        },
        "seed": seed,
        "target_relation": "relatedTo",
        "directionality": "relatedTo is an observed content-relatedness relation; no inverse or symmetry facts are generated.",
        "predicate_meanings": {
            "belongsToTopic": "Article is classified under Topic.",
            "belongsToSubject": "Article belongs to Subject.",
            "belongsToChapter": "Article is contained in Chapter.",
            "relatedTo": "Article is explicitly marked as content-related to another Article; this is not assumed to be a citation or symmetric relation.",
            "hasSource": "Article is associated with a SourceArticle instrument/reference.",
            "sourceArticleOf": "SourceArticle is a source instrument/reference described by SourceDocument.",
            "amendedBy": "SourceArticle annotation indicates modification by another source.",
            "supplementedBy": "SourceArticle annotation indicates supplementation by another source.",
            "repealedBy": "SourceArticle annotation indicates repeal by another source.",
        },
        "citation_resolution": resolution_stats,
        "export_contract": {
            "facts_file": "background predicates only",
            "train_file": "relatedTo positives only",
            "valid_file": "relatedTo positives only",
            "test_file": "relatedTo positives only",
            "rule_heads": ["relatedTo", "relatedTo"],
        },
    }
    (output_dir / "kg_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_report = validate_bundle(output_dir)
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2), encoding="utf-8"
    )
    return stats


def validate_bundle(output_dir: Path) -> Dict[str, object]:
    express = output_dir / "expressgnn"
    entities = {line.strip() for line in (express / "entities.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    relations = {line.strip() for line in (express / "relations.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    errors: List[str] = []
    entity_types = {}
    with (output_dir / "entities.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entity_types[row["entity_id"]] = row["entity_type"]
    expected_domains = {name: pair for name, pair in PREDICATES.items()}
    csv_triples = set()
    with (output_dir / "exported_facts.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            triple = (row["subject"], row["predicate"], row["object"])
            csv_triples.add(triple)
            domain, range_ = expected_domains.get(row["predicate"], ("", ""))
            if entity_types.get(row["subject"]) != domain or entity_types.get(row["object"]) != range_:
                errors.append(f"facts.csv: invalid domain/range for {row['predicate']}")
    fact_ids = set()
    with (output_dir / "facts.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fact_id = row["fact_id"]
            if fact_id in fact_ids:
                errors.append(f"facts.csv: duplicate fact id {fact_id}")
            fact_ids.add(fact_id)
    evidence_ids = set()
    with (output_dir / "evidence.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            evidence_ids.add(row["fact_id"])
    if fact_ids - evidence_ids:
        errors.append("facts without traceable evidence detected")
    file_rows = {}
    for filename in ("facts.txt", "train.txt", "valid.txt", "test.txt"):
        file_rows[filename] = []
        with (express / filename).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    errors.append(f"{filename}:{number}: expected three tab-separated fields")
                elif parts[0] not in entities or parts[2] not in entities:
                    errors.append(f"{filename}:{number}: unknown entity")
                elif parts[1] not in relations:
                    errors.append(f"{filename}:{number}: unknown relation")
                else:
                    file_rows[filename].append(tuple(parts))
    background_predicates = {row[1] for row in file_rows["facts.txt"]}
    if "relatedTo" in background_predicates:
        errors.append("facts.txt contains target relation relatedTo")
    if any(row[1] != "relatedTo" for row in file_rows["train.txt"]):
        errors.append("train.txt contains a non-target predicate")
    facts_set = set(file_rows["facts.txt"])
    train_set = set(file_rows["train.txt"])
    if len(file_rows["facts.txt"]) != len(facts_set) or len(file_rows["train.txt"]) != len(train_set):
        errors.append("facts.txt or train.txt contains duplicate triples")
    if facts_set & train_set:
        errors.append("training triple duplicated through facts.txt and train.txt")
    exported_triples = facts_set | train_set | set(file_rows["valid.txt"]) | set(file_rows["test.txt"])
    if not exported_triples.issubset(csv_triples):
        errors.append("exported triples are missing from exported_facts.csv")
    split_sets = {}
    for filename in ("train.txt", "valid.txt", "test.txt"):
        split_sets[filename] = {
            tuple(line.rstrip("\n").split("\t"))
            for line in (express / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if split_sets["train.txt"] & split_sets["valid.txt"] or split_sets["train.txt"] & split_sets["test.txt"] \
            or split_sets["valid.txt"] & split_sets["test.txt"]:
        errors.append("positive split overlap detected")
    for left_name, right_name in (("train.txt", "valid.txt"), ("train.txt", "test.txt"), ("valid.txt", "test.txt")):
        inverse_left = {(row[2], row[1], row[0]) for row in split_sets[left_name] if len(row) == 3}
        if inverse_left & split_sets[right_name]:
            errors.append(f"inverse-edge split leakage detected between {left_name} and {right_name}")
    if errors:
        raise ValueError("Bundle validation failed: " + "; ".join(errors[:5]))
    return {"valid": True, "entity_count": len(entities), "relation_count": len(relations)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Vietnamese legal KG bundle for ExpressGNN.")
    parser.add_argument("--input", type=Path, help="CSV, JSON, or JSONL source; omitted uses Hugging Face dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    records = load_records(args.input, args.limit) if args.input else load_records(
        Path("hf://tmquan/phapdien-moj-gov-vn"), args.limit
    )
    stats = build_bundle(records, args.output, args.seed, args.validation_ratio, args.test_ratio)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
