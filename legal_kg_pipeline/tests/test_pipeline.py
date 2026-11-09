import unittest
from pathlib import Path

from legal_kg_pipeline.pipeline import (
    Record,
    _article_candidates,
    normalize_citation,
    extract_graph,
    _source_token,
    build_bundle,
    validate_bundle,
)


class PipelineTests(unittest.TestCase):
    def test_preserves_source_url_identity(self):
        first = _source_token("https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=37977#Chuong_I_Dieu_3")
        second = _source_token("https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=160806#Dieu_1")
        self.assertNotEqual(first, second)
        self.assertIn("ItemID=37977", first)
        self.assertIn("#Chuong_I_Dieu_3", first)

    def test_parses_compound_article_reference(self):
        self.assertEqual(list(_article_candidates("x Điều 1.12.LQ.11. y")), ["1.12.lq.11"])
        self.assertEqual(normalize_citation("Dieu_1_12_LQ_11"), "1.12.lq.11")

    def test_missing_ids_use_distinct_stable_fallbacks(self):
        records = [Record({"article_id": "1"}), Record({"article_id": "2"})]
        entities, _, unresolved, _, metrics = extract_graph(records)
        articles = {row["entity_id"] for row in entities if row["entity_type"] == "Article"}
        self.assertEqual(articles, {"article:row-0", "article:row-1"})
        self.assertEqual(metrics["missing_record_ids"], 2)
        self.assertFalse(unresolved)

    def test_duplicate_rows_are_skipped_and_empty_metadata_is_not_exported(self):
        records = [
            Record({"record_id": "same", "article_id": "1", "subject_id": "s", "subject_title": "S"}),
            Record({"record_id": "same", "article_id": "2", "subject_id": "s2", "subject_title": "S2"}),
        ]
        entities, facts, unresolved, _, metrics = extract_graph(records)
        self.assertEqual({row["entity_id"] for row in entities if row["entity_type"] == "Article"},
                         {"article:same"})
        self.assertFalse(any(row["entity_type"] == "Topic" for row in entities))
        self.assertEqual(metrics["duplicate_record_ids"], 1)
        self.assertTrue(any(row["field"] == "record_id" for row in unresolved))
        self.assertEqual(len(facts), 1)

    def test_rules_have_related_to_heads_and_splits_do_not_duplicate(self):
        records = [Record({"record_id": str(i), "article_id": str(i), "related_note_text": f"dieu {i-1}"})
                   for i in range(12)]
        output = Path(__file__).parent / "generated_rules"
        stats = build_bundle(records, output)
        rules = (output / "expressgnn" / "rules.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(line.endswith("relatedTo(A, B)") for line in rules))
        self.assertTrue(validate_bundle(output)["valid"])
        self.assertEqual(stats["evaluation"]["article_candidate_count"], 12)

    def test_structured_source_links_are_all_preserved(self):
        records = [Record({"record_id": "r1", "source_links": [
            {"text": "a", "href": "https://vbpl.vn/a?ItemID=1#Dieu_1"},
            {"text": "b", "href": "https://vbpl.vn/a?ItemID=2#Dieu_2"},
        ]})]
        entities, facts, _, _, _ = extract_graph(records)
        sources = {row["entity_id"] for row in entities if row["entity_type"] == "SourceArticle"}
        self.assertEqual(len(sources), 2)
        self.assertEqual(sum(fact.predicate == "hasSource" for fact in facts), 2)

    def test_builds_and_validates_bundle(self):
        records = [
            Record({
                "record_id": "r1", "subject_id": "s1", "topic_id": "t1",
                "topic_title": "T", "subject_title": "S", "article_id": "1",
                "chapter_title": "C", "source_note_text": "Luật A được sửa đổi bởi Điều 3 Thông tư B",
                "source_links": "https://vbpl.vn/a?ItemID=1#Dieu_1",
                "related_note_text": "Điều 1.12.LQ.11.",
            }),
            Record({
                "record_id": "r2", "subject_id": "s2", "topic_id": "t2",
                "topic_title": "T2", "subject_title": "S2", "article_id": "1.12.LQ.11",
                "chapter_title": "C", "source_links": "https://vbpl.vn/a?ItemID=2#Dieu_1",
            }),
        ]
        output = Path(__file__).parent / "generated"
        stats = build_bundle(records, output)
        self.assertEqual(stats["unresolved"], 0)
        self.assertTrue((output / "aliases.csv").is_file())
        self.assertTrue(validate_bundle(output)["valid"])


if __name__ == "__main__":
    unittest.main()
