# Vietnamese legal KG pipeline

`pipeline.py` loads the `tmquan/phapdien-moj-gov-vn` `documents/train` split (or a CSV/JSONL fixture), extracts typed entities and evidence-backed relations, records unresolved references, validates the graph, and exports an official ExpressGNN `load_method=1` bundle.

## Local reproduction

```powershell
python -m legal_kg_pipeline.pipeline --input records.jsonl --output artifacts/sample --limit 1000
# For the Hugging Face source, install requirements-local.txt and omit --input.
python -m legal_kg_pipeline.pipeline --output artifacts/full
```

The output contains `entities.csv`, `aliases.csv`, `facts.csv`, `evidence.csv`, `unresolved.csv`, `negative_samples.csv`, `kg_statistics.json`, and `expressgnn/`. Article nodes use globally unique `record_id` values; `aliases.csv` maps article IDs, anchors, and titles to those nodes. Source URL identity preserves query parameters such as `ItemID` and fragments, including every structured `source_links` entry. `relatedTo` is the link-prediction target for explicit content-relatedness annotations. It is directional: the pipeline does not create inverse or symmetry facts. The generated MLN rules use shared-topic/shared-subject evidence to softly predict `relatedTo`; they are heuristic constraints, not legally established implications.

The citation resolver parses compound references such as `Điều 1.12.LQ.11`, searches aliases across the full corpus, and records ambiguous matches in `unresolved.csv`. Amendment/supplement/repeal notes are split into separately evidenced source-target annotations where the text provides a target.

## Evaluation protocol

The official `load_method=1` evaluator iterates over every positive test triple. For each query it scores both tail replacements `(head, candidate)` and head replacements `(candidate, tail)`, then computes one rank for each direction. The patched evaluator restricts `relatedTo` candidates to IDs with the `article:` prefix, so both directions use Article entities only. The reported MRR is the mean of reciprocal ranks across both directions; Hits is Hits@10, the fraction of those directional queries ranked at 10 or better. The current official code uses observed ranking (it does not remove other known positives during ranking), so this is not filtered ranking. `negative_samples.csv` is generated for audits and is not read by the official trainer.

The training message `observed: N, hidden: M` counts grounded rule consequences: `N` candidate head facts already present in observed facts and `M` candidate head facts absent from them. It is not the number of train/test examples. The sample `kg_statistics.json` records test-query and Article-candidate counts.

The checkout includes a minimal portability patch in `ExpressGNN/common/utils.py`: experiment timestamps use hyphens instead of colons, so the wrapper works on Windows as well as Kaggle/Linux.

## Kaggle

Upload `artifacts/full` and this repository, then run:

```bash
python kaggle_train.py --bundle artifacts/full --expressgnn ExpressGNN --epochs 2 --device cuda
```

The wrapper writes `run_config.json`, `training.log`, and `evaluation_results.json`; it verifies `options.txt`, `saved_model/`, and the performance report after a successful run. Use `--num-batches 1 --batchsize 8 --epochs 1` for a bounded smoke run. Local extraction uses the optional `datasets` dependency; Kaggle training uses the dependencies in `requirements-kaggle.txt` and the official ExpressGNN loader/model. The adapter was verified against `ExpressGNN/data_process/preprocess.py`: `load_method=1` requires exactly tab-separated triples in `facts.txt`, `train.txt`, `valid.txt`, `test.txt`, one ID per line in `entities.txt` and `relations.txt`, and weighted disjunctions (`0.5 !p(...) v q(...)`) in the selected rule file.
