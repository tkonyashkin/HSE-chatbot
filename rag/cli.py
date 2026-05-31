import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from pydantic import TypeAdapter

from parser.knowledge_schemas import KnowledgeDoc
from parser.schemas import DeadlineEntry
from rag.agent.llm_adapter import build_pydantic_ai_model
from rag.chunking.io import write_chunks
from rag.embedding.embedder import Embedder, EmbedderConfig
from rag.generation.generator import Generator, GeneratorConfig, PydanticAILLMClient
from rag.indexing.indexer import IndexerConfig, run_index
from rag.retrieval.pipeline import RetrievalPipeline

load_dotenv()

app = typer.Typer(help="RAG-стек для HSE-чатбота")
_KNOWLEDGE_DOC_ADAPTER: TypeAdapter[KnowledgeDoc] = TypeAdapter(KnowledgeDoc)


@app.command()
def chunk(
    programs_dir: Path = typer.Option(Path("data/programs/2026")),
    knowledge_dir: Path = typer.Option(Path("data/knowledge/2026")),
    output_path: Path = typer.Option(Path("data/chunks.jsonl")),
    config_path: Path = typer.Option(Path("config/chunking.yaml")),
    chunk_profile: str | None = typer.Option(None, "--chunk-profile"),
) -> None:
    result = write_chunks(
        programs_dir=programs_dir,
        knowledge_dir=knowledge_dir,
        output_path=output_path,
        config_path=config_path,
        chunk_profile=chunk_profile,
    )
    source_info = f"programs_dir={programs_dir}; knowledge_dir={knowledge_dir}"
    typer.echo(
        f"Записано чанков: {result.chunk_count} в {output_path} "
        f"(programs={result.program_chunk_count}, knowledge={result.knowledge_chunk_count}; "
        f"profile={result.profile_name}; {source_info})"
    )


@app.command()
def index(
    chunks_path: Path = typer.Option(Path("data/chunks.jsonl")),
    embedding_config_path: Path = typer.Option(Path("config/embedding.yaml")),
    retrieval_config_path: Path = typer.Option(Path("config/retrieval.yaml")),
) -> None:
    result = run_index(chunks_path, embedding_config_path, retrieval_config_path)
    typer.echo(f"Проиндексировано {result.chunk_count} чанков в '{result.collection_name}'")


@app.command()
def query(
    text: str = typer.Argument(..., help="Запрос на естественном языке"),
    embedding_config_path: Path = typer.Option(Path("config/embedding.yaml")),
    retrieval_config_path: Path = typer.Option(Path("config/retrieval.yaml")),
    prompt_path: Path = typer.Option(Path("rag/generation/prompts/baseline.yaml")),
    top_k: int = typer.Option(5),
) -> None:
    emb_config = EmbedderConfig.from_yaml(embedding_config_path)
    embedder = Embedder(emb_config)
    index_config = IndexerConfig.from_yaml(retrieval_config_path, dim=emb_config.dim)
    pipeline = RetrievalPipeline(embedder=embedder, index_config=index_config)
    hits = pipeline.search(text, top_k=top_k)
    typer.echo(f"Найдено чанков: {len(hits)}")
    for h in hits:
        typer.echo(
            f"  [{h.rank}] score={h.score:.3f} {h.chunk.program_slug}/{h.chunk.chunk_type.value}"
        )
    typer.echo("\n--- Генерация ответа ---")
    llm = PydanticAILLMClient(build_pydantic_ai_model())
    gen_config = GeneratorConfig.from_yaml(prompt_path)
    generator = Generator(llm=llm, config=gen_config)
    answer = generator.generate(text, hits)
    typer.echo(answer.text)
    typer.echo("\nИсточники:")
    for url in answer.source_urls:
        typer.echo(f"  - {url}")


def _load_deadlines(knowledge_dir: Path) -> list[DeadlineEntry]:
    out: list[DeadlineEntry] = []
    if not knowledge_dir.exists():
        return out
    for json_file in sorted(knowledge_dir.glob("*.json")):
        doc = _KNOWLEDGE_DOC_ADAPTER.validate_json(json_file.read_text(encoding="utf-8"))
        out.extend(getattr(doc, "deadlines", []))
    return out


@app.command()
def agent(
    text: str = typer.Argument(..., help="Запрос на естественном языке"),
    knowledge_dir: Path = typer.Option(Path("data/knowledge/2026")),
    embedding_config_path: Path = typer.Option(Path("config/embedding.yaml")),
    retrieval_config_path: Path = typer.Option(Path("config/retrieval.yaml")),
    prompts_dir: Path = typer.Option(Path("rag/agent/prompts")),
    acronyms_config_path: Path = typer.Option(Path("config/acronyms.yaml")),
) -> None:
    from rag.agent.context import AgentContext
    from rag.agent.runner import run_agent_planned_sync
    from rag.agent.tools import derive_acronyms_from_qdrant
    from rag.indexing.indexer import _client as _qdrant_client

    emb_config = EmbedderConfig.from_yaml(embedding_config_path)
    embedder = Embedder(emb_config)
    index_config = IndexerConfig.from_yaml(retrieval_config_path, dim=emb_config.dim)
    retrieval_pipeline = RetrievalPipeline(embedder=embedder, index_config=index_config)

    qdrant_client = _qdrant_client(index_config.qdrant_path)

    deadlines = _load_deadlines(knowledge_dir)
    context = AgentContext(
        retrieval=retrieval_pipeline,
        qdrant_client=qdrant_client,
        qdrant_collection=index_config.collection_name,
        acronyms=derive_acronyms_from_qdrant(
            qdrant_client, index_config.collection_name, acronyms_config_path
        ),
        deadlines=deadlines,
    )
    typer.echo(f"Контекст загружен: дедлайны: {len(deadlines)} записей")

    response = run_agent_planned_sync(query=text, context=context, worker_prompts_dir=prompts_dir)
    typer.echo("\n--- Ответ ---")
    typer.echo(response.text)
    if response.tool_calls:
        typer.echo(f"\nИнструменты: {', '.join(response.tool_calls)}")


@app.command("experiment")
def experiment(
    family: str = typer.Argument(
        ...,
        help="preprocessing | reranker | chunking | embedder | hybrid | generation_method",
    ),
    queries_path: Path = typer.Option(Path("experiments/golden_set/validation.yaml")),
    sources_path: Path = typer.Option(Path("experiments/golden_set/sources/validation.tsv")),
    acronyms_path: Path = typer.Option(Path("config/acronyms.yaml")),
    output_dir: Path = typer.Option(Path("experiments/results/validation")),
    embedding_config_path: Path = typer.Option(Path("config/embedding.yaml")),
    retrieval_config_path: Path = typer.Option(Path("config/retrieval.yaml")),
    pool_size: int = typer.Option(20),
    include_llm_variants: bool = typer.Option(False, "--llm/--no-llm"),
) -> None:
    from rag.agent.tools import load_program_acronyms_from_disk
    from rag.experiments.dataset import load_queries, load_source_labels
    from rag.experiments.preprocessors import expand_acronyms, lemma, lexical, raw
    from rag.experiments.results import write_json
    from rag.experiments.runner import run_variant

    queries = load_queries(queries_path)
    source_labels = load_source_labels(sources_path)
    acronyms = load_program_acronyms_from_disk(Path("data/programs/2026"), acronyms_path)

    emb_config = EmbedderConfig.from_yaml(embedding_config_path)
    embedder = Embedder(emb_config)
    index_config = IndexerConfig.from_yaml(retrieval_config_path, dim=emb_config.dim)
    base_pipeline = RetrievalPipeline(embedder=embedder, index_config=index_config)

    if family == "preprocessing":
        variant_specs: list[tuple[str, Callable[[str], str], Any]] = [
            ("raw", raw, base_pipeline),
            ("glossary", lambda q: expand_acronyms(q, acronyms), base_pipeline),
            ("lemma", lemma, base_pipeline),
            ("lexical", lambda q: lexical(q, acronyms), base_pipeline),
        ]
        baseline_name = "raw"
    elif family == "reranker":
        from rag.experiments.rerankers import (
            RerankingPipeline,
            build_cross_encoder_reranker,
            build_llm_reranker,
        )

        def glossary_fn(q: str) -> str:
            return expand_acronyms(q, acronyms)

        from rag.experiments.tracking import CallTracker

        variant_specs = [("no_reranker", glossary_fn, base_pipeline)]
        ce_pipeline = RerankingPipeline(
            base=base_pipeline, reranker=build_cross_encoder_reranker(), pool_size=pool_size
        )
        variant_specs.append(("cross_encoder_bge_m3", glossary_fn, ce_pipeline))
        reranker_trackers: dict[str, CallTracker] = {}
        if include_llm_variants:
            llm_models: list[tuple[str, str, dict[str, object] | None]] = [
                ("llm_haiku45", "anthropic/claude-haiku-4.5", None),
                ("llm_sonnet46", "anthropic/claude-sonnet-4.6", None),
                ("llm_gpt4omini", "openai/gpt-4o-mini", None),
                (
                    "llm_deepseek_v4_flash",
                    "deepseek/deepseek-v4-flash",
                    {"reasoning": {"enabled": False}},
                ),
                ("llm_deepseek_v4_flash_reasoning_on", "deepseek/deepseek-v4-flash", None),
            ]
            for vname, mid, extra_body in llm_models:
                llm_tracker = CallTracker(model_id=mid)
                reranker_trackers[vname] = llm_tracker
                llm_rerank_pipe = RerankingPipeline(
                    base=base_pipeline,
                    reranker=build_llm_reranker(
                        model_id=mid, tracker=llm_tracker, extra_body=extra_body
                    ),
                    pool_size=pool_size,
                )
                variant_specs.append((vname, glossary_fn, llm_rerank_pipe))
        baseline_name = "no_reranker"
    elif family == "chunking":
        from rag.chunking.io import write_chunks
        from rag.experiments.pipelines import DedupingPipeline
        from rag.indexing.indexer import run_index

        def glossary_fn(q: str) -> str:
            return expand_acronyms(q, acronyms)

        chunking_variants: list[tuple[str, str, bool]] = [
            ("structural", "structural_context_header", False),
            ("small", "small", False),
            ("parent_child", "small", True),
        ]
        chunks_path = Path("data/chunks.jsonl")
        chunking_cfg = Path("config/chunking.yaml")
        programs_dir = Path("data/programs/2026")
        knowledge_dir = Path("data/knowledge/2026")

        chunking_results = []
        last_profile: str | None = None
        for name, profile, dedupe in chunking_variants:
            if profile != last_profile:
                typer.echo(f"[{name}] chunking profile={profile}…")
                write_chunks(
                    programs_dir=programs_dir,
                    knowledge_dir=knowledge_dir,
                    output_path=chunks_path,
                    config_path=chunking_cfg,
                    chunk_profile=profile,
                )
                typer.echo(f"[{name}] reindex…")
                run_index(chunks_path, embedding_config_path, retrieval_config_path)
                last_profile = profile
            pipe: Any = base_pipeline
            if dedupe:
                pipe = DedupingPipeline(base=base_pipeline)
            typer.echo(f"[{name}] eval…")
            chunking_results.append(
                run_variant(
                    variant=name,
                    queries=queries,
                    source_labels=source_labels,
                    preprocess=glossary_fn,
                    pipeline=pipe,
                    pool_size=pool_size,
                )
            )

        typer.echo("restoring structural index…")
        write_chunks(
            programs_dir=programs_dir,
            knowledge_dir=knowledge_dir,
            output_path=chunks_path,
            config_path=chunking_cfg,
            chunk_profile="structural_context_header",
        )
        run_index(chunks_path, embedding_config_path, retrieval_config_path)

        results = chunking_results
        baseline_name = "structural"
        variant_specs = []
    elif family == "embedder":
        from rag.chunking.io import write_chunks
        from rag.indexing.indexer import build_index, load_chunks_jsonl

        def glossary_fn(q: str) -> str:
            return expand_acronyms(q, acronyms)

        embedder_configs: list[tuple[str, EmbedderConfig]] = [
            (
                "e5_large",
                EmbedderConfig(
                    model_name="intfloat/multilingual-e5-large",
                    dim=1024,
                    query_prefix="query: ",
                    passage_prefix="passage: ",
                    batch_size=16,
                    device="cpu",
                ),
            ),
            (
                "bge_m3",
                EmbedderConfig(
                    model_name="BAAI/bge-m3",
                    dim=1024,
                    query_prefix="",
                    passage_prefix="",
                    batch_size=16,
                    device="cpu",
                ),
            ),
            (
                "sbert_ru",
                EmbedderConfig(
                    model_name="ai-forever/sbert_large_nlu_ru",
                    dim=1024,
                    query_prefix="",
                    passage_prefix="",
                    batch_size=16,
                    device="cpu",
                ),
            ),
        ]
        chunks_path = Path("data/chunks.jsonl")
        write_chunks(
            programs_dir=Path("data/programs/2026"),
            knowledge_dir=Path("data/knowledge/2026"),
            output_path=chunks_path,
            config_path=Path("config/chunking.yaml"),
            chunk_profile="structural_context_header",
        )
        chunks = load_chunks_jsonl(chunks_path)

        embedder_results = []
        for name, ecfg in embedder_configs:
            typer.echo(f"[{name}] reindex with {ecfg.model_name}…")
            variant_embedder = Embedder(ecfg)
            embeddings = variant_embedder.encode_passages([c.embedding_input for c in chunks])
            variant_index_cfg = IndexerConfig.from_yaml(retrieval_config_path, dim=ecfg.dim)
            build_index(chunks, embeddings, variant_index_cfg)
            typer.echo(f"[{name}] eval…")
            variant_pipeline = RetrievalPipeline(
                embedder=variant_embedder, index_config=variant_index_cfg
            )
            embedder_results.append(
                run_variant(
                    variant=name,
                    queries=queries,
                    source_labels=source_labels,
                    preprocess=glossary_fn,
                    pipeline=variant_pipeline,
                    pool_size=pool_size,
                )
            )

        typer.echo("restoring e5_large index…")
        restore_ecfg = embedder_configs[0][1]
        restore_embedder = Embedder(restore_ecfg)
        restore_embeddings = restore_embedder.encode_passages([c.embedding_input for c in chunks])
        restore_index_cfg = IndexerConfig.from_yaml(retrieval_config_path, dim=restore_ecfg.dim)
        build_index(chunks, restore_embeddings, restore_index_cfg)

        results = embedder_results
        baseline_name = "e5_large"
        variant_specs = []
    elif family == "hybrid":
        from dataclasses import replace

        from rag.indexing.indexer import RetrievalMode, load_chunks_jsonl
        from rag.retrieval.bm25 import BM25Retriever

        def glossary_fn(q: str) -> str:
            return expand_acronyms(q, acronyms)

        chunks = load_chunks_jsonl(Path("data/chunks.jsonl"))
        bm25 = BM25Retriever.from_chunks(chunks)

        hybrid_variants: list[tuple[str, RetrievalMode]] = [
            ("dense_only", "dense_only"),
            ("hybrid_rrf", "hybrid_rrf"),
        ]
        hybrid_results = []
        for name, mode in hybrid_variants:
            typer.echo(f"[{name}] mode={mode}…")
            variant_cfg = replace(index_config, mode=mode)
            variant_pipeline = RetrievalPipeline(
                embedder=embedder, index_config=variant_cfg, bm25_retriever=bm25
            )
            hybrid_results.append(
                run_variant(
                    variant=name,
                    queries=queries,
                    source_labels=source_labels,
                    preprocess=glossary_fn,
                    pipeline=variant_pipeline,
                    pool_size=pool_size,
                )
            )
        results = hybrid_results
        baseline_name = "dense_only"
        variant_specs = []
    elif family == "generation_method":
        import json as _json

        from rag.agent.context import AgentContext
        from rag.agent.tools import derive_acronyms_from_qdrant
        from rag.experiments.deepeval_judge import build_judge
        from rag.experiments.eval_generation import GenerationRow, aggregate_generation
        from rag.experiments.generators import build_agentic_rag, build_vanilla_text
        from rag.experiments.rerankers import RerankingPipeline, build_llm_reranker
        from rag.indexing.indexer import _client as _qdrant_client

        def glossary_fn(q: str) -> str:
            return expand_acronyms(q, acronyms)

        reranker_model = os.environ.get("RERANKER_MODEL")
        reranker_extra_body_env = os.environ.get("RERANKER_EXTRA_BODY")
        reranker_extra_body = (
            _json.loads(reranker_extra_body_env) if reranker_extra_body_env else None
        )
        reranker_pipeline = RerankingPipeline(
            base=base_pipeline,
            reranker=build_llm_reranker(model_id=reranker_model, extra_body=reranker_extra_body),
            pool_size=pool_size,
        )

        qc = _qdrant_client(index_config.qdrant_path)

        deadlines = _load_deadlines(Path("data/knowledge/2026"))
        agent_ctx = AgentContext(
            retrieval=reranker_pipeline,
            qdrant_client=qc,
            qdrant_collection=index_config.collection_name,
            acronyms=derive_acronyms_from_qdrant(qc, index_config.collection_name, acronyms_path),
            deadlines=deadlines,
        )

        generators_to_run = [
            ("vanilla_text", build_vanilla_text(), reranker_pipeline),
            ("agentic_rag", build_agentic_rag(agent_context=agent_ctx), reranker_pipeline),
        ]
        gen_results = []
        for gname, generator, gpipe in generators_to_run:
            typer.echo(f"[{gname}] generating answers for {len(queries)} queries…")
            rows: list[GenerationRow] = []
            for q in queries:
                pre = glossary_fn(q.text)
                hits = gpipe.search(pre, top_k=5)
                answer = generator(q.text, hits)
                rows.append(
                    GenerationRow(
                        query_id=q.query_id,
                        query_type=q.query_type,
                        is_refusal_query=q.expected_refusal,
                        answer=answer,
                    )
                )
            gen_results.append(aggregate_generation(gname, rows))

        typer.echo("scoring with LLM-judge…")
        judge = build_judge()
        judge_scores: dict[str, dict[str, dict[str, float]]] = {}
        refusal_scores: dict[str, dict[str, float]] = {}
        for gname, _generator, gpipe in generators_to_run:
            judge_scores[gname] = {}
            refusal_scores[gname] = {}
            for q in queries:
                answer_lookup = next(
                    (
                        r.answer
                        for r in next(g for g in gen_results if g.variant == gname).rows
                        if r.query_id == q.query_id
                    ),
                    "",
                )
                if not answer_lookup:
                    continue
                if q.expected_refusal:
                    refusal_scores[gname][q.query_id] = judge.score_refusal(
                        variant=gname,
                        query_id=q.query_id,
                        query=q.text,
                        answer=answer_lookup,
                    )
                else:
                    pre = glossary_fn(q.text)
                    hits = gpipe.search(pre, top_k=5)
                    score = judge.score(
                        variant=gname,
                        query_id=q.query_id,
                        query=q.text,
                        expected_facts=list(q.expected_answer_facts),
                        retrieved=hits,
                        answer=answer_lookup,
                    )
                    judge_scores[gname][q.query_id] = {
                        "faithfulness": score.faithfulness,
                        "correctness": score.correctness,
                    }

        from rag.experiments.eval_generation import compute_metrics

        info_qids = [q.query_id for q in queries if not q.expected_refusal]
        refusal_qids = [q.query_id for q in queries if q.expected_refusal]

        info_refusal_per_variant: dict[str, dict[str, bool]] = {}
        for gname, _generator, _gpipe in generators_to_run:
            info_refusal_per_variant[gname] = {}
            for q in queries:
                if q.expected_refusal:
                    continue
                answer_lookup_q = next(
                    (
                        r.answer
                        for r in next(g for g in gen_results if g.variant == gname).rows
                        if r.query_id == q.query_id
                    ),
                    "",
                )
                info_refusal_per_variant[gname][q.query_id] = judge.is_refusal_on_info(
                    variant=gname, query_id=q.query_id, query=q.text, answer=answer_lookup_q
                )

        metrics_payload: dict[str, dict[str, float]] = {}
        for v in gen_results:
            vjudge = judge_scores.get(v.variant, {})
            metrics_payload[v.variant] = compute_metrics(
                faithfulness_scores={qid: s["faithfulness"] for qid, s in vjudge.items()},
                correctness_scores={qid: s["correctness"] for qid, s in vjudge.items()},
                refusal_scores=refusal_scores.get(v.variant, {}),
                info_qids=info_qids,
                refusal_qids=refusal_qids,
                info_refusal_lookup=info_refusal_per_variant.get(v.variant, {}),
            )

        payload_json: dict[str, Any] = {
            "family": family,
            "config": {
                "variants": [v.variant for v in gen_results],
                "queries_path": str(queries_path),
                "retrieval": "glossary + llm_haiku reranker",
                "top_k": 5,
                "pool_size": pool_size,
                "n_queries": len(queries),
            },
            "metrics": metrics_payload,
            "per_query": [
                {
                    "qid": row.query_id,
                    "query_type": row.query_type,
                    "variant": variant.variant,
                    "is_refusal_query": row.is_refusal_query,
                    "refusal_score": refusal_scores.get(variant.variant, {}).get(row.query_id),
                    "faithfulness": judge_scores.get(variant.variant, {})
                    .get(row.query_id, {})
                    .get("faithfulness"),
                    "correctness": judge_scores.get(variant.variant, {})
                    .get(row.query_id, {})
                    .get("correctness"),
                    "answer": row.answer,
                }
                for variant in gen_results
                for row in variant.rows
            ],
        }
        output_path = output_dir / f"{family}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _json.dumps(payload_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        typer.echo(f"Written: {output_path}")

        return
    else:
        available = "preprocessing, reranker, chunking, embedder, hybrid, generation_method"
        raise typer.BadParameter(f"Unknown family: {family}. Available: {available}")

    if family in ("preprocessing", "reranker"):
        results = [
            run_variant(
                variant=name,
                queries=queries,
                source_labels=source_labels,
                preprocess=fn,
                pipeline=pipe,
                pool_size=pool_size,
            )
            for name, fn, pipe in variant_specs
        ]
    output_path = output_dir / f"{family}.json"
    write_json(
        output_path,
        family=family,
        variants=results,
        config={
            "baseline": baseline_name,
            "queries_path": str(queries_path),
            "sources_path": str(sources_path),
            "retrieval_pool_size": pool_size,
            "reported_metrics": ["Hit@5", "MRR@5"]
            if family == "reranker"
            else ["Hit@5", "MRR@5", "Recall@20"],
            "include_llm_variants": include_llm_variants,
        },
    )

    if family == "reranker" and reranker_trackers:
        import json as _json2

        from rag.experiments.cost import cost_per_100q

        payload = _json2.loads(output_path.read_text(encoding="utf-8"))
        for vname, tracker in reranker_trackers.items():
            if vname in payload["metrics"]:
                payload["metrics"][vname]["latency_p50_ms"] = tracker.latency_percentile(0.5)
                payload["metrics"][vname]["cost_per_100q_usd"] = cost_per_100q(
                    tracker.total_cost(), len(queries)
                )
        output_path.write_text(
            _json2.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    typer.echo(f"Written: {output_path}")


if __name__ == "__main__":
    app()
