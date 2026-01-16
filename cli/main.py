import argparse
from rich.console import Console
from rich.table import Table
import numpy as np

from semantic_search.search import semantic_search
from semantic_search.compare import TextComparator
from semantic_search.index import VectorIndex

console = Console()


def run_compare(args):
    comparator = TextComparator()
    result = comparator.compare(args.text1, args.text2)

    table = Table(title="🔍 Text Similarity Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Cosine Similarity", f"{result['cosine_similarity']:.4f}")
    table.add_row("Euclidean Distance", f"{result['euclidean_distance']:.4f}")
    table.add_row("Dot Product", f"{result['dot_product']:.4f}")
    table.add_row("Interpretation", result["interpretation"])

    console.print(table)


def run_search(args):
    try:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        console.print(f"[red]File not found:[/red] {args.corpus_file}")
        return

    results = semantic_search(
        query=args.query,
        documents=documents,
        top_k=args.top_k,
        metric=args.metric,
    )

    table = Table(title="📚 Semantic Search Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Document")
    table.add_column("Score", style="green")

    for i, (doc, score) in enumerate(results, start=1):
        table.add_row(str(i), doc, f"{score:.4f}")

    console.print(table)


def run_index_build(args):
    with open(args.corpus_file, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    index = VectorIndex()
    index.build(documents)
    index.save(args.output)

    console.print(f"✅ Index built and saved to [green]{args.output}[/green]")


def run_index_search(args):
    index = VectorIndex()
    index.load(args.index)

    results = index.search(
        query=args.query,
        top_k=args.top_k,
        metric=args.metric,
    )

    table = Table(title="🔎 Indexed Search Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Document")
    table.add_column("Score", style="green")

    for i, (doc, score) in enumerate(results, start=1):
        table.add_row(str(i), doc, f"{score:.4f}")

    console.print(table)
def run_embed(args):
    from semantic_search.embeddings import EmbeddingGenerator
    import numpy as np

    embedder = EmbeddingGenerator()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            console.print(f"[red]File not found:[/red] {args.file}")
            return

        embeddings = embedder.embed_batch(texts)

        console.print("[green]Batch embeddings generated[/green]")
        console.print(f"• Number of texts: {len(texts)}")
        console.print(f"• Embedding shape: {embeddings.shape}")

    else:
        embedding = embedder.embed_single(args.text)

        console.print("[green]Embedding generated successfully[/green]")
        console.print(f"• Dimensions: {embedding.shape[0]}")
        console.print(f"• First 10 values: {embedding[:10]}")


def run_benchmark(args):
    from semantic_search.embeddings import EmbeddingGenerator
    import time

    models = args.models.split(",")
    text1, text2 = args.text1, args.text2

    table = Table(title="📊 Model Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Cosine", style="green")
    table.add_column("Euclidean", style="green")
    table.add_column("Dot", style="green")
    table.add_column("Time (s)", style="magenta")

    for model_name in models:
        start = time.time()
        embedder = EmbeddingGenerator(model_name=model_name)
        vec1 = embedder.embed_single(text1)
        vec2 = embedder.embed_single(text2)
        elapsed = time.time() - start

        cos = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        eucl = float(np.linalg.norm(vec1 - vec2))
        dot = float(np.dot(vec1, vec2))

        table.add_row(model_name, f"{cos:.4f}", f"{eucl:.4f}", f"{dot:.4f}", f"{elapsed:.2f}")

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare two texts")
    compare_parser.add_argument("text1")
    compare_parser.add_argument("text2")
    compare_parser.set_defaults(func=run_compare)

    # search (non-indexed)
    search_parser = subparsers.add_parser("search", help="Semantic search over corpus")
    search_parser.add_argument("query")
    search_parser.add_argument("corpus_file")
    search_parser.add_argument("--top_k", type=int, default=5)
    search_parser.add_argument("--metric", default="cosine")
    search_parser.set_defaults(func=run_search)

    # index
    index_parser = subparsers.add_parser("index", help="Vector index operations")
    index_sub = index_parser.add_subparsers(dest="index_command", required=True)

    # index build
    build_parser = index_sub.add_parser("build", help="Build vector index")
    build_parser.add_argument("corpus_file")
    build_parser.add_argument("--output", default="index.npz")
    build_parser.set_defaults(func=run_index_build)

    # index search
    index_search_parser = index_sub.add_parser("search", help="Search vector index")
    index_search_parser.add_argument("query")
    index_search_parser.add_argument("--index", default="index.npz")
    index_search_parser.add_argument("--top_k", type=int, default=5)
    index_search_parser.add_argument("--metric", default="cosine")
    index_search_parser.set_defaults(func=run_index_search)

    # embed command
    embed_parser = subparsers.add_parser("embed", help="Generate text embeddings")
    embed_parser.add_argument("text", nargs="?", help="Input text")
    embed_parser.add_argument("--file", type=str, help="File with one text per line")
    embed_parser.set_defaults(func=run_embed)

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare embedding models")
    benchmark_parser.add_argument("text1", type=str)
    benchmark_parser.add_argument("text2", type=str)
    benchmark_parser.add_argument(
    "--models", type=str, default="BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5",
    help="Comma-separated list of models to benchmark"
    )
    benchmark_parser.set_defaults(func=run_benchmark)


    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
