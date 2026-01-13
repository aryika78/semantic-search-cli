import argparse
from rich.console import Console
from rich.table import Table

from semantic_search.search import semantic_search
from semantic_search.compare import TextComparator

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


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two texts")
    compare_parser.add_argument("text1", type=str)
    compare_parser.add_argument("text2", type=str)
    compare_parser.set_defaults(func=run_compare)

    # search command
    search_parser = subparsers.add_parser("search", help="Semantic search over corpus")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("corpus_file", type=str)
    search_parser.add_argument("--top_k", type=int, default=5)
    search_parser.add_argument("--metric", type=str, default="cosine")
    search_parser.set_defaults(func=run_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
