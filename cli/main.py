import argparse
from semantic_search.compare import TextComparator


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser("compare", help="Compare two texts")
    compare_parser.add_argument("text1", type=str)
    compare_parser.add_argument("text2", type=str)

    args = parser.parse_args()

    if args.command == "compare":
        comparator = TextComparator()
        result = comparator.compare(args.text1, args.text2)

        print("\n🔍 Similarity Results")
        print("----------------------")
        print(f"Cosine Similarity   : {result['cosine_similarity']:.4f}")
        print(f"Euclidean Distance : {result['euclidean_distance']:.4f}")
        print(f"Dot Product        : {result['dot_product']:.4f}")
        print(f"Interpretation     : {result['interpretation']}")
        print()


if __name__ == "__main__":
    main()
