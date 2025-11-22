"""
7_visualize_purity_tree.py
--------------------------
Visualize the purity tree built by 6_recursive_purity_tree.py as a Graphviz DOT file.

Each node is labeled with that node's split_term (i.e., the top G term at that level).
Edges are annotated as "L" (left/has) and "R" (right/not).

Usage:
  python scripts/7_visualize_purity_tree.py --variant clean --max_depth 5

Then render with:
  dot -Tpng data/visuals/purity_tree_clean.dot -o data/visuals/purity_tree_clean.png
"""

import os
import json
import argparse

DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
VISUALS_DIR = os.path.join(DATA_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)


def load_tree(variant: str):
    tree_path = os.path.join(RESULTS_DIR, f"purity_tree_{variant}.json")
    with open(tree_path) as f:
        tree = json.load(f)
    return tree, tree_path


def build_dot(root, max_depth: int, variant: str):
    """
    Traverse the tree and build a Graphviz DOT string.
    Each node label is the node's split_term (or 'leaf' if none).
    """

    lines = []
    lines.append("digraph PURITY_TREE {")
    lines.append('  rankdir=TB;')  # top-to-bottom; change to LR for left-to-right
    lines.append('  node [shape=box, fontsize=10];')

    # We'll assign numeric ids: n0, n1, ...
    node_counter = {"i": 0}

    def visit(node, parent_id=None, edge_label=None):
        if node is None:
            return

        depth = node.get("depth", 0)
        if depth > max_depth:
            return

        # assign ID
        my_id = f"n{node_counter['i']}"
        node_counter["i"] += 1

        split_term = node.get("split_term")
        label = split_term if split_term is not None else "leaf"
        n_docs = node.get("n_docs", 0)
        split_G = node.get("split_G", 0.0)

        # keep label simple; you can uncomment extras if you want them later
        # label_text = f"{label}\\nG={split_G:.3f}\\nn={n_docs}"
        label_text = label

        lines.append(f'  {my_id} [label="{label_text}"];')

        if parent_id is not None:
            # connect parent -> this node
            if edge_label is not None:
                lines.append(f'  {parent_id} -> {my_id} [label="{edge_label}"];')
            else:
                lines.append(f'  {parent_id} -> {my_id};')

        # stop if we've reached max_depth or this is a leaf (no split_term)
        if depth >= max_depth or node.get("split_term") is None:
            return

        # recurse on children
        left = node.get("left")
        right = node.get("right")

        if left is not None:
            visit(left, parent_id=my_id, edge_label="L")
        if right is not None:
            visit(right, parent_id=my_id, edge_label="R")

    visit(root)

    lines.append("}")
    dot_str = "\n".join(lines)
    out_path = os.path.join(VISUALS_DIR, f"purity_tree_{variant}.dot")
    with open(out_path, "w") as f:
        f.write(dot_str)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="clean",
        choices=["clean", "nostop"],
        help="Which purity tree to visualize (default: clean)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth to include in the visualization (default: 5)",
    )
    args = parser.parse_args()

    tree, tree_path = load_tree(args.variant)
    print(f"Loaded tree from {tree_path}")

    dot_path = build_dot(tree, max_depth=args.max_depth, variant=args.variant)
    print(f"Wrote DOT file to {dot_path}")
    print("Render with e.g.:")
    print(f"  dot -Tpng {dot_path} -o {dot_path.replace('.dot', '.png')}")


if __name__ == "__main__":
    main()
