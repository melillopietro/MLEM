from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from common import OUTPUT_DIR, RANDOM_STATE

JACCARD_NETWORK_THRESHOLD = 0.35


def main() -> None:
    matrix_path = OUTPUT_DIR / "actor_profile_matrix.csv"
    metadata_path = OUTPUT_DIR / "actor_profile_metadata.csv"
    if not matrix_path.exists():
        raise FileNotFoundError("Run 02_build_actor_profiles.py first")

    matrix = pd.read_csv(matrix_path)
    metadata = pd.read_csv(metadata_path)
    actors = matrix["actor_label"].tolist()
    X = matrix.drop(columns=["actor_label"]).to_numpy(dtype=np.uint8)

    pair_rows = []
    similarity = np.eye(len(actors), dtype=float)
    for i, j in combinations(range(len(actors)), 2):
        a = X[i].astype(bool)
        b = X[j].astype(bool)
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        jaccard = intersection / union if union else 0.0
        denom = float(np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
        cosine = float(np.dot(X[i], X[j]) / denom) if denom else 0.0
        similarity[i, j] = similarity[j, i] = jaccard
        pair_rows.append(
            {
                "actor_a": actors[i],
                "actor_b": actors[j],
                "shared_techniques": intersection,
                "union_techniques": union,
                "jaccard_similarity": jaccard,
                "cosine_similarity": cosine,
                "identical_profile": bool(jaccard == 1.0),
            }
        )

    pairs = pd.DataFrame(pair_rows).sort_values(
        ["jaccard_similarity", "shared_techniques"], ascending=False
    )
    pairs.to_csv(OUTPUT_DIR / "actor_pairwise_similarity.csv", index=False)
    pairs.head(50).to_csv(OUTPUT_DIR / "actor_top_similarity_pairs.csv", index=False)
    pairs[pairs["identical_profile"]].to_csv(
        OUTPUT_DIR / "actor_identical_profiles.csv", index=False
    )

    distance = 1.0 - similarity
    best_k = None
    best_score = -1.0
    best_labels = None
    upper_k = min(12, len(actors) - 1)
    for k in range(2, upper_k + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(distance)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(distance, labels, metric="precomputed")
        if score > best_score:
            best_k = k
            best_score = float(score)
            best_labels = labels

    cluster_df = metadata.copy()
    if best_labels is not None:
        cluster_map = dict(zip(actors, best_labels + 1))
        cluster_df["cluster"] = cluster_df["actor_label"].map(cluster_map)
    else:
        cluster_df["cluster"] = 1
    cluster_df["cluster_method"] = "agglomerative_average_jaccard"
    cluster_df["selected_cluster_count"] = best_k if best_k is not None else 1
    cluster_df["silhouette_score"] = best_score if best_labels is not None else np.nan
    cluster_df.to_csv(OUTPUT_DIR / "actor_clusters.csv", index=False)

    G = nx.Graph()
    disclosure_lookup = metadata.set_index("actor_label")["disclosures"].to_dict()
    for actor in actors:
        G.add_node(actor, disclosures=int(disclosure_lookup.get(actor, 1)))
    for row in pair_rows:
        if row["jaccard_similarity"] >= JACCARD_NETWORK_THRESHOLD:
            G.add_edge(
                row["actor_a"],
                row["actor_b"],
                weight=row["jaccard_similarity"],
            )

    edge_df = nx.to_pandas_edgelist(G)
    edge_df.to_csv(OUTPUT_DIR / "actor_similarity_network_edges.csv", index=False)

    visible_nodes = [node for node in G.nodes if G.degree(node) > 0]
    H = G.subgraph(visible_nodes).copy()
    plt.figure(figsize=(14, 10))
    if len(H) > 0:
        pos = nx.spring_layout(H, seed=RANDOM_STATE, weight="weight", k=1.0)
        node_sizes = [250 + 80 * np.log1p(H.nodes[n]["disclosures"]) for n in H.nodes]
        edge_widths = [1.0 + 4.0 * H.edges[e]["weight"] for e in H.edges]
        nx.draw_networkx_nodes(H, pos, node_size=node_sizes, alpha=0.85)
        nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.45)
        nx.draw_networkx_labels(H, pos, font_size=8)
    plt.title(f"Actor-profile similarity network (Jaccard ≥ {JACCARD_NETWORK_THRESHOLD:.2f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actor_similarity_network.png", dpi=220)
    plt.close()

    top_metadata = metadata.sort_values("disclosures", ascending=False).head(30)
    top_actors = top_metadata["actor_label"].tolist()
    idx = [actors.index(a) for a in top_actors]
    heat = similarity[np.ix_(idx, idx)]
    plt.figure(figsize=(13, 11))
    plt.imshow(heat, vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(label="Jaccard similarity")
    plt.xticks(range(len(top_actors)), top_actors, rotation=90, fontsize=7)
    plt.yticks(range(len(top_actors)), top_actors, fontsize=7)
    plt.title("Jaccard similarity among the 30 most frequently disclosed actor labels")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actor_similarity_heatmap_top30.png", dpi=220)
    plt.close()

    print(f"Pairwise similarities written: {len(pairs)} actor pairs")
    print(f"Network edges at threshold {JACCARD_NETWORK_THRESHOLD}: {len(edge_df)}")
    print(f"Selected clusters: {best_k}; silhouette={best_score:.4f}")


if __name__ == "__main__":
    main()
