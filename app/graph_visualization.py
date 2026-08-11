"""Interactive Neo4j graph visualization (Streamlit + Pyvis).

Features:
  - Recommendation subgraph around a seed movie
  - Shortest-path highlight between two movies
  - Zoom / drag / physics (Pyvis)
  - Node search & focus
  - Export HTML (interactive) and PNG (static snapshot)

Run:
  uv run streamlit run app/graph_visualization.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "recommender"))

load_dotenv(ROOT / ".env")

from explanation_engine import (  # noqa: E402
    ExplanationEngine,
    ExplanationResult,
    _json_safe,
)
from graph_recommender import GraphRecommender, Recommendation  # noqa: E402
from neo4j_config import connect_neo4j  # noqa: E402

def dumps_graph_json(payload: Any) -> str:
    """Serialize explorer payloads, including Neo4j DateTime properties."""
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_json_safe)


# ---------------------------------------------------------------------------
# Visual theme
# ---------------------------------------------------------------------------

COLOR_BY_LABEL = {
    "Movie": "#2E86AB",
    "Director": "#E94F37",
    "Actor": "#F6AE2D",
    "Genre": "#6A994E",
    "Keyword": "#9B5DE5",
    "User": "#ADB5BD",
}

PATH_NODE_COLOR = "#FF006E"
PATH_EDGE_COLOR = "#FF006E"
REC_EDGE_COLOR = "#8338EC"
DEFAULT_EDGE_COLOR = "#ADB5BD"

PAGE_TITLE = "Cine Graph — Interactive Visualization"


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_driver():
    driver, database = connect_neo4j()
    return driver, database


def search_movie_titles(query: str, limit: int = 20) -> list[dict[str, Any]]:
    if not query or len(query.strip()) < 2:
        return []
    driver, database = get_driver()
    cypher = """
    MATCH (m:Movie)
    WHERE toLower(m.title) CONTAINS toLower($q)
    RETURN m.movieId AS movieId, m.title AS title, coalesce(m.tmdbEnriched, false) AS enriched
    ORDER BY size(m.title) ASC
    LIMIT $limit
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(cypher, q=query.strip(), limit=limit)]


def fetch_recommendation_subgraph(
    seed_movie_id: int,
    recommendations: list[Recommendation],
    path_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a visualization payload: seed + top recs + optional explanation path."""
    driver, database = get_driver()
    rec_ids = [r.movie_id for r in recommendations]
    cypher = """
    MATCH (seed:Movie {movieId: $seedId})
    OPTIONAL MATCH (seed)-[r1:DIRECTED_BY|ACTED_BY|HAS_GENRE|HAS_KEYWORD]->(bridge)
    WHERE bridge:Director OR bridge:Actor OR bridge:Genre OR bridge:Keyword
    WITH seed, collect(DISTINCT bridge)[0..40] AS bridges
    UNWIND bridges AS bridge
    OPTIONAL MATCH (bridge)<-[r2:DIRECTED_BY|ACTED_BY|HAS_GENRE|HAS_KEYWORD]-(rec:Movie)
    WHERE rec.movieId IN $recIds
    WITH seed,
         collect(DISTINCT {
           bridgeId: elementId(bridge),
           bridgeLabels: labels(bridge),
           bridgeName: coalesce(bridge.name, bridge.title),
           bridgeProps: properties(bridge),
           relType: type(r2),
           recId: rec.movieId,
           recTitle: rec.title,
           recElementId: elementId(rec)
         }) AS links
    RETURN seed.movieId AS seedId,
           seed.title AS seedTitle,
           elementId(seed) AS seedElementId,
           properties(seed) AS seedProps,
           links
    """
    with driver.session(database=database) as session:
        row = session.run(cypher, seedId=seed_movie_id, recIds=rec_ids).single()

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def upsert_node(
        element_id: str,
        label: str,
        display: str,
        *,
        highlight: bool = False,
        group: str = "default",
        props: dict | None = None,
    ) -> None:
        if element_id in nodes and not highlight:
            return
        nodes[element_id] = {
            "id": element_id,
            "label": label,
            "display": display,
            "highlight": highlight or nodes.get(element_id, {}).get("highlight", False),
            "group": group,
            "properties": _json_safe(props or {}),
        }

    if not row:
        return {"nodes": [], "edges": [], "seed_title": "Unknown"}

    seed_eid = row["seedElementId"]
    upsert_node(
        seed_eid,
        "Movie",
        row["seedTitle"],
        highlight=True,
        group="seed",
        props=dict(row["seedProps"] or {}),
    )

    # Index recommendations for score tooltips
    score_by_id = {r.movie_id: r for r in recommendations}

    for link in row["links"] or []:
        if not link or link.get("recId") is None:
            continue
        bridge_eid = link["bridgeId"]
        bridge_labels = link["bridgeLabels"] or ["Node"]
        bridge_label = bridge_labels[0]
        upsert_node(
            bridge_eid,
            bridge_label,
            str(link["bridgeName"]),
            props=dict(link.get("bridgeProps") or {}),
        )
        rec_eid = link["recElementId"]
        rec = score_by_id.get(int(link["recId"]))
        display = link["recTitle"]
        if rec:
            display = f"{rec.title}\n(score {rec.score:.1f})"
        upsert_node(
            rec_eid,
            "Movie",
            display,
            group="recommendation",
            props={"movieId": link["recId"], "title": link["recTitle"]},
        )
        # seed -> bridge
        edges.append(
            {
                "id": f"{seed_eid}-{bridge_eid}-{link['relType']}-out",
                "source": seed_eid,
                "target": bridge_eid,
                "type": link["relType"],
                "highlight": False,
                "kind": "recommendation",
            }
        )
        # bridge -> rec
        edges.append(
            {
                "id": f"{bridge_eid}-{rec_eid}-{link['relType']}-in",
                "source": bridge_eid,
                "target": rec_eid,
                "type": link["relType"],
                "highlight": False,
                "kind": "recommendation",
            }
        )

    # Merge shortest-path highlight if provided
    if path_payload and path_payload.get("found"):
        path_node_ids = set()
        for n in path_payload.get("nodes") or []:
            eid = n["element_id"]
            path_node_ids.add(eid)
            upsert_node(
                eid,
                n.get("key") or (n.get("labels") or ["Node"])[0],
                n.get("display") or "node",
                highlight=True,
                group="path",
                props=n.get("properties") or {},
            )
        for e in path_payload.get("edges") or []:
            edges.append(
                {
                    "id": e.get("element_id") or f"path-{e['source']}-{e['target']}",
                    "source": e["source"],
                    "target": e["target"],
                    "type": e["type"],
                    "highlight": True,
                    "kind": "path",
                }
            )

    # Deduplicate edges by endpoints+type
    uniq_edges = {}
    for e in edges:
        key = (e["source"], e["target"], e["type"], e.get("kind"))
        prev = uniq_edges.get(key)
        if prev is None or e.get("highlight"):
            uniq_edges[key] = e

    return {
        "seed_title": row["seedTitle"],
        "nodes": list(nodes.values()),
        "edges": list(uniq_edges.values()),
    }


# ---------------------------------------------------------------------------
# Pyvis / NetworkX builders
# ---------------------------------------------------------------------------


def build_pyvis_network(
    payload: dict[str, Any],
    *,
    height: str = "720px",
    filter_query: str = "",
    physics: bool = True,
) -> Network:
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0f1116",
        font_color="#F8F9FA",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.02,
        damping=0.4,
    )
    if not physics:
        net.toggle_physics(False)

    fq = filter_query.strip().lower()
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []

    visible_ids: set[str] = set()
    for node in nodes:
        display = str(node.get("display") or "")
        label = str(node.get("label") or "")
        if fq and fq not in display.lower() and fq not in label.lower():
            # Keep highlighted / seed / path nodes always visible
            if not node.get("highlight") and node.get("group") not in {"seed", "path"}:
                continue
        visible_ids.add(node["id"])

        color = PATH_NODE_COLOR if node.get("highlight") else COLOR_BY_LABEL.get(
            label, "#868E96"
        )
        size = 28 if node.get("group") == "seed" else 22 if node.get("highlight") else 16
        title = (
            f"<b>{html.escape(display)}</b><br/>"
            f"type: {html.escape(label)}<br/>"
            f"group: {html.escape(str(node.get('group')))}"
        )
        net.add_node(
            node["id"],
            label=display if len(display) < 42 else display[:39] + "...",
            title=title,
            color=color,
            size=size,
            borderWidth=3 if node.get("highlight") else 1,
            borderWidthSelected=4,
        )

    for edge in edges:
        if edge["source"] not in visible_ids or edge["target"] not in visible_ids:
            continue
        color = PATH_EDGE_COLOR if edge.get("highlight") else (
            REC_EDGE_COLOR if edge.get("kind") == "recommendation" else DEFAULT_EDGE_COLOR
        )
        width = 4 if edge.get("highlight") else 1.5
        net.add_edge(
            edge["source"],
            edge["target"],
            title=edge.get("type", ""),
            label=edge.get("type", "") if edge.get("highlight") else "",
            color=color,
            width=width,
        )

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true
          },
          "nodes": {
            "font": {"size": 14, "face": "Tahoma"},
            "scaling": {"min": 12, "max": 36}
          },
          "edges": {
            "smooth": {"type": "dynamic"}
          }
        }
        """
    )
    return net


def pyvis_to_html(net: Network) -> str:
    """Render Pyvis HTML in-memory with UTF-8 (avoids Windows cp1252 errors)."""
    return net.generate_html(notebook=False)


def build_networkx_graph(payload: dict[str, Any]) -> nx.Graph:
    g = nx.Graph()
    for node in payload.get("nodes") or []:
        g.add_node(
            node["id"],
            label=node.get("display"),
            kind=node.get("label"),
            highlight=bool(node.get("highlight")),
        )
    for edge in payload.get("edges") or []:
        g.add_edge(
            edge["source"],
            edge["target"],
            type=edge.get("type"),
            highlight=bool(edge.get("highlight")),
        )
    return g


def export_png_bytes(payload: dict[str, Any]) -> bytes:
    """Static PNG snapshot via NetworkX + Matplotlib (export image)."""
    import io

    import matplotlib.pyplot as plt

    g = build_networkx_graph(payload)
    if g.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f1116")
        ax.set_facecolor("#0f1116")
        ax.text(0.5, 0.5, "Empty graph", color="white", ha="center", va="center")
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0f1116")
        ax.set_facecolor("#0f1116")
        pos = nx.spring_layout(g, seed=42, k=0.6)
        node_colors = [
            PATH_NODE_COLOR
            if g.nodes[n].get("highlight")
            else COLOR_BY_LABEL.get(g.nodes[n].get("kind"), "#868E96")
            for n in g.nodes
        ]
        edge_colors = [
            PATH_EDGE_COLOR if g.edges[e].get("highlight") else DEFAULT_EDGE_COLOR
            for e in g.edges
        ]
        widths = [3.0 if g.edges[e].get("highlight") else 1.0 for e in g.edges]
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors, width=widths, alpha=0.85)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.95)
        labels = {
            n: (g.nodes[n].get("label") or "")[:28]
            for n in g.nodes
            if g.nodes[n].get("highlight") or g.nodes[n].get("kind") == "Movie"
        }
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7, font_color="white")
        ax.set_title("Cine Graph snapshot", color="white", fontsize=14, pad=12)
        ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1200px; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def movie_picker(label: str, key: str, default: str = "") -> dict[str, Any] | None:
    text = st.text_input(label, value=default, key=f"{key}_text")
    matches = search_movie_titles(text) if text else []
    if not matches:
        return None
    options = {f"{m['title']}  (id={m['movieId']})": m for m in matches}
    choice = st.selectbox(
        f"Select {label}",
        options=list(options.keys()),
        key=f"{key}_select",
    )
    return options[choice]


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_style()

    st.title("Cine Graph Visualization")
    st.caption(
        "Interactive Neo4j subgraphs — recommendations, shortest paths, search, zoom, export."
    )

    with st.sidebar:
        st.header("Controls")
        mode = st.radio(
            "Mode",
            ["Recommendation subgraph", "Shortest path", "Combined"],
            index=2,
        )
        top_k = st.slider("Top-N recommendations", 3, 15, 8)
        physics = st.checkbox("Enable physics", value=True)
        height = st.select_slider("Canvas height", options=[560, 640, 720, 860], value=720)
        node_filter = st.text_input(
            "Search / filter nodes",
            placeholder="e.g. Nolan, Matrix, Sci-Fi",
            help="Keeps matching nodes (+ highlighted path/seed). Zoom with mouse wheel.",
        )
        st.divider()
        st.markdown("**Legend**")
        st.markdown(
            "- 🔵 Movie  🔴 Director  🟡 Actor  \n"
            "- 🟢 Genre  🟣 Keyword  \n"
            "- 💗 Highlighted shortest path"
        )

    col_a, col_b = st.columns(2)
    with col_a:
        seed = movie_picker("Seed movie (A)", "seed", default="Interstellar")
    with col_b:
        other = None
        if mode in {"Shortest path", "Combined"}:
            other = movie_picker("Compare movie (B)", "other", default="Inception")

    run = st.button("Build graph", type="primary", width="stretch")

    if not run:
        st.info("Pick a seed movie and click **Build graph**.")
        return

    if seed is None:
        st.error("Select a valid seed movie.")
        return

    driver, database = get_driver()
    recommendations: list[Recommendation] = []
    explanation: ExplanationResult | None = None
    path_json: dict[str, Any] | None = None

    with st.spinner("Querying Neo4j..."):
        if mode in {"Recommendation subgraph", "Combined"}:
            recommender = GraphRecommender(driver, database)
            recommendations = recommender.recommend(int(seed["movieId"]), limit=top_k)

        if mode in {"Shortest path", "Combined"}:
            if other is None:
                st.error("Select movie B for shortest-path mode.")
                return
            engine = ExplanationEngine(driver, database)
            explanation = engine.explain(
                movie_id_a=int(seed["movieId"]),
                movie_id_b=int(other["movieId"]),
            )
            path_json = explanation.to_visualization_json()

        if mode == "Shortest path":
            # Path-only payload
            payload = {
                "seed_title": seed["title"],
                "nodes": [
                    {
                        "id": n["element_id"],
                        "label": n.get("key") or (n.get("labels") or ["Node"])[0],
                        "display": n.get("display"),
                        "highlight": True,
                        "group": "path",
                        "properties": n.get("properties") or {},
                    }
                    for n in (path_json or {}).get("nodes", [])
                ],
                "edges": [
                    {
                        "id": e.get("element_id"),
                        "source": e["source"],
                        "target": e["target"],
                        "type": e["type"],
                        "highlight": True,
                        "kind": "path",
                    }
                    for e in (path_json or {}).get("edges", [])
                ],
            }
        else:
            payload = fetch_recommendation_subgraph(
                int(seed["movieId"]),
                recommendations,
                path_payload=path_json if mode == "Combined" else None,
            )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes", len(payload.get("nodes") or []))
    m2.metric("Edges", len(payload.get("edges") or []))
    m3.metric("Recommendations", len(recommendations))
    m4.metric(
        "Path hops",
        explanation.hops if explanation and explanation.found else "—",
    )

    if explanation is not None:
        if explanation.found:
            st.success(explanation.human_explanation)
            st.code(explanation.path_text or "", language=None)
        else:
            st.warning(explanation.human_explanation)

    if recommendations:
        with st.expander("Recommendation table", expanded=False):
            st.dataframe(
                [
                    {
                        "rank": i + 1,
                        "title": r.title,
                        "score": round(r.score, 3),
                        "avg_rating": None if r.avg_rating is None else round(r.avg_rating, 3),
                        "n_ratings": r.n_ratings,
                        "top_explanation": (r.explanation_paths[0] if r.explanation_paths else ""),
                    }
                    for i, r in enumerate(recommendations)
                ],
                width="stretch",
            )

    # Interactive Pyvis canvas
    st.subheader("Interactive graph")
    st.caption("Scroll to zoom · drag to pan · hover for details · sidebar filter to search nodes")
    net = build_pyvis_network(
        payload,
        height=f"{height}px",
        filter_query=node_filter,
        physics=physics,
    )
    html_content = pyvis_to_html(net)
    components.html(html_content, height=height + 40, scrolling=True)

    # Exports
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download interactive HTML",
            data=html_content.encode("utf-8"),
            file_name="cine_graph.html",
            mime="text/html",
            width="stretch",
        )
    with c2:
        png_bytes = export_png_bytes(payload)
        st.download_button(
            "Download PNG image",
            data=png_bytes,
            file_name="cine_graph.png",
            mime="image/png",
            width="stretch",
        )
    with c3:
        st.download_button(
            "Download graph JSON",
            data=dumps_graph_json(payload).encode("utf-8"),
            file_name="cine_graph.json",
            mime="application/json",
            width="stretch",
        )


if __name__ == "__main__":
    main()
