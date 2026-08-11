"""CineGraphRAG Streamlit app.

Pages: Home, Personalized Recommendations, Explain Recommendation,
Graph Explorer, Graph Analytics Dashboard.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "recommender"))

from services import (  # noqa: E402
    analytics_degree_hubs,
    analytics_rating_distribution,
    analytics_relationship_counts,
    analytics_top_genres,
    analytics_top_movies,
    get_driver,
    get_graph_rag,
    graph_kpis,
    list_users,
    search_movies,
    user_profile,
)

# Reuse visualization builders from the explorer module
import graph_visualization as viz  # noqa: E402
from explanation_engine import ExplanationEngine  # noqa: E402
from graph_recommender import GraphRecommender  # noqa: E402

# ---------------------------------------------------------------------------
# Theme / chrome
# ---------------------------------------------------------------------------

BRAND = "CineGraphRAG"
ACCENT = "#2E86AB"
ACCENT_2 = "#E94F37"
BG_CARD = "rgba(255,255,255,0.04)"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
  font-family: 'Source Sans 3', sans-serif;
}}
h1, h2, h3, .brand-title {{
  font-family: 'Fraunces', Georgia, serif !important;
  letter-spacing: -0.02em;
}}
.block-container {{
  padding-top: 1.25rem;
  max-width: 1180px;
}}
.hero {{
  padding: 1.4rem 1.6rem;
  border-radius: 18px;
  background:
    radial-gradient(1200px 400px at 10% -10%, rgba(46,134,171,0.35), transparent 55%),
    radial-gradient(900px 360px at 100% 0%, rgba(233,79,55,0.22), transparent 50%),
    linear-gradient(160deg, #12151c 0%, #1a1f29 55%, #12151c 100%);
  border: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 1.2rem;
}}
.hero h1 {{
  margin: 0 0 0.35rem 0;
  font-size: 2.1rem;
  color: #f8f9fa;
}}
.hero p {{
  margin: 0;
  color: rgba(248,249,250,0.78);
  max-width: 46rem;
  line-height: 1.5;
}}
.card {{
  background: {BG_CARD};
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
}}
.kpi {{
  font-size: 1.55rem;
  font-weight: 700;
  color: {ACCENT};
}}
.muted {{ color: rgba(248,249,250,0.65); font-size: 0.92rem; }}
.pill {{
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(46,134,171,0.18);
  color: #9ed0e6;
  font-size: 0.8rem;
  margin-right: 0.35rem;
}}
div[data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
</style>
"""


def inject_chrome() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def movie_select(label: str, key: str, default: str = "") -> dict[str, Any] | None:
    q = st.text_input(label, value=default, key=f"{key}_q")
    matches = search_movies(q) if q else []
    if not matches:
        return None
    labels = {
        f"{m['title']}  ·  id={m['movieId']}{'  ·  TMDB' if m.get('enriched') else ''}": m
        for m in matches
    }
    choice = st.selectbox(f"Select — {label}", list(labels.keys()), key=f"{key}_sel")
    return labels[choice]


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_home() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="pill">Neo4j · GraphRAG · Ollama</div>
          <h1 class="brand-title">{BRAND}</h1>
          <p>
            Movie intelligence on a knowledge graph: personalized recommendations,
            path-level explanations, interactive exploration, analytics, and
            natural-language GraphRAG — built on MovieLens, TMDB, and Neo4j.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        kpis = graph_kpis()
    except Exception as exc:
        st.error(f"Could not connect to Neo4j. Check `.env` and that the database is running. ({exc})")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{int(kpis.get('movies', 0)):,}")
    c2.metric("Users", f"{int(kpis.get('users', 0)):,}")
    c3.metric("Ratings", f"{int(kpis.get('ratings', 0)):,}")
    c4.metric("Avg rating", f"{float(kpis.get('avgRating') or 0):.2f}")

    st.markdown("#### Product map")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="card">
              <b>Personalized Recommendations</b><br/>
              <span class="muted">Graph paths via directors, actors, genres, keywords + co-fans.</span>
            </div>
            <div class="card">
              <b>Explain Recommendation</b><br/>
              <span class="muted">Shortest path between two titles with human + JSON explanation.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
              <b>Graph Explorer</b><br/>
              <span class="muted">Interactive Pyvis canvas with zoom, search, path highlight, export.</span>
            </div>
            <div class="card">
              <b>Analytics Dashboard</b><br/>
              <span class="muted">Rating skew, genre demand, hubs, relationship inventory.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Ask the graph (GraphRAG)")
    st.caption(
        "LangChain GraphCypherQAChain + Neo4j + local Ollama. "
        "Best on relational facts (who directed X, shared cast) — the model writes Cypher, Neo4j returns rows. "
        "The first question can take 1–2 minutes."
    )
    sample = st.selectbox(
        "Try a sample question",
        [
            "Custom question…",
            "Recommend movies like Interstellar",
            "Show connections between The Matrix and Cloud Atlas",
            "Find movies connected through actors or directors to Inception",
            "Which genres does Toy Story belong to?",
        ],
    )
    custom = ""
    if sample == "Custom question…":
        custom = st.text_input("Your question", placeholder="e.g. Who directed Inception?")
    question = custom.strip() if sample == "Custom question…" else sample

    if st.button("Ask GraphRAG", type="primary") and question:
        try:
            rag = get_graph_rag(verbose=False)
            with st.spinner("Generating Cypher → querying Neo4j → writing answer…"):
                payload = rag.ask(question)
            st.success(payload.get("answer") or "(empty answer)")
            with st.expander("Generated Cypher"):
                st.code(payload.get("cypher") or "(none)", language="cypher")
            with st.expander("Raw context"):
                st.write(payload.get("context"))
        except Exception as exc:
            st.error(
                "GraphRAG failed. Ensure Ollama is running "
                f"(`ollama serve` / model pulled). Details: {exc}"
            )


def page_recommendations() -> None:
    section_title(
        "Personalized Recommendations",
        "Start from a user taste profile or a seed movie. Ranked by graph signal strength.",
    )

    mode = st.radio(
        "Recommendation mode",
        ["From user profile", "From seed movie"],
        horizontal=True,
    )

    top_n = st.slider("Top-N", 5, 20, 10)
    seed_movie: dict[str, Any] | None = None

    if mode == "From user profile":
        users = list_users(limit=300)
        if not users:
            st.warning("No users found in the graph.")
            return
        user_id = st.selectbox("Active users (by rating volume)", users, index=0)
        profile = user_profile(int(user_id), limit=10)
        st.markdown("**Taste profile (highest ratings)**")
        st.dataframe(pd.DataFrame(profile), width="stretch", hide_index=True)
        if profile:
            options = {f"{p['title']} ({p['rating']})": p for p in profile}
            pick = st.selectbox("Recommend similar to", list(options.keys()))
            seed_movie = {
                "movieId": options[pick]["movieId"],
                "title": options[pick]["title"],
            }
    else:
        seed_movie = movie_select("Seed movie", "rec_seed", default="Interstellar")

    if seed_movie is None:
        st.info("Select a seed movie to continue.")
        return

    if st.button("Generate recommendations", type="primary"):
        driver, database = get_driver()
        recommender = GraphRecommender(driver, database)
        with st.spinner("Traversing graph signals…"):
            recs = recommender.recommend(int(seed_movie["movieId"]), limit=top_n)

        st.session_state["last_seed"] = seed_movie
        st.session_state["last_recs"] = [
            {
                "movie_id": r.movie_id,
                "title": r.title,
                "score": r.score,
                "avg_rating": r.avg_rating,
                "n_ratings": r.n_ratings,
                "explanations": r.explanation_paths,
            }
            for r in recs
        ]

    if "last_recs" in st.session_state and st.session_state.get("last_seed"):
        seed = st.session_state["last_seed"]
        st.markdown(f"**Seed:** {seed['title']} (id={seed['movieId']})")
        rows = []
        for i, r in enumerate(st.session_state["last_recs"], start=1):
            rows.append(
                {
                    "rank": i,
                    "title": r["title"],
                    "score": round(r["score"], 3),
                    "avg_rating": None
                    if r["avg_rating"] is None
                    else round(float(r["avg_rating"]), 3),
                    "n_ratings": r["n_ratings"],
                    "why": r["explanations"][0] if r["explanations"] else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with st.expander("Full explanation paths"):
            for r in st.session_state["last_recs"]:
                st.markdown(f"**{r['title']}** — score {r['score']:.3f}")
                for line in r["explanations"]:
                    st.markdown(f"- {line}")


def page_explain() -> None:
    section_title(
        "Explain Recommendation",
        "Shortest meaningful path between two movies — Cypher, prose, and visualization JSON.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        movie_a = movie_select("Movie A", "exp_a", default="Interstellar")
    with col_b:
        movie_b = movie_select("Movie B", "exp_b", default="Inception")

    max_depth = st.slider("Max path depth", 2, 6, 4)

    if st.button("Explain connection", type="primary"):
        if not movie_a or not movie_b:
            st.error("Select both movies.")
            return
        driver, database = get_driver()
        engine = ExplanationEngine(driver, database, max_depth=max_depth)
        with st.spinner("Finding best shortest path…"):
            result = engine.explain(
                movie_id_a=int(movie_a["movieId"]),
                movie_id_b=int(movie_b["movieId"]),
            )
        st.session_state["last_explanation"] = result.to_visualization_json()

    payload = st.session_state.get("last_explanation")
    if not payload:
        st.info("Pick two titles and click **Explain connection**.")
        return

    if payload.get("found"):
        st.success(payload.get("human_explanation"))
        st.code(payload.get("path_text") or "", language=None)
    else:
        st.warning(payload.get("human_explanation"))

    t1, t2, t3 = st.tabs(["Cypher", "Path graph", "JSON"])
    with t1:
        st.code(payload.get("cypher") or "", language="cypher")
    with t2:
        viz_payload = {
            "seed_title": payload.get("movie_a", {}).get("title"),
            "nodes": [
                {
                    "id": n["element_id"],
                    "label": n.get("key") or (n.get("labels") or ["Node"])[0],
                    "display": n.get("display"),
                    "highlight": True,
                    "group": "path",
                    "properties": n.get("properties") or {},
                }
                for n in payload.get("nodes") or []
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
                for e in payload.get("edges") or []
            ],
        }
        net = viz.build_pyvis_network(viz_payload, height="520px", physics=True)
        components.html(viz.pyvis_to_html(net), height=560, scrolling=False)
    with t3:
        st.download_button(
            "Download explanation JSON",
            data=viz.dumps_graph_json(payload).encode("utf-8"),
            file_name="explanation.json",
            mime="application/json",
            width="stretch",
        )
        st.json(payload)


def page_explorer() -> None:
    section_title(
        "Graph Explorer",
        "Recommendation subgraph + optional shortest-path highlight. Zoom, search, export.",
    )

    mode = st.radio(
        "Explorer mode",
        ["Recommendation subgraph", "Shortest path", "Combined"],
        horizontal=True,
        key="explorer_mode",
    )
    top_k = st.slider("Top-N recommendations", 3, 15, 8, key="explorer_topk")
    physics = st.checkbox("Physics", value=True, key="explorer_physics")
    node_filter = st.text_input("Search / filter nodes", key="explorer_filter")

    c1, c2 = st.columns(2)
    with c1:
        seed = movie_select("Seed movie (A)", "ex_seed", default="Interstellar")
    with c2:
        other = None
        if mode in {"Shortest path", "Combined"}:
            other = movie_select("Compare movie (B)", "ex_other", default="Inception")

    if not st.button("Build interactive graph", type="primary"):
        st.info("Configure options and click **Build interactive graph**.")
        return
    if seed is None:
        st.error("Select a seed movie.")
        return

    driver, database = get_driver()
    recommendations = []
    path_json = None
    explanation = None

    with st.spinner("Querying Neo4j…"):
        if mode in {"Recommendation subgraph", "Combined"}:
            recommendations = GraphRecommender(driver, database).recommend(
                int(seed["movieId"]), limit=top_k
            )
        if mode in {"Shortest path", "Combined"}:
            if other is None:
                st.error("Select movie B.")
                return
            explanation = ExplanationEngine(driver, database).explain(
                movie_id_a=int(seed["movieId"]),
                movie_id_b=int(other["movieId"]),
            )
            path_json = explanation.to_visualization_json()

        if mode == "Shortest path":
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
            payload = viz.fetch_recommendation_subgraph(
                int(seed["movieId"]),
                recommendations,
                path_payload=path_json if mode == "Combined" else None,
            )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes", len(payload.get("nodes") or []))
    m2.metric("Edges", len(payload.get("edges") or []))
    m3.metric("Recommendations", len(recommendations))
    m4.metric("Path hops", explanation.hops if explanation and explanation.found else "—")

    if explanation is not None:
        if explanation.found:
            st.success(explanation.human_explanation)
        else:
            st.warning(explanation.human_explanation)

    net = viz.build_pyvis_network(
        payload,
        height="720px",
        filter_query=node_filter,
        physics=physics,
    )
    html_content = viz.pyvis_to_html(net)
    components.html(html_content, height=760, scrolling=True)

    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "HTML",
        data=html_content.encode("utf-8"),
        file_name="explorer.html",
        mime="text/html",
        width="stretch",
    )
    e2.download_button(
        "PNG",
        data=viz.export_png_bytes(payload),
        file_name="explorer.png",
        mime="image/png",
        width="stretch",
    )
    e3.download_button(
        "JSON",
        data=viz.dumps_graph_json(payload).encode("utf-8"),
        file_name="explorer.json",
        mime="application/json",
        width="stretch",
    )


def page_analytics() -> None:
    section_title(
        "Graph Analytics Dashboard",
        "Descriptive analytics over the MovieLens + TMDB knowledge graph.",
    )

    with st.spinner("Computing analytics…"):
        kpis = graph_kpis()
        rating_dist = analytics_rating_distribution()
        genres = analytics_top_genres()
        top_movies = analytics_top_movies()
        hubs = analytics_degree_hubs()
        rels = analytics_relationship_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{int(kpis.get('movies', 0)):,}")
    c2.metric("Directors", f"{int(kpis.get('directors', 0)):,}")
    c3.metric("Actors", f"{int(kpis.get('actors', 0)):,}")
    c4.metric("Keywords", f"{int(kpis.get('keywords', 0)):,}")

    left, right = st.columns(2)
    with left:
        df = pd.DataFrame(rating_dist)
        if not df.empty:
            fig = px.bar(
                df,
                x="rating",
                y="n",
                title="Rating distribution",
                labels={"n": "Count", "rating": "Rating"},
                color_discrete_sequence=[ACCENT],
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        df_rel = pd.DataFrame(rels)
        if not df_rel.empty:
            fig = px.bar(
                df_rel,
                x="n",
                y="rel",
                orientation="h",
                title="Relationship inventory",
                labels={"n": "Count", "rel": "Type"},
                color_discrete_sequence=[ACCENT_2],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, width="stretch")

    with right:
        df_g = pd.DataFrame(genres)
        if not df_g.empty:
            fig = px.bar(
                df_g.sort_values("nRatings"),
                x="nRatings",
                y="genre",
                orientation="h",
                title="Genre demand (rating volume)",
                color="avgRating",
                color_continuous_scale="Tealgrn",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        df_h = pd.DataFrame(hubs)
        if not df_h.empty:
            fig = px.bar(
                df_h.sort_values("degree"),
                x="degree",
                y="name",
                color="type",
                orientation="h",
                title="Cast & crew hubs",
                color_discrete_sequence=[ACCENT, ACCENT_2],
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Highest-rated movies (min 30 ratings)")
    st.dataframe(pd.DataFrame(top_movies), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=BRAND,
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_chrome()

    with st.sidebar:
        st.markdown(f"## {BRAND}")
        st.caption("Explainable movie knowledge graph")
        st.divider()
        st.markdown(
            """
            **Stack**
            - Neo4j
            - Graph recommender
            - Explanation engine
            - GraphRAG + Ollama
            - Pyvis explorer
            """
        )
        st.divider()
        st.caption(
            "Neo4j and Ollama must be running for GraphRAG. "
            "Prefer qwen3:8b — smaller models often emit invalid Cypher."
        )

    pages = [
        st.Page(page_home, title="Home", icon="🏠", default=True),
        st.Page(page_recommendations, title="Personalized Recommendations", icon="🎯"),
        st.Page(page_explain, title="Explain Recommendation", icon="🧭"),
        st.Page(page_explorer, title="Graph Explorer", icon="🕸️"),
        st.Page(page_analytics, title="Graph Analytics Dashboard", icon="📊"),
    ]
    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()
