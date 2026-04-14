import streamlit as st
import json, time, os, math
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OAI_OK = True
except ImportError:
    OAI_OK = False

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="GraphRAG Lab", page_icon="🕸️", layout="wide")
st.markdown("""
<style>
body,.stApp{background:#0f172a;color:#e2e8f0}
.stSidebar{background:#1e293b}
.card{background:#1e293b;border-radius:10px;padding:16px;margin:8px 0;border-left:4px solid}
h1,h2,h3,h4{color:#e2e8f0!important}
div[data-testid="stMarkdownContainer"] p{color:#e2e8f0}
.hop-box{background:#1e293b;border-radius:8px;padding:10px 14px;margin:4px 0;
         border-left:3px solid #3b82f6;font-family:monospace;font-size:0.82rem}
.stat-box{background:#1e293b;border-radius:8px;padding:14px;text-align:center;
          border:1px solid #334155}
.stat-num{font-size:1.8rem;font-weight:bold;margin:4px 0}
</style>""", unsafe_allow_html=True)

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if "docs" not in st.session_state:
    st.session_state.docs = {
        "DOC-001": "[TICKET BUG-2847] Bug critico en payments-service v2.3.1. Falla al procesar tarjetas de debito >$10,000. NullPointerException en autorizacion. Estado: ABIERTO. Prioridad: CRITICA.",
        "DOC-002": "[ARQUITECTURA] payments-service es dependencia directa de billing-service. billing-service consume API REST en /v2/charge. Sin payments operativo, billing no puede facturar. Acoplamiento: ALTO.",
        "DOC-003": "[CONFIG REGIONAL] billing-service procesa transacciones en LATAM, EMEA y NA. Region Europa (EMEA-EU): Alemania, Francia, Espana, UK. Volumen Europa: 125,000 tx/mes. SLA: 99.9%.",
        "DOC-004": "[ENTERPRISE EUROPA] Goldman EU: 50,000 tx/mes. BNP Corporate: 30,000 tx/mes. Siemens Financial: 45,000 tx/mes. Todos con SLA 99.9% y penalidad 5% por incumplimiento.",
        "DOC-005": "[MANUAL] payments-service v2.3: Java 17, Spring Boot 3.1, PostgreSQL 15. CI/CD: GitHub Actions + ArgoCD. Owner: squad-payments@fenixoft.com",
    }
if "graph_nodes" not in st.session_state:
    st.session_state.graph_nodes = [
        {"id":"BUG-2847","tipo":"Bug","props":{"severidad":"CRITICA","estado":"ABIERTO"}},
        {"id":"payments-service","tipo":"Modulo","props":{"version":"v2.3.1","tech":"Java 17"}},
        {"id":"billing-service","tipo":"Servicio","props":{"endpoint":"/v2/charge","acoplamiento":"ALTO"}},
        {"id":"Region Europa","tipo":"Region","props":{"volumen":"125K tx/mes","sla":"99.9%"}},
        {"id":"Goldman EU","tipo":"Cliente","props":{"tx_mes":"50,000","mrr":"$45K","tier":"Enterprise"}},
        {"id":"BNP Corporate","tipo":"Cliente","props":{"tx_mes":"30,000","mrr":"$38K","tier":"Enterprise"}},
        {"id":"Siemens Financial","tipo":"Cliente","props":{"tx_mes":"45,000","mrr":"$12K","tier":"Business"}},
    ]
if "graph_edges" not in st.session_state:
    st.session_state.graph_edges = [
        {"src":"BUG-2847","dst":"payments-service","rel":"AFECTA_A"},
        {"src":"payments-service","dst":"billing-service","rel":"ES_DEPENDENCIA_DE"},
        {"src":"billing-service","dst":"Region Europa","rel":"PROCESA_TX_DE"},
        {"src":"Region Europa","dst":"Goldman EU","rel":"TIENE_CLIENTE"},
        {"src":"Region Europa","dst":"BNP Corporate","rel":"TIENE_CLIENTE"},
        {"src":"Region Europa","dst":"Siemens Financial","rel":"TIENE_CLIENTE"},
    ]
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🕸️ GraphRAG Lab")
    st.caption("Plataforma interactiva")
    st.markdown("---")
    api_key = st.text_input("🔑 OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY",""), placeholder="sk-...")
    if api_key:
        st.success("API key activa")
    else:
        st.info("Sin key = modo demo")
    st.markdown("---")
    st.markdown(f"**📄 Documentos:** {len(st.session_state.docs)}")
    st.markdown(f"**📍 Nodos:** {len(st.session_state.graph_nodes)}")
    st.markdown(f"**🔗 Relaciones:** {len(st.session_state.graph_edges)}")
    st.markdown(f"**📊 Comparaciones:** {len(st.session_state.comparison_history)}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
COLORES_TIPO = {"Bug":"#ef4444","Modulo":"#3b82f6","Servicio":"#a855f7",
                "Region":"#eab308","Cliente":"#22c55e","Decision":"#f97316",
                "RFC":"#f97316","Incidente":"#dc2626","Otro":"#6b7280"}
SIMBOLOS = {"Bug":"diamond","Modulo":"square","Servicio":"square",
            "Region":"circle","Cliente":"circle","Decision":"hexagon",
            "Incidente":"star","Otro":"circle"}
ICONOS = {"Bug":"🔴","Modulo":"🔵","Servicio":"🟣","Region":"🟡","Cliente":"🟢",
          "Decision":"🟠","Incidente":"⭐","Otro":"⚪"}

def build_nx_graph():
    G = nx.DiGraph()
    for n in st.session_state.graph_nodes:
        G.add_node(n["id"], tipo=n["tipo"], **n["props"])
    for e in st.session_state.graph_edges:
        G.add_edge(e["src"], e["dst"], relacion=e["rel"])
    return G

def plotly_graph(G, title="", highlight_nodes=None, highlight_edges=None, height=480):
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",height=200,
                          annotations=[dict(text="Grafo vacio — agrega nodos",
                          x=0.5,y=0.5,showarrow=False,font=dict(color="#64748b",size=16))])
        return fig
    pos = nx.spring_layout(G, seed=42, k=2.8, iterations=60)
    hl_n = highlight_nodes or set()
    hl_e = highlight_edges or set()

    edge_traces, annotations = [], []
    for u, v, data in G.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        rel = data.get("relacion","")
        is_hl = (u,v) in hl_e or (u in hl_n and v in hl_n)
        color = "#f97316" if is_hl else "#475569"
        w = 3 if is_hl else 1.5
        edge_traces.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
            line=dict(width=w,color=color),hoverinfo="none",showlegend=False))
        annotations.append(dict(ax=x0,ay=y0,x=x1,y=y1,xref="x",yref="y",axref="x",ayref="y",
            showarrow=True,arrowhead=2,arrowsize=1.5,arrowwidth=w,arrowcolor=color,opacity=0.6))
        if rel:
            mx,my=(x0+x1)/2,(y0+y1)/2
            annotations.append(dict(x=mx,y=my,text=f"<i>{rel}</i>",showarrow=False,
                font=dict(size=8,color="#94a3b8"),bgcolor="#1e293b",opacity=0.9))

    tipos = {}
    for node, data in G.nodes(data=True):
        t = data.get("tipo","Otro")
        tipos.setdefault(t,[]).append(node)

    node_traces = []
    for tipo, nodes in tipos.items():
        xs,ys,texts,hovers=[],[],[],[]
        for n in nodes:
            x,y=pos[n]; xs.append(x); ys.append(y); texts.append(n)
            props="<br>".join(f"<b>{k}:</b> {v}" for k,v in G.nodes[n].items() if k!="tipo")
            hovers.append(f"<b>{n}</b> [{tipo}]<br>{props}")
        color = COLORES_TIPO.get(tipo,"#6b7280")
        sym = SIMBOLOS.get(tipo,"circle")
        sz = [22 if n in hl_n else 16 for n in nodes]
        node_traces.append(go.Scatter(x=xs,y=ys,mode="markers+text",name=tipo,
            marker=dict(size=sz,color=color,symbol=sym,line=dict(width=2,color="#0f172a")),
            text=texts,textposition="top center",textfont=dict(size=9,color="#e2e8f0"),
            hovertext=hovers,hoverinfo="text"))

    fig = go.Figure(data=edge_traces+node_traces)
    fig.update_layout(title=dict(text=title,font=dict(color="#e2e8f0",size=13)),
        paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",height=height,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        margin=dict(l=10,r=10,t=40,b=10),font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0"),bgcolor="#1e293b",bordercolor="#334155"),
        annotations=annotations,
        hoverlabel=dict(bgcolor="#1e293b",font_size=11,font_color="#e2e8f0"))
    return fig

def rag_search(pregunta, docs, top_k=2):
    textos = list(docs.values()); ids = list(docs.keys())
    vec = TfidfVectorizer()
    mat = vec.fit_transform(textos + [pregunta])
    sims = cosine_similarity(mat[-1], mat[:-1])[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = [{"id":ids[i],"sim":round(float(sims[i]),4),"text":textos[i]} for i in top_idx]
    all_s = {ids[i]:round(float(sims[i]),4) for i in range(len(ids))}
    return results, all_s

def graphrag_traverse(G, seed, max_hops=4):
    visited = {seed}; frontier = {seed}; log = []
    log.append({"hop":0,"from":None,"rel":None,"to":seed,"tipo":G.nodes[seed].get("tipo","?")})
    for hop in range(1, max_hops+1):
        nxt = set()
        for n in frontier:
            for v in G.successors(n):
                d = G.get_edge_data(n,v)
                log.append({"hop":hop,"from":n,"rel":d.get("relacion",""),"to":v,
                            "tipo":G.nodes[v].get("tipo","?")})
                nxt.add(v); visited.add(v)
        frontier = nxt
        if not frontier: break
    return G.subgraph(visited).copy(), log

def subgraph_to_context(sg):
    lines = ["SUBGRAFO GRAPHRAG:\n","ENTIDADES:"]
    for n,d in sg.nodes(data=True):
        props=", ".join(f"{k}={v}" for k,v in d.items())
        lines.append(f"  {n} ({props})")
    lines.append("\nRELACIONES:")
    for u,v,d in sg.edges(data=True):
        lines.append(f"  ({u}) --[{d.get('relacion','')}]--> ({v})")
    return "\n".join(lines)

def get_client(key):
    if OAI_OK and key:
        clean = key.strip().encode("ascii",errors="ignore").decode("ascii")
        if clean: return OpenAI(api_key=clean)
    return None

def ask_llm(pregunta, contexto, client=None):
    if client:
        try:
            r = client.chat.completions.create(model="gpt-4o-mini",
                messages=[{"role":"system","content":"Eres asistente tecnico de Fenixoft. Responde SOLO con el contexto dado. Si falta info, dilo."},
                          {"role":"user","content":f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"}],
                temperature=0)
            return r.choices[0].message.content
        except: pass
    has_chain = all(w in contexto.lower() for w in ["dependencia","billing","region","europa"])
    if has_chain:
        return ("RESPUESTA (contexto completo):\n\n"
            "Cadena de impacto confirmada: BUG-2847 -> payments-service -> billing-service -> Region Europa\n\n"
            "Clientes afectados:\n"
            "  Goldman EU: 50,000 tx/mes EN RIESGO\n"
            "  BNP Corporate: 30,000 tx/mes EN RIESGO\n"
            "  Siemens Financial: 45,000 tx/mes EN RIESGO\n"
            "  TOTAL: 125,000 tx/mes\n\n"
            "SLA 99.9% en riesgo. Penalidad: 5% del contrato mensual.\n"
            "Accion: Escalar a squad-payments@fenixoft.com inmediatamente.")
    return ("RESPUESTA (contexto incompleto):\n\n"
        "Confirmo Bug #BUG-2847 en payments-service (CRITICA) y clientes enterprise en Europa.\n\n"
        "SIN EMBARGO, NO puedo establecer si el bug impacta directamente a los clientes europeos. "
        "Los documentos disponibles no especifican la relacion tecnica entre payments-service "
        "y los servicios regionales.\n\n"
        "Recomendacion: consultar al equipo de arquitectura.")


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Base de Conocimiento",
    "🕸️ Graph Builder",
    "⚡ RAG vs GraphRAG",
    "📈 Niveles de Grafo",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — KNOWLEDGE BASE EDITOR
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Base de Conocimiento")

    col_list, col_edit = st.columns([1, 1])

    with col_list:
        st.markdown("**Documentos actuales**")
        for doc_id, texto in st.session_state.docs.items():
            with st.expander(f"📄 {doc_id}", expanded=False):
                st.text_area(f"edit_{doc_id}", texto, key=f"ta_{doc_id}", height=100,
                             label_visibility="collapsed")
                c1, c2 = st.columns(2)
                if c1.button("💾 Guardar", key=f"save_{doc_id}", use_container_width=True):
                    st.session_state.docs[doc_id] = st.session_state[f"ta_{doc_id}"]
                    st.success("Guardado")
                    st.rerun()
                if c2.button("🗑️ Eliminar", key=f"del_{doc_id}", use_container_width=True):
                    del st.session_state.docs[doc_id]
                    st.rerun()

    with col_edit:
        st.markdown("**Agregar documento**")
        new_id = st.text_input("ID del documento", value=f"DOC-{len(st.session_state.docs)+1:03d}",
                               key="new_doc_id")
        new_text = st.text_area("Contenido", height=120, key="new_doc_text",
                                placeholder="Pega aqui el contenido del documento...")
        if st.button("➕ Agregar documento", type="primary", use_container_width=True):
            if new_id and new_text:
                st.session_state.docs[new_id] = new_text
                st.success(f"{new_id} agregado")
                st.rerun()
            else:
                st.warning("Completa ID y contenido")

        st.markdown("---")
        st.markdown("**Vista rapida: similitudes entre documentos**")
        if len(st.session_state.docs) >= 2:
            textos = list(st.session_state.docs.values())
            ids = list(st.session_state.docs.keys())
            vec = TfidfVectorizer()
            mat = vec.fit_transform(textos)
            sim_matrix = cosine_similarity(mat)
            fig_heat = go.Figure(go.Heatmap(
                z=sim_matrix, x=ids, y=ids,
                colorscale="Blues", zmin=0, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in sim_matrix],
                texttemplate="%{text}", hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"
            ))
            fig_heat.update_layout(paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),height=300,margin=dict(l=80,r=20,t=30,b=60),
                title="Similitud coseno entre documentos")
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Los documentos muy similares se agrupan en RAG vectorial. Los disimilares se pierden.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRAPH BUILDER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Graph Builder")

    col_tools, col_canvas = st.columns([1, 2])

    with col_tools:
        # ADD NODE + RELATIONS
        with st.expander("➕ Agregar nodo + relaciones", expanded=True):
            node_id = st.text_input("Nombre del nodo", key="add_node_id",
                                    placeholder="ej: auth-service")
            node_tipo = st.selectbox("Tipo", list(COLORES_TIPO.keys()), key="add_node_tipo")
            node_props_str = st.text_input("Propiedades (key:val, key:val)",
                                           key="add_node_props", placeholder="version:v1.0, team:backend")

            # Dynamic relation slots
            existing_ids = [n["id"] for n in st.session_state.graph_nodes]
            if "n_rel_slots" not in st.session_state:
                st.session_state.n_rel_slots = 1

            if existing_ids:
                st.markdown('<span style="color:#94a3b8;font-size:.82rem">Relaciones al crear (opcional):</span>', unsafe_allow_html=True)
                new_rels = []
                for i in range(st.session_state.n_rel_slots):
                    rc1, rc2, rc3 = st.columns([2, 3, 2])
                    direction = rc1.selectbox("Dir", ["→ sale de","← entra a"], key=f"rel_dir_{i}",
                                              label_visibility="collapsed")
                    rel_type = rc2.text_input("Relacion", key=f"rel_type_{i}",
                                              placeholder="DEPENDE_DE", label_visibility="collapsed")
                    target = rc3.selectbox("Nodo", existing_ids, key=f"rel_target_{i}",
                                           label_visibility="collapsed")
                    if rel_type:
                        new_rels.append({"dir": direction, "rel": rel_type, "target": target})

                bc1, bc2 = st.columns(2)
                if bc1.button("+ Otra relacion", use_container_width=True, key="btn_more_rels"):
                    st.session_state.n_rel_slots = min(st.session_state.n_rel_slots + 1, 6)
                    st.rerun()
                if bc2.button("- Quitar ultima", use_container_width=True, key="btn_less_rels",
                              disabled=st.session_state.n_rel_slots <= 1):
                    st.session_state.n_rel_slots = max(st.session_state.n_rel_slots - 1, 1)
                    st.rerun()
            else:
                new_rels = []

            if st.button("Crear nodo", type="primary", use_container_width=True, key="btn_add_node"):
                if node_id:
                    props = {}
                    if node_props_str:
                        for p in node_props_str.split(","):
                            if ":" in p:
                                k,v = p.split(":",1)
                                props[k.strip()] = v.strip()
                    if node_id not in existing_ids:
                        st.session_state.graph_nodes.append({"id":node_id,"tipo":node_tipo,"props":props})
                        for r in new_rels:
                            if r["dir"].startswith("→"):
                                st.session_state.graph_edges.append({"src":node_id,"dst":r["target"],"rel":r["rel"]})
                            else:
                                st.session_state.graph_edges.append({"src":r["target"],"dst":node_id,"rel":r["rel"]})
                        st.session_state.n_rel_slots = 1
                        st.rerun()
                    else:
                        st.warning("Ya existe un nodo con ese nombre")

        # ADD EDGE (standalone)
        with st.expander("🔗 Agregar relacion entre nodos existentes"):
            node_ids = [n["id"] for n in st.session_state.graph_nodes]
            if len(node_ids) >= 2:
                e_src = st.selectbox("Origen", node_ids, key="e_src")
                e_dst = st.selectbox("Destino", node_ids, key="e_dst")
                e_rel = st.text_input("Tipo de relacion", key="e_rel", placeholder="ej: DEPENDE_DE")
                if st.button("Agregar relacion", type="primary", use_container_width=True, key="btn_add_edge"):
                    if e_src != e_dst and e_rel:
                        st.session_state.graph_edges.append({"src":e_src,"dst":e_dst,"rel":e_rel})
                        st.rerun()
            else:
                st.info("Agrega al menos 2 nodos primero")

        # DELETE
        with st.expander("🗑️ Eliminar"):
            del_node = st.selectbox("Eliminar nodo", ["(ninguno)"] + node_ids, key="del_node")
            if st.button("Eliminar nodo", use_container_width=True, key="btn_del_node"):
                if del_node != "(ninguno)":
                    st.session_state.graph_nodes = [n for n in st.session_state.graph_nodes if n["id"]!=del_node]
                    st.session_state.graph_edges = [e for e in st.session_state.graph_edges
                                                     if e["src"]!=del_node and e["dst"]!=del_node]
                    st.rerun()
            if st.session_state.graph_edges:
                edge_labels = [f"{e['src']} --[{e['rel']}]--> {e['dst']}" for e in st.session_state.graph_edges]
                del_edge_idx = st.selectbox("Eliminar relacion", range(len(edge_labels)),
                                            format_func=lambda i: edge_labels[i], key="del_edge")
                if st.button("Eliminar relacion", use_container_width=True, key="btn_del_edge"):
                    st.session_state.graph_edges.pop(del_edge_idx)
                    st.rerun()

        # STATS
        G = build_nx_graph()
        st.markdown("---")
        st.markdown(f"**📍 {G.number_of_nodes()} nodos** | **🔗 {G.number_of_edges()} relaciones**")
        if G.number_of_nodes() > 0 and nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G):
            st.success("Grafo conectado")
        elif G.number_of_nodes() > 0:
            comps = nx.number_weakly_connected_components(G)
            st.warning(f"{comps} componentes desconectados")

    with col_canvas:
        G = build_nx_graph()
        fig = plotly_graph(G, title="Tu Knowledge Graph (editable)")
        st.plotly_chart(fig, use_container_width=True)

        # Node table
        if st.session_state.graph_nodes:
            st.markdown("**Nodos registrados:**")
            df_nodes = pd.DataFrame([
                {"Nodo":n["id"], "Tipo":f"{ICONOS.get(n['tipo'],'⚪')} {n['tipo']}",
                 "Propiedades": ", ".join(f"{k}={v}" for k,v in n["props"].items()) or "—"}
                for n in st.session_state.graph_nodes
            ])
            st.dataframe(df_nodes, use_container_width=True, hide_index=True, height=200)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RAG vs GRAPHRAG LIVE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("RAG vs GraphRAG — Comparacion en vivo")

    # Controls
    pregunta = st.text_input("🔍 Tu pregunta:", value="Cual es el impacto del bug en pagos sobre los clientes de Europa?",
                              key="user_question")
    col_p1, col_p2, col_p3 = st.columns(3)
    top_k = col_p1.slider("RAG: top-K docs", 1, min(5,len(st.session_state.docs)), 2, key="top_k")
    max_hops = col_p2.slider("GraphRAG: max hops", 1, 6, 4, key="max_hops_compare")
    G = build_nx_graph()
    seed_options = list(G.nodes()) if G.number_of_nodes() > 0 else ["(sin nodos)"]
    seed_node = col_p3.selectbox("GraphRAG: nodo semilla", seed_options, key="seed_compare")

    run = st.button("⚡ Ejecutar ambos", type="primary", use_container_width=True)

    if run and pregunta and G.number_of_nodes() > 0:
        client = get_client(api_key)
        col_rag, col_graph = st.columns(2)

        # ── RAG ──
        with col_rag:
            st.markdown("""<div class="card" style="border-color:#ef4444">
<b style="color:#ef4444">RAG Vectorial</b></div>""", unsafe_allow_html=True)

            with st.spinner("Buscando por similitud..."):
                t0 = time.time()
                results, all_scores = rag_search(pregunta, st.session_state.docs, top_k)
                rag_time = time.time() - t0

            # Similarity chart
            ids = list(all_scores.keys()); scores = list(all_scores.values())
            recovered = {r["id"] for r in results}
            colors = ["#22c55e" if d in recovered else "#ef4444" for d in ids]
            fig_s = go.Figure(go.Bar(x=ids,y=scores,marker_color=colors,
                text=[f"{s:.3f}" for s in scores],textposition="outside",
                textfont=dict(color="#e2e8f0",size=10)))
            fig_s.update_layout(paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),height=220,
                yaxis=dict(gridcolor="#1e293b",title="Similitud"),
                margin=dict(l=40,r=10,t=10,b=40))
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown(f"**Recuperados:** {', '.join(r['id'] for r in results)} | "
                        f"**Perdidos:** {', '.join(d for d in ids if d not in recovered)}")

            ctx_rag = "\n---\n".join(f"{r['id']}:\n{r['text']}" for r in results)
            with st.spinner("Consultando LLM..."):
                resp_rag = ask_llm(pregunta, ctx_rag, client)
            st.text_area("Respuesta RAG", resp_rag, height=200, key="resp_rag_out", disabled=True)

        # ── GRAPHRAG ──
        with col_graph:
            st.markdown("""<div class="card" style="border-color:#22c55e">
<b style="color:#22c55e">GraphRAG</b></div>""", unsafe_allow_html=True)

            with st.spinner("Traversia relacional..."):
                t0 = time.time()
                sg, log = graphrag_traverse(G, seed_node, max_hops)
                graph_time = time.time() - t0

            # Subgraph
            hl_nodes = set(sg.nodes())
            hl_edges = set(sg.edges())
            fig_g = plotly_graph(sg, highlight_nodes=hl_nodes, highlight_edges=hl_edges, height=220)
            st.plotly_chart(fig_g, use_container_width=True)

            # Traversal log
            for entry in log:
                if entry["hop"] == 0:
                    st.markdown(f'<div class="hop-box" style="border-color:#eab308">Semilla: <b>[{entry["to"]}]</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hop-box">Hop {entry["hop"]}: {entry["from"]} --[{entry["rel"]}]--> <b>{entry["to"]}</b></div>', unsafe_allow_html=True)

            ctx_graph = subgraph_to_context(sg)
            with st.spinner("Consultando LLM..."):
                resp_graph = ask_llm(pregunta, ctx_graph, client)
            st.text_area("Respuesta GraphRAG", resp_graph, height=200, key="resp_graph_out", disabled=True)

        # ── METRICS ──
        st.markdown("---")
        st.markdown("**📊 Metricas de la comparacion**")
        m1,m2,m3,m4,m5 = st.columns(5)
        rag_doc_count = len(results)
        graph_node_count = sg.number_of_nodes()
        rag_words = len(resp_rag.split())
        graph_words = len(resp_graph.split())
        can_answer_rag = "no puedo" not in resp_rag.lower()
        can_answer_graph = "no puedo" not in resp_graph.lower()

        m1.metric("RAG: docs usados", f"{rag_doc_count}/{len(st.session_state.docs)}")
        m2.metric("GraphRAG: nodos", f"{graph_node_count}/{G.number_of_nodes()}")
        m3.metric("RAG responde?", "SI" if can_answer_rag else "NO")
        m4.metric("GraphRAG responde?", "SI" if can_answer_graph else "NO")
        m5.metric("Hops usados", max(e["hop"] for e in log))

        # Save to history
        st.session_state.comparison_history.append({
            "pregunta": pregunta[:50],
            "rag_docs": rag_doc_count, "graph_nodes": graph_node_count,
            "rag_ok": can_answer_rag, "graph_ok": can_answer_graph,
            "top_k": top_k, "hops": max_hops,
        })
        st.session_state.comparison_history = st.session_state.comparison_history[-10:]

    # History
    if st.session_state.comparison_history:
        st.markdown("---")
        st.markdown("**Historial de comparaciones**")
        df_h = pd.DataFrame(st.session_state.comparison_history)
        df_h.columns = ["Pregunta","RAG docs","Graph nodos","RAG OK?","Graph OK?","Top-K","Hops"]
        st.dataframe(df_h, use_container_width=True, hide_index=True)

        if st.button("🗑️ Limpiar historial"):
            st.session_state.comparison_history = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — GRAPH EVOLUTION LEVELS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Evolucion del Grafo: 5 Niveles")
    st.caption("Selecciona un nivel para ver como crece el grafo y que preguntas puede responder.")

    nivel = st.radio("Nivel:", [1,2,3,4,5], format_func=lambda x: {
        1:"1 — No Dirigido",2:"2 — Dirigido + Tipos",3:"3 — Property Graph",
        4:"4 — Algoritmos (PageRank)",5:"5 — Context Graph (5 Pilares)"}[x], horizontal=True)

    col_v, col_q = st.columns([2,1])

    # Build level graph
    if nivel == 1:
        Gn = nx.Graph()
        for n in ["BUG-2847","payments-service","billing-service","Region Europa"]:
            Gn.add_node(n, tipo={"BUG-2847":"Bug","payments-service":"Modulo",
                "billing-service":"Servicio","Region Europa":"Region"}[n])
        Gn.add_edges_from([("BUG-2847","payments-service",{"relacion":"---"}),
            ("payments-service","billing-service",{"relacion":"---"}),
            ("billing-service","Region Europa",{"relacion":"---"})])
    elif nivel == 2:
        Gn = nx.DiGraph()
        for nid,t in [("BUG-2847","Bug"),("payments-service","Modulo"),("billing-service","Servicio"),
                      ("Region Europa","Region"),("Goldman EU","Cliente"),("BNP Corporate","Cliente"),
                      ("Siemens Financial","Cliente")]:
            Gn.add_node(nid,tipo=t)
        for s,d,r in [("BUG-2847","payments-service","AFECTA_A"),("payments-service","billing-service","DEPENDE_DE"),
                      ("billing-service","Region Europa","PROCESA_EN"),("Region Europa","Goldman EU","TIENE_CLIENTE"),
                      ("Region Europa","BNP Corporate","TIENE_CLIENTE"),("Region Europa","Siemens Financial","TIENE_CLIENTE")]:
            Gn.add_edge(s,d,relacion=r)
    elif nivel == 3:
        Gn = nx.DiGraph()
        for nid,t,p in [("BUG-2847","Bug",{"severidad":"CRITICO","horas_abiertas":72}),
                        ("payments-service","Modulo",{"version":"v2.3.1"}),
                        ("billing-service","Servicio",{"txs_mes":125000}),
                        ("Region Europa","Region",{"sla":"99.9%"}),
                        ("Goldman EU","Cliente",{"mrr_usd":45000,"tier":"Enterprise"}),
                        ("BNP Corporate","Cliente",{"mrr_usd":38000,"tier":"Enterprise"}),
                        ("Siemens Financial","Cliente",{"mrr_usd":12000,"tier":"Business"})]:
            Gn.add_node(nid,tipo=t,**p)
        for s,d,r in [("BUG-2847","payments-service","AFECTA_A"),("payments-service","billing-service","DEPENDE_DE"),
                      ("billing-service","Region Europa","PROCESA_EN"),("Region Europa","Goldman EU","TIENE_CLIENTE"),
                      ("Region Europa","BNP Corporate","TIENE_CLIENTE"),("Region Europa","Siemens Financial","TIENE_CLIENTE")]:
            Gn.add_edge(s,d,relacion=r)
    elif nivel == 4:
        Gn = nx.DiGraph()
        for nid,t in [("BUG-2847","Bug"),("payments-service","Modulo"),("billing-service","Servicio"),
                      ("Region Europa","Region"),("Goldman EU","Cliente"),("BNP Corporate","Cliente"),
                      ("Siemens Financial","Cliente")]:
            Gn.add_node(nid,tipo=t)
        for s,d,r in [("BUG-2847","payments-service","AFECTA_A"),("payments-service","billing-service","DEPENDE_DE"),
                      ("billing-service","Region Europa","PROCESA_EN"),("Region Europa","Goldman EU","TIENE_CLIENTE"),
                      ("Region Europa","BNP Corporate","TIENE_CLIENTE"),("Region Europa","Siemens Financial","TIENE_CLIENTE")]:
            Gn.add_edge(s,d,relacion=r)
        pr = nx.pagerank(Gn, alpha=0.85)
        nx.set_node_attributes(Gn, pr, "pagerank")
    else:  # nivel 5
        Gn = nx.DiGraph()
        nodes5 = [("BUG-2847","Bug"),("payments-service","Modulo"),("billing-service","Servicio"),
                  ("Region Europa","Region"),("Goldman EU","Cliente"),("BNP Corporate","Cliente"),
                  ("Siemens Financial","Cliente"),
                  ("DEC-2023-11","Decision"),("RFC-0042","Decision"),
                  ("INC-2023-07","Incidente"),("POST-MORT-07","Incidente"),]
        extra_props = {
            "DEC-2023-11":{"titulo":"Separar billing como microservicio","autor":"CTO",
                           "razon":"Cumplimiento GDPR"},
            "RFC-0042":{"titulo":"Integracion sincrona payments->billing",
                        "razon":"Consistencia transaccional EU"},
            "INC-2023-07":{"descripcion":"Fallo similar Q3-2023","duracion":"4h"},
            "POST-MORT-07":{"leccion":"billing hereda SLA de payments"},
        }
        for nid,t in nodes5:
            Gn.add_node(nid, tipo=t, **extra_props.get(nid,{}))
        edges5 = [("BUG-2847","payments-service","AFECTA_A"),("payments-service","billing-service","DEPENDE_DE"),
                  ("billing-service","Region Europa","PROCESA_EN"),("Region Europa","Goldman EU","TIENE_CLIENTE"),
                  ("Region Europa","BNP Corporate","TIENE_CLIENTE"),("Region Europa","Siemens Financial","TIENE_CLIENTE"),
                  ("DEC-2023-11","billing-service","MOTIVO_CREACION"),("RFC-0042","payments-service","DEFINE_INTERFAZ"),
                  ("RFC-0042","billing-service","DEFINE_INTERFAZ"),("INC-2023-07","payments-service","AFECTO_A"),
                  ("POST-MORT-07","INC-2023-07","ANALIZA"),("POST-MORT-07","billing-service","IDENTIFICO_RIESGO")]
        for s,d,r in edges5:
            Gn.add_edge(s,d,relacion=r)

    with col_v:
        fig_n = plotly_graph(Gn, title=f"Nivel {nivel}", height=450)
        st.plotly_chart(fig_n, use_container_width=True)

        if nivel == 4:
            pr = nx.pagerank(Gn, alpha=0.85)
            pr_sorted = sorted(pr.items(), key=lambda x:-x[1])
            fig_pr = go.Figure(go.Bar(
                x=[n for n,_ in pr_sorted], y=[v for _,v in pr_sorted],
                marker_color=[COLORES_TIPO.get(Gn.nodes[n].get("tipo",""),"#6b7280") for n,_ in pr_sorted],
                text=[f"{v:.3f}" for _,v in pr_sorted], textposition="outside",
                textfont=dict(color="#e2e8f0")))
            fig_pr.update_layout(title="PageRank",paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),height=250,yaxis=dict(gridcolor="#1e293b"),
                margin=dict(l=40,r=10,t=40,b=40))
            st.plotly_chart(fig_pr, use_container_width=True)

    with col_q:
        st.metric("Nodos", Gn.number_of_nodes())
        st.metric("Relaciones", Gn.number_of_edges())

        PREGUNTA_CRITICA = "Que decision arquitectonica causo que BUG-2847 afecte $83K/mes de MRR?"
        st.markdown(f"""<div class="card" style="border-color:#a855f7">
<b style="color:#a855f7">Pregunta critica:</b><br>
<span style="font-size:.85rem">{PREGUNTA_CRITICA}</span></div>""", unsafe_allow_html=True)

        st.markdown("---")
        answers = {
            1: [("Conexion bug-Europa?", True, "SI, 3 hops"),
                ("Cuantos clientes?", False, "IMPOSIBLE"),
                ("Direccion impacto?", False, "IMPOSIBLE"),
                ("MRR en riesgo?", False, "IMPOSIBLE")],
            2: [("Conexion bug-Europa?", True, "SI, dirigida"),
                ("Cuantos clientes?", True, "3 clientes"),
                ("Direccion impacto?", True, "BUG->pay->bill->EU"),
                ("MRR en riesgo?", False, "IMPOSIBLE")],
            3: [("Conexion bug-Europa?", True, "SI"),
                ("Cuantos clientes?", True, "3 clientes"),
                ("MRR en riesgo?", True, "$83K/mes Enterprise"),
                ("Por que esta dependencia?", False, "IMPOSIBLE")],
            4: [("Cuello de botella?", True, f"{max(nx.pagerank(Gn),key=nx.pagerank(Gn).get) if nivel==4 else '?'}"),
                ("Comunidades?", True, "3 clusters"),
                ("MRR en riesgo?", True, "$83K/mes"),
                ("Por que esta dependencia?", False, "IMPOSIBLE")],
            5: [("MRR en riesgo?", True, "$83K/mes"),
                ("Cuello de botella?", True, "payments-service"),
                ("Por que?", True, "DEC-2023-11: GDPR"),
                ("Precedentes?", True, "INC-2023-07: fallo Q3")],
        }
        for q, ok, ans in answers.get(nivel, []):
            if ok:
                st.success(f"**{q}** → {ans}")
            else:
                st.error(f"**{q}** → {ans}")

    # Summary table
    st.markdown("---")
    df_lvl = pd.DataFrame([
        {"Nivel":"1","Tipo":"No Dirigido","Nodos":4,"Aristas":3,"Responde":"Estan conectados?"},
        {"Nivel":"2","Tipo":"Dirigido","Nodos":7,"Aristas":6,"Responde":"Quien afecta a quien?"},
        {"Nivel":"3","Tipo":"Property Graph","Nodos":7,"Aristas":6,"Responde":"Cuanto MRR?"},
        {"Nivel":"4","Tipo":"KG + Algoritmos","Nodos":7,"Aristas":6,"Responde":"Cuello de botella?"},
        {"Nivel":"5","Tipo":"Context Graph","Nodos":11,"Aristas":12,"Responde":"Por que existe?"},
    ])
    st.dataframe(df_lvl, use_container_width=True, hide_index=True)
