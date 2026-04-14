import streamlit as st
import json, time, os, math
import plotly.graph_objects as go
import plotly.express as px
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
st.set_page_config(page_title="RAG vs GraphRAG", page_icon="🕸️", layout="wide")
st.markdown("""
<style>
body,.stApp{background:#0f172a;color:#e2e8f0}
.stSidebar{background:#1e293b}
.card{background:#1e293b;border-radius:10px;padding:16px;margin:8px 0;border-left:4px solid}
.score{font-size:2.2rem;font-weight:bold}
h1,h2,h3,h4{color:#e2e8f0!important}
div[data-testid="stMarkdownContainer"] p{color:#e2e8f0}
.hop-box{background:#1e293b;border-radius:8px;padding:12px;margin:6px 0;border-left:3px solid #3b82f6;font-family:monospace;font-size:0.85rem}
</style>""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🕸️ RAG vs GraphRAG")
    st.caption("Demo interactiva: Grafos de Conocimiento")
    st.markdown("---")
    api_key = st.text_input("🔑 OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY",""), placeholder="sk-...")
    if api_key:
        st.success("API key lista")
    else:
        st.info("Sin key -> modo demo (mock)")
    st.markdown("---")
    st.markdown("**Fases de la demo:**")
    st.info("📦 Fase 1: Documentos aislados")
    st.info("⚠️ Fase 2: RAG Vectorial (colapso)")
    st.info("🕸️ Fase 3: Knowledge Graph")
    st.info("🚀 Fase 4: GraphRAG al rescate")
    st.info("📈 Fase 5: Evolucion del Grafo")
    st.markdown("---")
    st.caption("Fenixoft S.A.S. · AI-First SDLC")
    st.caption("Ingenieria de Contexto para Agentes de IA")

# ── BASE DE CONOCIMIENTO ─────────────────────────────────────────────────────
DOCUMENTOS = {
    "DOC-001": (
        "[TICKET BUG-2847] Bug critico en payments-service v2.3.1. "
        "Se detecto una falla al procesar tarjetas de debito con montos superiores a $10,000. "
        "El error lanza una excepcion NullPointerException en el modulo de autorizacion. "
        "Estado: ABIERTO. Prioridad: CRITICA. Asignado a: Equipo Backend. "
        "Fecha de reporte: 2026-03-28."
    ),
    "DOC-002": (
        "[ARQUITECTURA] payments-service es una dependencia directa del billing-service. "
        "El billing-service consume la API REST de payments-service en el endpoint /v2/charge. "
        "Sin payments-service operativo, billing-service no puede emitir facturas. "
        "Nivel de acoplamiento: ALTO. Patron: cliente-servidor sincrono."
    ),
    "DOC-003": (
        "[CONFIGURACION REGIONAL] El billing-service procesa transacciones para tres regiones: "
        "LATAM, EMEA (Europa, Oriente Medio y Africa) y NA (Norteamerica). "
        "La region Europa (EMEA-EU) incluye clientes enterprise en Alemania, Francia, Espana y Reino Unido. "
        "Volumen mensual Europa: 125,000 transacciones. SLA contractual: 99.9% de disponibilidad."
    ),
    "DOC-004": (
        "[DIRECTORIO ENTERPRISE - REGION EUROPA] Clientes activos con contrato enterprise: "
        "(1) Goldman EU -- 50,000 transacciones/mes, renovacion 2027. "
        "(2) BNP Corporate -- 30,000 transacciones/mes, renovacion 2026. "
        "(3) Siemens Financial -- 45,000 transacciones/mes, renovacion 2027. "
        "Todos con SLA de 99.9% y penalidades por incumplimiento del 5% del contrato mensual."
    ),
    "DOC-005": (
        "[MANUAL TECNICO] payments-service v2.3 gestiona autenticacion de tarjetas, "
        "tokenizacion PCI-DSS y procesamiento asincrono de cargos. "
        "Tecnologias: Java 17, Spring Boot 3.1, PostgreSQL 15. "
        "Pipeline de CI/CD: GitHub Actions + ArgoCD. "
        "Propietario del codigo: squad-payments@fenixoft.com"
    ),
}

PREGUNTA = "Cual es el impacto del bug en el modulo de pagos sobre los clientes de Europa?"

# ── FUNCIONES RAG ─────────────────────────────────────────────────────────────
def rag_busqueda_vectorial(pregunta, documentos, top_k=2):
    textos = list(documentos.values())
    ids = list(documentos.keys())
    vectorizador = TfidfVectorizer()
    matriz_tfidf = vectorizador.fit_transform(textos + [pregunta])
    vector_pregunta = matriz_tfidf[-1]
    vectores_docs = matriz_tfidf[:-1]
    similitudes = cosine_similarity(vector_pregunta, vectores_docs)[0]
    top_indices = np.argsort(similitudes)[::-1][:top_k]
    resultados = []
    for idx in top_indices:
        resultados.append({
            "id": ids[idx], "similitud": round(float(similitudes[idx]), 4),
            "texto": textos[idx]
        })
    all_scores = {ids[i]: round(float(similitudes[i]), 4) for i in range(len(ids))}
    return resultados, all_scores

def get_client(key):
    if OAI_OK and key:
        clean = key.strip().encode("ascii", errors="ignore").decode("ascii")
        if clean:
            return OpenAI(api_key=clean)
    return None

def consultar_llm(pregunta, contexto, client=None):
    if client:
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Eres el asistente tecnico de Fenixoft. Responde UNICAMENTE basandote en el contexto proporcionado. Si el contexto no tiene suficiente informacion, di explicitamente que te falta."},
                    {"role":"user","content":f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"}
                ], temperature=0)
            return r.choices[0].message.content
        except:
            pass

    tiene_cadena = ("dependencia" in contexto.lower() and "billing" in contexto.lower()
                    and "region" in contexto.lower() and "europa" in contexto.lower())
    if tiene_cadena:
        return """RESPUESTA DEL LLM (contexto GraphRAG completo)

Basandome en el subgrafo extraido, puedo trazar la cadena de impacto completa (A -> B -> C -> D):

  [A] Bug BUG-2847  --[AFECTA_A]-->  payments-service v2.3.1
  [B] payments-service  --[ES_DEPENDENCIA_DE]-->  billing-service
  [C] billing-service  --[PROCESA_TRANSACCIONES_DE]-->  Region Europa
  [D] Region Europa  --[CONTIENE_CLIENTE]-->  Goldman EU | BNP Corporate | Siemens Financial

IMPACTO DIRECTO CONFIRMADO:
  Goldman EU ............. 50,000 transacciones/mes EN RIESGO
  BNP Corporate .......... 30,000 transacciones/mes EN RIESGO
  Siemens Financial ...... 45,000 transacciones/mes EN RIESGO
  TOTAL IMPACTADO ........ 125,000 transacciones/mes

RIESGO COMERCIAL:
  SLA contractual 99.9% en riesgo de incumplimiento.
  Penalidad aplicable: 5% del contrato mensual por cliente.

ACCION RECOMENDADA:
  [INMEDIATA] Escalar a squad-payments@fenixoft.com
  [1h] Activar protocolo de incidente para region EMEA-EU.
  [2h] Notificar a los 3 clientes enterprise afectados."""
    else:
        return """RESPUESTA DEL LLM (contexto RAG incompleto)

Basandome en la documentacion disponible puedo confirmar:

1. Existe el Bug #BUG-2847 en payments-service v2.3.1, con prioridad
   CRITICA, afectando transacciones de debito superiores a $10,000.

2. Fenixoft tiene clientes enterprise en la Region Europa: Goldman EU,
   BNP Corporate y Siemens Financial.

SIN EMBARGO, con la informacion que tengo disponible NO puedo
establecer si el bug del modulo de pagos tiene impacto directo sobre
los clientes de Europa. Los documentos no especifican la relacion
tecnica entre payments-service y los servicios que atienden esa region.

RECOMENDACION: Consultar con el equipo de arquitectura para determinar
si payments-service es una dependencia de los servicios de facturacion
regionales."""


# ── KNOWLEDGE GRAPH ───────────────────────────────────────────────────────────
def build_knowledge_graph():
    G = nx.DiGraph()
    G.add_node("BUG-2847", tipo="Bug", descripcion="NullPointerException en modulo de autorizacion",
               prioridad="CRITICA", estado="ABIERTO", fecha="2026-03-28")
    G.add_node("payments-service", tipo="Modulo", version="v2.3.1",
               equipo="squad-payments@fenixoft.com", tecnologia="Java 17 / Spring Boot")
    G.add_node("billing-service", tipo="Servicio", endpoint="/v2/charge",
               acoplamiento="ALTO", patron="cliente-servidor sincrono")
    G.add_node("Region Europa", tipo="Region", codigo="EMEA-EU",
               paises="Alemania, Francia, Espana, Reino Unido",
               volumen_mensual=125000, sla="99.9%")
    G.add_node("Goldman EU", tipo="Cliente", transacciones_mes=50000,
               contrato="Enterprise", renovacion="2027")
    G.add_node("BNP Corporate", tipo="Cliente", transacciones_mes=30000,
               contrato="Enterprise", renovacion="2026")
    G.add_node("Siemens Financial", tipo="Cliente", transacciones_mes=45000,
               contrato="Enterprise", renovacion="2027")
    G.add_edge("BUG-2847", "payments-service", relacion="AFECTA_A")
    G.add_edge("payments-service", "billing-service", relacion="ES_DEPENDENCIA_DE")
    G.add_edge("billing-service", "Region Europa", relacion="PROCESA_TRANSACCIONES_DE")
    G.add_edge("Region Europa", "Goldman EU", relacion="CONTIENE_CLIENTE")
    G.add_edge("Region Europa", "BNP Corporate", relacion="CONTIENE_CLIENTE")
    G.add_edge("Region Europa", "Siemens Financial", relacion="CONTIENE_CLIENTE")
    return G

def graphrag_traversal(G, nodo_inicio, max_hops=4):
    nodos_subgrafo = {nodo_inicio}
    nodos_actuales = {nodo_inicio}
    log = []
    log.append({"hop": 0, "from": None, "rel": None, "to": nodo_inicio,
                "tipo": G.nodes[nodo_inicio].get("tipo","?")})
    for hop in range(1, max_hops + 1):
        proximos = set()
        for nodo in nodos_actuales:
            for vecino in G.successors(nodo):
                datos = G.get_edge_data(nodo, vecino)
                tipo = G.nodes[vecino].get("tipo","?")
                log.append({"hop": hop, "from": nodo, "rel": datos["relacion"],
                            "to": vecino, "tipo": tipo})
                proximos.add(vecino)
                nodos_subgrafo.add(vecino)
        nodos_actuales = proximos
        if not nodos_actuales:
            break
    return G.subgraph(nodos_subgrafo).copy(), log

def subgrafo_a_contexto(subgrafo):
    lineas = ["SUBGRAFO EXTRAIDO -- CONTEXTO GRAPHRAG:\n", "ENTIDADES:"]
    for nodo, datos in subgrafo.nodes(data=True):
        props = ", ".join(f"{k}={v}" for k, v in datos.items())
        lineas.append(f"  {nodo} ({props})")
    lineas.append("\nRELACIONES:")
    for u, v, datos in subgrafo.edges(data=True):
        lineas.append(f"  ({u}) --[{datos['relacion']}]--> ({v})")
    return "\n".join(lineas)


# ── PLOTLY GRAPH VISUALIZATION ────────────────────────────────────────────────
COLORES_TIPO = {
    "Bug": "#ef4444", "Modulo": "#3b82f6", "Servicio": "#a855f7",
    "Region": "#eab308", "Cliente": "#22c55e", "Decision": "#f97316",
    "RFC": "#f97316", "Incidente": "#dc2626", "PostMortem": "#dc2626",
    "Documento": "#0ea5e9", "Runbook": "#0ea5e9", "Transaccion": "#10b981",
}
SIMBOLOS_TIPO = {
    "Bug": "diamond", "Modulo": "square", "Servicio": "square",
    "Region": "circle", "Cliente": "circle", "Decision": "hexagon",
    "RFC": "hexagon", "Incidente": "star", "PostMortem": "star",
    "Documento": "triangle-up", "Runbook": "triangle-up", "Transaccion": "pentagon",
}

def plotly_graph(G, title="", highlight_path=None, height=500):
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)

    # Edge traces
    edge_traces = []
    annotations = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rel = data.get("relacion", data.get("rel", ""))
        is_highlight = highlight_path and u in highlight_path and v in highlight_path
        color = "#f97316" if is_highlight else "#475569"
        width = 3 if is_highlight else 1.5

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines", line=dict(width=width, color=color),
            hoverinfo="none", showlegend=False
        ))
        # Arrow annotation
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=width,
            arrowcolor=color, opacity=0.6
        ))
        # Edge label
        mx, my = (x0+x1)/2, (y0+y1)/2
        if rel:
            annotations.append(dict(
                x=mx, y=my, text=f"<i>{rel}</i>",
                showarrow=False, font=dict(size=8, color="#94a3b8"),
                bgcolor="#1e293b", opacity=0.9
            ))

    # Group nodes by type for legend
    tipos = {}
    for node, data in G.nodes(data=True):
        t = data.get("tipo", "Otro")
        tipos.setdefault(t, []).append(node)

    node_traces = []
    for tipo, nodes in tipos.items():
        xs, ys, texts, hovers = [], [], [], []
        for n in nodes:
            x, y = pos[n]
            xs.append(x); ys.append(y)
            texts.append(n)
            props = "<br>".join(f"<b>{k}:</b> {v}" for k, v in G.nodes[n].items() if k != "tipo")
            hovers.append(f"<b>{n}</b> [{tipo}]<br>{props}")

        color = COLORES_TIPO.get(tipo, "#6b7280")
        symbol = SIMBOLOS_TIPO.get(tipo, "circle")
        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=tipo,
            marker=dict(size=18, color=color, symbol=symbol, line=dict(width=2, color="#0f172a")),
            text=texts, textposition="top center",
            textfont=dict(size=9, color="#e2e8f0"),
            hovertext=hovers, hoverinfo="text"
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=14)),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height, margin=dict(l=20,r=20,t=50,b=20),
        font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0"), bgcolor="#1e293b",
                    bordercolor="#334155", borderwidth=1),
        annotations=annotations,
        hoverlabel=dict(bgcolor="#1e293b", font_size=11, font_color="#e2e8f0")
    )
    return fig


# ── BUILD LEVEL GRAPHS ───────────────────────────────────────────────────────
def build_level1():
    G = nx.Graph()
    G.add_nodes_from([
        ("BUG-2847", {"tipo":"Bug"}), ("payments-service", {"tipo":"Modulo"}),
        ("billing-service", {"tipo":"Servicio"}), ("Region Europa", {"tipo":"Region"})
    ])
    G.add_edges_from([
        ("BUG-2847","payments-service",{"relacion":"---"}),
        ("payments-service","billing-service",{"relacion":"---"}),
        ("billing-service","Region Europa",{"relacion":"---"})
    ])
    return G

def build_level2():
    G = nx.DiGraph()
    COLORES_TIPO_L = {'Bug':'#ef4444','Modulo':'#3b82f6','Servicio':'#a855f7',
                      'Region':'#eab308','Cliente':'#22c55e'}
    nodos = [('BUG-2847',{'tipo':'Bug'}),('payments-service',{'tipo':'Modulo'}),
             ('billing-service',{'tipo':'Servicio'}),('Region Europa',{'tipo':'Region'}),
             ('Goldman EU',{'tipo':'Cliente'}),('BNP Corporate',{'tipo':'Cliente'}),
             ('Siemens Financial',{'tipo':'Cliente'})]
    for n, p in nodos:
        G.add_node(n, **p)
    aristas = [('BUG-2847','payments-service',{'rel':'AFECTA_A'}),
               ('payments-service','billing-service',{'rel':'DEPENDE_DE'}),
               ('billing-service','Region Europa',{'rel':'PROCESA_EN'}),
               ('Region Europa','Goldman EU',{'rel':'TIENE_CLIENTE'}),
               ('Region Europa','BNP Corporate',{'rel':'TIENE_CLIENTE'}),
               ('Region Europa','Siemens Financial',{'rel':'TIENE_CLIENTE'})]
    for s,d,p in aristas:
        G.add_edge(s, d, **p)
    return G

def build_level3():
    G = build_level2()
    G.nodes['Goldman EU'].update({'mrr_usd':45000,'tier':'Enterprise'})
    G.nodes['BNP Corporate'].update({'mrr_usd':38000,'tier':'Enterprise'})
    G.nodes['Siemens Financial'].update({'mrr_usd':12000,'tier':'Business'})
    G.nodes['BUG-2847'].update({'severidad':'CRITICO','horas_abiertas':72})
    G.nodes['billing-service'].update({'txs_mes':125000,'valor_usd':95_000_000})
    return G

def build_level5():
    G5 = nx.DiGraph()
    p1 = [('BUG-2847',{'pilar':1,'tipo':'Bug','color':'#ef4444'}),
          ('payments-service',{'pilar':1,'tipo':'Modulo','color':'#3b82f6'}),
          ('billing-service',{'pilar':1,'tipo':'Servicio','color':'#a855f7'}),
          ('Region Europa',{'pilar':1,'tipo':'Region','color':'#eab308'}),
          ('Goldman EU',{'pilar':1,'tipo':'Cliente','color':'#22c55e'}),
          ('BNP Corporate',{'pilar':1,'tipo':'Cliente','color':'#22c55e'}),
          ('Siemens Financial',{'pilar':1,'tipo':'Cliente','color':'#22c55e'})]
    p2 = [('DEC-2023-11',{'pilar':2,'tipo':'Decision','color':'#f97316',
            'titulo':'Separar billing como microservicio',
            'autor':'CTO','fecha':'2023-11-08',
            'razon':'Cumplimiento GDPR requeria aislamiento de facturacion'}),
          ('RFC-0042',{'pilar':2,'tipo':'RFC','color':'#f97316',
            'titulo':'Integracion sincrona payments->billing',
            'autor':'Arq. Software','fecha':'2023-11-15',
            'razon':'Consistencia transaccional requerida por regulacion EU'})]
    p3 = [('INC-2023-07',{'pilar':3,'tipo':'Incidente','color':'#dc2626',
            'descripcion':'Fallo similar en payments-service Q3-2023',
            'duracion_hrs':4,'resolucion':'Patch en timeout handling'}),
          ('POST-MORT-07',{'pilar':3,'tipo':'PostMortem','color':'#dc2626',
            'leccion':'billing-service hereda el SLA de payments-service'})]
    p4 = [('DOC-ARCH-001',{'pilar':4,'tipo':'Documento','color':'#0ea5e9',
            'titulo':'Diagrama de dependencias microservicios','similitud_bug':0.87}),
          ('RUNBOOK-PAY',{'pilar':4,'tipo':'Runbook','color':'#0ea5e9',
            'titulo':'Procedimiento de rollback payments-service','similitud_bug':0.92})]
    p5 = [('TX-BATCH-2024Q1',{'pilar':5,'tipo':'Transaccion','color':'#10b981',
            'volumen_txs':125000,'valor_usd':95_000_000,'afectadas_pct':23})]
    for pilar in [p1,p2,p3,p4,p5]:
        for n,d in pilar:
            G5.add_node(n, **d)
    aristas = [
        ('BUG-2847','payments-service',{'rel':'AFECTA_A','p':1}),
        ('payments-service','billing-service',{'rel':'DEPENDE_DE','p':1}),
        ('billing-service','Region Europa',{'rel':'PROCESA_EN','p':1}),
        ('Region Europa','Goldman EU',{'rel':'TIENE_CLIENTE','p':1}),
        ('Region Europa','BNP Corporate',{'rel':'TIENE_CLIENTE','p':1}),
        ('Region Europa','Siemens Financial',{'rel':'TIENE_CLIENTE','p':1}),
        ('DEC-2023-11','billing-service',{'rel':'MOTIVO_CREACION','p':2}),
        ('RFC-0042','payments-service',{'rel':'DEFINE_INTERFAZ','p':2}),
        ('RFC-0042','billing-service',{'rel':'DEFINE_INTERFAZ','p':2}),
        ('INC-2023-07','payments-service',{'rel':'AFECTO_A','p':3}),
        ('POST-MORT-07','INC-2023-07',{'rel':'ANALIZA','p':3}),
        ('POST-MORT-07','billing-service',{'rel':'IDENTIFICO_RIESGO','p':3}),
        ('DOC-ARCH-001','payments-service',{'rel':'DOCUMENTA','p':4}),
        ('RUNBOOK-PAY','BUG-2847',{'rel':'GUIA_RESOLUCION','p':4}),
        ('TX-BATCH-2024Q1','billing-service',{'rel':'ES_PROCESADO_POR','p':5}),
        ('TX-BATCH-2024Q1','Goldman EU',{'rel':'ORIGINADO_POR','p':5}),
        ('TX-BATCH-2024Q1','BNP Corporate',{'rel':'ORIGINADO_POR','p':5}),
    ]
    for s,d,p in aristas:
        G5.add_edge(s,d,**p)
    return G5


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 RAG Vectorial (El Problema)",
    "🕸️ Knowledge Graph",
    "🚀 GraphRAG al Rescate",
    "📈 5 Niveles de Grafos",
    "🏆 Comparacion Final",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAG VECTORIAL
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("La Ilusion del RAG Vectorial")
    st.caption("El RAG vectorial busca por similitud lexica. Cuando la respuesta requiere razonamiento multi-hop, colapsa.")

    st.markdown("""<div class="card" style="border-color:#eab308">
<b style="color:#eab308">La trampa esta aqui:</b><br>
<span style="font-size:.9rem">El documento del <b>Bug</b> no menciona Europa.
El documento de <b>clientes Europa</b> no menciona el bug.
Los documentos intermedios B y C son el <b>eslabon perdido</b>.</span><br><br>
<code style="color:#94a3b8">A (Bug) --> B (Dependencia) --> C (Region) --> D (Clientes Europa)</code>
</div>""", unsafe_allow_html=True)

    col_docs, col_rag = st.columns([1, 1])

    with col_docs:
        st.markdown("**📄 Base de Conocimiento (5 documentos aislados)**")
        for doc_id, texto in DOCUMENTOS.items():
            is_link = doc_id in ("DOC-002", "DOC-003")
            border = "#f97316" if is_link else "#334155"
            label = " (ESLABON PERDIDO)" if is_link else ""
            st.markdown(f"""<div class="card" style="border-color:{border}">
<b>{doc_id}{label}</b><br>
<span style="font-size:.82rem">{texto[:150]}...</span>
</div>""", unsafe_allow_html=True)

    with col_rag:
        st.markdown(f"**🔍 Pregunta:** *{PREGUNTA}*")

        if st.button("▶️ Ejecutar busqueda RAG vectorial", type="primary", use_container_width=True):
            with st.spinner("Calculando similitudes TF-IDF..."):
                time.sleep(0.5)
                resultados, all_scores = rag_busqueda_vectorial(PREGUNTA, DOCUMENTOS, top_k=2)
                st.session_state["rag_results"] = resultados
                st.session_state["rag_all_scores"] = all_scores

        if st.session_state.get("rag_results"):
            resultados = st.session_state["rag_results"]
            all_scores = st.session_state["rag_all_scores"]

            # Similarity bar chart
            doc_ids = list(all_scores.keys())
            scores = list(all_scores.values())
            recovered = {r["id"] for r in resultados}
            colors = ["#22c55e" if d in recovered else "#ef4444" for d in doc_ids]

            fig_sim = go.Figure(go.Bar(
                x=doc_ids, y=scores,
                marker_color=colors,
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
                textfont=dict(color="#e2e8f0")
            ))
            fig_sim.update_layout(
                title="Similitud coseno con la pregunta",
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=300,
                yaxis=dict(title="Similitud", gridcolor="#1e293b"),
                xaxis=dict(title="Documento"),
                margin=dict(l=40,r=20,t=50,b=40)
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            st.markdown("**Recuperados (top-2):**")
            for r in resultados:
                st.markdown(f'<div class="card" style="border-color:#22c55e"><b style="color:#22c55e">{r["id"]}</b> — Similitud: {r["similitud"]:.4f}</div>', unsafe_allow_html=True)

            st.markdown("**No recuperados (eslabones perdidos):**")
            for doc_id in DOCUMENTOS:
                if doc_id not in recovered:
                    st.markdown(f'<div class="card" style="border-color:#ef4444"><b style="color:#ef4444">{doc_id}</b> — Similitud demasiado baja</div>', unsafe_allow_html=True)

            # LLM response
            st.markdown("---")
            st.markdown("**🤖 Respuesta del LLM con contexto incompleto:**")
            client = get_client(api_key)
            contexto_rag = "\n---\n".join(f"{r['id']}:\n{r['texto']}" for r in resultados)
            resp = consultar_llm(PREGUNTA, contexto_rag, client)
            st.markdown(f"""<div class="card" style="border-color:#ef4444">
<b style="color:#ef4444">RAG Vectorial — Contexto Incompleto</b><br><br>
<pre style="white-space:pre-wrap;color:#e2e8f0;font-size:.82rem">{resp}</pre>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE GRAPH
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("El Gemelo Digital: Knowledge Graph de Fenixoft")
    st.caption("Las relaciones que el RAG ignoro quedan explicitas y navegables en el grafo.")

    G = build_knowledge_graph()

    col_graph, col_info = st.columns([2, 1])

    with col_graph:
        chain = {"BUG-2847","payments-service","billing-service","Region Europa",
                 "Goldman EU","BNP Corporate","Siemens Financial"}
        fig_kg = plotly_graph(G, title="Knowledge Graph — Fenixoft (cadena BUG-2847 -> Clientes Europa)",
                              highlight_path=chain, height=500)
        st.plotly_chart(fig_kg, use_container_width=True)

    with col_info:
        st.markdown("**📍 Nodos del grafo:**")
        iconos = {"Bug":"🔴","Modulo":"🔵","Servicio":"🟣","Region":"🟡","Cliente":"🟢"}
        for nodo, datos in G.nodes(data=True):
            tipo = datos.get("tipo","")
            icono = iconos.get(tipo,"⚪")
            st.markdown(f"{icono} **{nodo}** ({tipo})")

        st.markdown("---")
        st.markdown("**🔗 Relaciones:**")
        for u, v, datos in G.edges(data=True):
            st.markdown(f'<span style="font-size:.82rem;color:#94a3b8">({u}) --[{datos["relacion"]}]--> ({v})</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"**Nodos:** {G.number_of_nodes()} | **Relaciones:** {G.number_of_edges()}")

    # Node detail expander
    st.markdown("---")
    st.markdown("**🔍 Explora los nodos — haz click para ver propiedades:**")
    node_cols = st.columns(min(len(G.nodes), 4))
    for i, (nodo, datos) in enumerate(G.nodes(data=True)):
        col = node_cols[i % len(node_cols)]
        tipo = datos.get("tipo","")
        color = COLORES_TIPO.get(tipo, "#6b7280")
        with col:
            with st.expander(f"{iconos.get(tipo,'⚪')} {nodo}"):
                for k, v in datos.items():
                    if k != "tipo":
                        st.markdown(f"**{k}:** {v}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPHRAG
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("GraphRAG al Rescate: Travesia Relacional")
    st.caption("GraphRAG opera en dos fases: entry points vectoriales + travesia relacional.")

    G = build_knowledge_graph()

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.markdown("**Configuracion de la travesia:**")
        nodo_semilla = st.selectbox("Nodo semilla (entry point):",
                                     list(G.nodes()), index=0)
        max_hops = st.slider("Profundidad maxima (hops):", 1, 4, 4)

        run_graphrag = st.button("🕸️ Ejecutar GraphRAG Traversal", type="primary", use_container_width=True)

    if run_graphrag:
        st.session_state["graphrag_run"] = True
        st.session_state["graphrag_seed"] = nodo_semilla
        st.session_state["graphrag_hops"] = max_hops

    if st.session_state.get("graphrag_run"):
        seed = st.session_state["graphrag_seed"]
        hops = st.session_state["graphrag_hops"]
        subgrafo, log = graphrag_traversal(G, seed, hops)

        with col_ctrl:
            st.markdown("---")
            st.markdown("**📋 Log de travesia:**")
            for entry in log:
                if entry["hop"] == 0:
                    st.markdown(f'<div class="hop-box" style="border-color:#eab308">🔍 <b>Semilla:</b> [{entry["to"]}] ({entry["tipo"]})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hop-box"><b>Salto {entry["hop"]}:</b> ({entry["from"]}) --[{entry["rel"]}]--> ({entry["to"]}) [{entry["tipo"]}]</div>', unsafe_allow_html=True)

            st.markdown(f"""<div class="card" style="border-color:#22c55e">
<b>Subgrafo extraido:</b> {subgrafo.number_of_nodes()} nodos, {subgrafo.number_of_edges()} relaciones
</div>""", unsafe_allow_html=True)

        with col_viz:
            path_nodes = set(subgrafo.nodes())
            fig_sub = plotly_graph(subgrafo, title=f"Subgrafo GraphRAG desde [{seed}] ({hops} hops)",
                                   highlight_path=path_nodes, height=450)
            st.plotly_chart(fig_sub, use_container_width=True)

            # LLM with complete context
            st.markdown("**🤖 Respuesta del LLM con contexto GraphRAG completo:**")
            ctx = subgrafo_a_contexto(subgrafo)
            client = get_client(api_key)
            resp_graph = consultar_llm(PREGUNTA, ctx, client)
            st.markdown(f"""<div class="card" style="border-color:#22c55e">
<b style="color:#22c55e">GraphRAG — Contexto Completo (A->B->C->D)</b><br><br>
<pre style="white-space:pre-wrap;color:#e2e8f0;font-size:.82rem">{resp_graph}</pre>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — 5 NIVELES
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Evolucion del Grafo: 5 Niveles")
    st.caption("Del dato aislado al gemelo digital completo. Cada nivel responde mas preguntas.")

    # Level selector
    nivel = st.radio("Selecciona el nivel:", [1,2,3,4,5],
                     format_func=lambda x: {
                         1:"Nivel 1: Grafo No Dirigido Simple",
                         2:"Nivel 2: Grafo Dirigido con Etiquetas",
                         3:"Nivel 3: Property Graph (Neo4j)",
                         4:"Nivel 4: Knowledge Graph + Algoritmos",
                         5:"Nivel 5: Context Graph (5 Pilares)"
                     }[x], horizontal=True)

    col_g, col_i = st.columns([2, 1])

    PREGUNTA_CRITICA = "Que decision arquitectonica causo que BUG-2847 afecte $83K/mes de MRR Enterprise?"

    with col_i:
        st.markdown(f"""<div class="card" style="border-color:#a855f7">
<b style="color:#a855f7">Pregunta critica:</b><br>
<span style="font-size:.88rem">{PREGUNTA_CRITICA}</span>
</div>""", unsafe_allow_html=True)

    if nivel == 1:
        G1 = build_level1()
        with col_g:
            fig1 = plotly_graph(G1, "Nivel 1: Grafo No Dirigido Simple", height=400)
            st.plotly_chart(fig1, use_container_width=True)
        with col_i:
            st.metric("Nodos", G1.number_of_nodes())
            st.metric("Aristas", G1.number_of_edges())
            st.markdown("---")
            st.markdown("**Que puede responder:**")
            st.success("Existe conexion entre bug y Europa? SI (3 hops)")
            st.markdown("**Que NO puede responder:**")
            st.error("Cuantos clientes afectados? IMPOSIBLE")
            st.error("En que direccion viaja el impacto? IMPOSIBLE")
            st.error("Cuanto MRR en riesgo? IMPOSIBLE")

    elif nivel == 2:
        G2 = build_level2()
        with col_g:
            fig2 = plotly_graph(G2, "Nivel 2: Grafo Dirigido con Etiquetas", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        with col_i:
            st.metric("Nodos", G2.number_of_nodes())
            st.metric("Aristas", G2.number_of_edges())
            afectados = [n for n in G2.nodes() if G2.nodes[n].get('tipo')=='Cliente'
                         and nx.has_path(G2, 'BUG-2847', n)]
            st.markdown("---")
            st.markdown("**Que puede responder:**")
            st.success(f"Clientes afectados: {len(afectados)} ({', '.join(afectados)})")
            st.success("Direccion del impacto: BUG -> payments -> billing -> Europa -> Clientes")
            st.markdown("**Que NO puede responder:**")
            st.error("Cuanto MRR en riesgo? IMPOSIBLE (sin propiedades)")

    elif nivel == 3:
        G3 = build_level3()
        with col_g:
            fig3 = plotly_graph(G3, "Nivel 3: Property Graph (Neo4j)", height=400)
            st.plotly_chart(fig3, use_container_width=True)
        with col_i:
            st.metric("Nodos", G3.number_of_nodes())
            st.metric("Aristas", G3.number_of_edges())
            enterprise = [n for n in G3.nodes() if G3.nodes[n].get('tier')=='Enterprise'
                          and nx.has_path(G3, 'BUG-2847', n)]
            mrr = sum(G3.nodes[c]['mrr_usd'] for c in enterprise)
            st.markdown("---")
            st.markdown("**Que puede responder:**")
            st.success(f"MRR Enterprise en riesgo: ${mrr:,}/mes")
            for c in enterprise:
                st.info(f"  {c}: ${G3.nodes[c]['mrr_usd']:,}/mes")
            st.markdown("**Que NO puede responder:**")
            st.error("Que decision creo esta dependencia? IMPOSIBLE")

            # Show Cypher
            with st.expander("Ver Cypher equivalente"):
                st.code("""MATCH (b:Bug)-[:AFECTA_A*..4]->(r:Region)
      -[:TIENE_CLIENTE]->(c:Cliente)
WHERE c.tier = 'Enterprise'
RETURN sum(c.mrr_usd) AS mrr_en_riesgo,
       count(c) AS clientes_enterprise""", language="cypher")

    elif nivel == 4:
        G2 = build_level2()
        G4 = G2.copy()
        pagerank = nx.pagerank(G4, alpha=0.85)
        nx.set_node_attributes(G4, pagerank, 'pagerank')
        comunidades = {'BUG-2847':0,'payments-service':1,'billing-service':1,
                       'Region Europa':2,'Goldman EU':2,'BNP Corporate':2,'Siemens Financial':2}
        nx.set_node_attributes(G4, comunidades, 'comunidad')

        with col_g:
            fig4 = plotly_graph(G4, "Nivel 4: Knowledge Graph + Algoritmos", height=400)
            st.plotly_chart(fig4, use_container_width=True)

            # PageRank bar chart
            pr_sorted = sorted(pagerank.items(), key=lambda x: -x[1])
            fig_pr = go.Figure(go.Bar(
                x=[n for n,_ in pr_sorted], y=[v for _,v in pr_sorted],
                marker_color=[COLORES_TIPO.get(G4.nodes[n].get('tipo',''),'#6b7280') for n,_ in pr_sorted],
                text=[f"{v:.4f}" for _,v in pr_sorted], textposition="outside",
                textfont=dict(color="#e2e8f0")
            ))
            fig_pr.update_layout(
                title="PageRank — Importancia relativa de cada nodo",
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"), height=280,
                yaxis=dict(gridcolor="#1e293b"), margin=dict(l=40,r=20,t=50,b=40)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with col_i:
            nodo_critico = max(pagerank, key=pagerank.get)
            st.metric("Nodo mas critico (PageRank)", nodo_critico)
            st.metric("PageRank", f"{pagerank[nodo_critico]:.4f}")
            st.metric("Comunidades detectadas", len(set(comunidades.values())))
            st.markdown("---")
            st.markdown("**Comunidades:**")
            com_names = {0:"Incidencias", 1:"Servicios de Pago", 2:"Geografia + Clientes"}
            for cid, cname in com_names.items():
                members = [n for n,c in comunidades.items() if c==cid]
                st.info(f"**{cname}:** {', '.join(members)}")
            st.markdown("---")
            st.success(f"Cuello de botella: {nodo_critico}")
            st.error("Que decision creo la dependencia? AUN IMPOSIBLE")

    elif nivel == 5:
        G5 = build_level5()
        with col_g:
            fig5 = plotly_graph(G5, "Nivel 5: Context Graph — 5 Pilares", height=550)
            st.plotly_chart(fig5, use_container_width=True)
        with col_i:
            st.metric("Nodos", G5.number_of_nodes())
            st.metric("Aristas", G5.number_of_edges())
            nombres_p = {1:'Static Domain',2:'Decision Traces',3:'Historical Context',
                         4:'Semantic Embeddings',5:'Transaction Graph'}
            pilar_colors = {1:'#475569',2:'#f97316',3:'#dc2626',4:'#0ea5e9',5:'#10b981'}
            st.markdown("---")
            st.markdown("**Los 5 Pilares:**")
            for pid, pname in nombres_p.items():
                count = sum(1 for _,d in G5.nodes(data=True) if d.get('pilar')==pid)
                st.markdown(f'<span style="color:{pilar_colors[pid]}">Pilar {pid}: <b>{pname}</b> ({count} nodos)</span>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**RESPUESTA COMPLETA:**")
            for n in G5.predecessors('billing-service'):
                d = G5.nodes[n]
                if d.get('pilar') == 2:
                    st.success(f"**{n}**: {d.get('titulo','')}\nAutor: {d.get('autor','')} | Razon: {d.get('razon','')}")
            for n in G5.predecessors('payments-service'):
                d = G5.nodes[n]
                if d.get('pilar') == 2:
                    st.success(f"**{n}**: {d.get('titulo','')}\nRazon: {d.get('razon','')}")
            for n in G5.predecessors('payments-service'):
                d = G5.nodes[n]
                if d.get('pilar') == 3:
                    st.warning(f"Precedente: **{n}** — {d.get('descripcion','')} ({d.get('duracion_hrs','?')}h)")

    # Summary table always visible
    st.markdown("---")
    st.markdown("**Resumen: lo que responde cada nivel**")
    df_niveles = pd.DataFrame([
        {"Nivel":"1 - No Dirigido","Nodos":4,"Aristas":3,"Pregunta respondible":"Estan conectados?"},
        {"Nivel":"2 - Dirigido","Nodos":7,"Aristas":6,"Pregunta respondible":"Quien afecta a quien?"},
        {"Nivel":"3 - Property Graph","Nodos":"7+","Aristas":"6+","Pregunta respondible":"Cuanto MRR en riesgo?"},
        {"Nivel":"4 - Knowledge Graph","Nodos":"7+","Aristas":"6+","Pregunta respondible":"Cuello de botella?"},
        {"Nivel":"5 - Context Graph","Nodos":17,"Aristas":17,"Pregunta respondible":"Por que existe esa arquitectura?"},
    ])
    st.dataframe(df_niveles, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARACION FINAL
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Comparacion Final: RAG Vectorial vs. GraphRAG")
    st.caption("La diferencia entre adivinar y razonar.")

    # Comparison table
    COMPARACION = [
        ("Documentos recuperados", "DOC-001 + DOC-004 (solo similares)", "Cadena completa A->B->C->D"),
        ("Nodos intermedios (B, C)", "Ignorados por baja similitud", "Incluidos por travesia relacional"),
        ("Respuesta sobre impacto", "Incompleta / No puede confirmar", "125,000 tx/mes de 3 clientes"),
        ("Explicabilidad", "Caja negra", "Cada salto es auditable"),
        ("EU AI Act compliance", "Sin trazabilidad", "Subgrafo como evidencia"),
        ("Riesgo de alucinacion", "ALTO — contexto incompleto", "BAJO — datos verificados"),
    ]

    col_t, col_r = st.columns([2, 1])

    with col_t:
        df_comp = pd.DataFrame(COMPARACION, columns=["Dimension", "RAG Vectorial", "GraphRAG"])
        st.dataframe(df_comp, use_container_width=True, hide_index=True, height=280)

    with col_r:
        # Radar comparison
        dims = ["Docs\nRecuperados","Multi-hop","Impacto","Explicabilidad","Compliance","Anti-alucinacion"]
        rag_vals = [0.4, 0.1, 0.2, 0.1, 0.1, 0.2]
        graph_vals = [0.95, 0.95, 0.95, 0.95, 0.9, 0.9]

        fig_radar = go.Figure()
        for vals, name, color in [(rag_vals,"RAG Vectorial","#ef4444"),
                                   (graph_vals,"GraphRAG","#22c55e")]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=dims + [dims[0]],
                fill="toself", name=name, line=dict(color=color, width=2), opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0,1], color="#94a3b8"), bgcolor="#1e293b",
                       angularaxis=dict(color="#94a3b8")),
            paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
            legend=dict(font=dict(color="#e2e8f0")), height=350,
            margin=dict(l=60,r=60,t=30,b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Impact metrics
    st.markdown("**📊 Impacto Cuantificado (solo con GraphRAG):**")
    clientes = {"Goldman EU": 50_000, "BNP Corporate": 30_000, "Siemens Financial": 45_000}
    total_tx = sum(clientes.values())

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Goldman EU", f"{50_000:,} tx/mes", "Enterprise")
    mc2.metric("BNP Corporate", f"{30_000:,} tx/mes", "Enterprise")
    mc3.metric("Siemens Financial", f"{45_000:,} tx/mes", "Enterprise")
    mc4.metric("TOTAL EN RIESGO", f"{total_tx:,} tx/mes", "SLA 99.9% en riesgo")

    # Conclusions
    st.markdown("---")
    st.markdown("### Conclusiones")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="card" style="border-color:#ef4444">
<b style="color:#ef4444">RAG Vectorial</b><br>
<b>Ve:</b> Similitud lexica<br>
<b>Recupera:</b> Fragmentos aislados<br>
<b>Razona:</b> Dentro de un fragmento<br>
<b>Explica:</b> No sabe por que
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card" style="border-color:#22c55e">
<b style="color:#22c55e">GraphRAG</b><br>
<b>Ve:</b> Conexiones relacionales<br>
<b>Recupera:</b> Vecindarios de entidades<br>
<b>Razona:</b> A traves de multiples saltos<br>
<b>Explica:</b> Cada arista es auditable
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="card" style="border-color:#a855f7;text-align:center;margin-top:16px">
<b style="color:#a855f7;font-size:1.1rem">Garbage In, Garbage Out</b><br>
<span style="font-size:.9rem">El RAG vectorial no fallo por ser un mal retriever —<br>
fallo porque su arquitectura es <b>ciega a las relaciones</b>.</span>
</div>""", unsafe_allow_html=True)
