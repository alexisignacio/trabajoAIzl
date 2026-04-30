
from __future__ import annotations

import copy
import json
import math
import unicodedata
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from matplotlib.ticker import FuncFormatter
from streamlit_folium import st_folium


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_tarea_ind.xlsx"
GEOJSON_FILE = BASE_DIR / "comunas_metropolitana-1.geojson"


st.set_page_config(
    page_title="Análisis geoespacial de ventas",
    page_icon="",
    layout="wide",
)


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def parse_decimal_series(series: pd.Series) -> pd.Series:
    def parse_one(value: object) -> float:
        if pd.isna(value):
            return math.nan
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if "," in text:
            text = text.replace(".", "").replace(",", ".")
        return pd.to_numeric(text, errors="coerce")

    return series.map(parse_one)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)

    for column in ["venta_neta", "lat", "lng", "kms_dist", "lat_cd", "lng_cd"]:
        df[column] = parse_decimal_series(df[column])

    df["fecha_compra"] = pd.to_datetime(
        df["fecha_compra"], format="%d-%m-%y", errors="coerce"
    )

    for column in ["canal", "centro_dist", "comuna", "city", "state"]:
        df[column] = df[column].astype(str).str.strip()

    df["mes"] = df["fecha_compra"].dt.to_period("M").astype(str)
    df["comuna_key"] = df["comuna"].map(normalize_text)
    return df


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    with open(GEOJSON_FILE, encoding="utf-8") as file:
        return json.load(file)


def money(value: float) -> str:
    return f"${value:,.0f}".replace(",", ".")


def compact_money(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f} mil MM".replace(".", ",")
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f} MM".replace(".", ",")
    return money(value)


def money_axis(value: float, _: object) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    return f"${value:,.0f}".replace(",", ".")


def commune_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("comuna", as_index=False)
        .agg(
            ordenes=("orden", "count"),
            venta_neta=("venta_neta", "sum"),
            unidades=("unidades", "sum"),
            productos=("productos", "sum"),
            ticket_promedio=("venta_neta", "mean"),
            kms_promedio=("kms_dist", "mean"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
        )
        .sort_values("venta_neta", ascending=False)
    )


def center_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("centro_dist", as_index=False)
        .agg(
            ordenes=("orden", "count"),
            venta_neta=("venta_neta", "sum"),
            unidades=("unidades", "sum"),
            comunas=("comuna", "nunique"),
            kms_promedio=("kms_dist", "mean"),
            lat_cd=("lat_cd", "first"),
            lng_cd=("lng_cd", "first"),
        )
        .sort_values("venta_neta", ascending=False)
    )


def metric_table(df: pd.DataFrame) -> pd.DataFrame:
    total_sales = df["venta_neta"].sum()
    orders = len(df)
    units = df["unidades"].sum()
    ticket = df["venta_neta"].mean()
    distance = df["kms_dist"].mean()
    top_commune = commune_summary(df).iloc[0]["comuna"] if orders else "Sin datos"

    return pd.DataFrame(
        {
            "Indicador": [
                "Venta neta",
                "Ordenes",
                "Unidades",
                "Ticket promedio",
                "Distancia promedio",
                "Comuna lider",
            ],
            "Valor": [
                compact_money(total_sales),
                f"{orders:,.0f}".replace(",", "."),
                f"{units:,.0f}".replace(",", "."),
                money(ticket),
                f"{distance:.1f} km".replace(".", ","),
                top_commune,
            ],
        }
    )


def build_base_map(df: pd.DataFrame) -> folium.Map:
    center = [df["lat"].mean(), df["lng"].mean()]
    return folium.Map(
        location=center,
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True,
    )


def add_layer_control(mapa: folium.Map) -> folium.Map:
    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa


def logistics_map(df: pd.DataFrame) -> folium.Map:
    mapa = build_base_map(df)
    centers = center_summary(df)
    communes = commune_summary(df)

    center_cluster = MarkerCluster(name="Centros de distribucion").add_to(mapa)
    for _, row in centers.iterrows():
        popup = (
            f"<b>{row['centro_dist']}</b><br>"
            f"Venta: {money(row['venta_neta'])}<br>"
            f"Ordenes: {row['ordenes']:,.0f}<br>"
            f"Comunas atendidas: {row['comunas']}"
        ).replace(",", ".")
        folium.Marker(
            [row["lat_cd"], row["lng_cd"]],
            tooltip=row["centro_dist"],
            popup=popup,
            icon=folium.Icon(color="blue", icon="shopping-cart", prefix="fa"),
        ).add_to(center_cluster)

    max_sales = max(communes["venta_neta"].max(), 1)
    for _, row in communes.iterrows():
        radius = 5 + 18 * math.sqrt(row["venta_neta"] / max_sales)
        popup = (
            f"<b>{row['comuna']}</b><br>"
            f"Venta: {money(row['venta_neta'])}<br>"
            f"Ordenes: {row['ordenes']:,.0f}<br>"
            f"Ticket promedio: {money(row['ticket_promedio'])}<br>"
            f"Distancia promedio: {row['kms_promedio']:.1f} km"
        ).replace(",", ".")
        folium.CircleMarker(
            [row["lat"], row["lng"]],
            radius=radius,
            color="#d1495b",
            fill=True,
            fill_color="#d1495b",
            fill_opacity=0.55,
            weight=1,
            tooltip=row["comuna"],
            popup=popup,
        ).add_to(mapa)

    routes = (
        df.groupby(["centro_dist", "comuna"], as_index=False)
        .agg(
            venta_neta=("venta_neta", "sum"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
            lat_cd=("lat_cd", "first"),
            lng_cd=("lng_cd", "first"),
        )
        .sort_values("venta_neta", ascending=False)
        .head(45)
    )
    max_route = max(routes["venta_neta"].max(), 1)
    for _, row in routes.iterrows():
        folium.PolyLine(
            [(row["lat_cd"], row["lng_cd"]), (row["lat"], row["lng"])],
            color="#3d5a80",
            weight=0.8 + 4 * row["venta_neta"] / max_route,
            opacity=0.28,
            tooltip=f"{row['centro_dist']} -> {row['comuna']}",
        ).add_to(mapa)

    return add_layer_control(mapa)


def heat_map(df: pd.DataFrame, weight_column: str) -> folium.Map:
    mapa = build_base_map(df)
    heat = df[["lat", "lng", weight_column]].dropna().copy()

    if weight_column == "ordenes":
        heat["peso"] = 1
    else:
        upper = heat[weight_column].quantile(0.95)
        heat["peso"] = heat[weight_column].clip(upper=upper)
        max_weight = max(heat["peso"].max(), 1)
        heat["peso"] = heat["peso"] / max_weight

    HeatMap(
        heat[["lat", "lng", "peso"]].values.tolist(),
        name="Intensidad",
        radius=15,
        blur=18,
        min_opacity=0.25,
        max_zoom=13,
    ).add_to(mapa)

    return add_layer_control(mapa)


def choropleth_map(df: pd.DataFrame, geojson: dict, metric_column: str) -> folium.Map:
    mapa = build_base_map(df)
    geo = copy.deepcopy(geojson)
    summary = commune_summary(df)
    summary_by_key = {
        normalize_text(row["comuna"]): row for _, row in summary.iterrows()
    }

    labels = {
        "venta_neta": "Venta neta total",
        "ordenes": "Ordenes",
        "unidades": "Unidades",
        "ticket_promedio": "Ticket promedio",
        "kms_promedio": "Distancia promedio",
    }

    for feature in geo["features"]:
        name = feature["properties"]["name"]
        row = summary_by_key.get(normalize_text(name))
        if row is None:
            values = {
                "venta_neta": 0,
                "ordenes": 0,
                "unidades": 0,
                "ticket_promedio": 0,
                "kms_promedio": 0,
            }
        else:
            values = {
                "venta_neta": row["venta_neta"],
                "ordenes": row["ordenes"],
                "unidades": row["unidades"],
                "ticket_promedio": row["ticket_promedio"],
                "kms_promedio": row["kms_promedio"],
            }
        feature["properties"].update(
            {
                "venta_label": money(values["venta_neta"]),
                "ordenes_label": f"{values['ordenes']:,.0f}".replace(",", "."),
                "unidades_label": f"{values['unidades']:,.0f}".replace(",", "."),
                "ticket_label": money(values["ticket_promedio"]),
                "kms_label": f"{values['kms_promedio']:.1f} km".replace(".", ","),
            }
        )

    choropleth_data = summary[["comuna", metric_column]].copy()
    folium.Choropleth(
        geo_data=geo,
        data=choropleth_data,
        columns=["comuna", metric_column],
        key_on="feature.properties.name",
        fill_color="YlOrRd",
        fill_opacity=0.78,
        line_opacity=0.35,
        nan_fill_color="#f5f5f5",
        legend_name=labels[metric_column],
        name=labels[metric_column],
    ).add_to(mapa)

    folium.GeoJson(
        geo,
        name="Detalle por comuna",
        style_function=lambda _: {
            "fillOpacity": 0,
            "color": "#333333",
            "weight": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "name",
                "venta_label",
                "ordenes_label",
                "unidades_label",
                "ticket_label",
                "kms_label",
            ],
            aliases=[
                "Comuna:",
                "Venta:",
                "Ordenes:",
                "Unidades:",
                "Ticket:",
                "Distancia:",
            ],
            localize=True,
        ),
    ).add_to(mapa)

    return add_layer_control(mapa)


def sales_by_channel_chart(df: pd.DataFrame) -> plt.Figure:
    data = (
        df.groupby("canal", as_index=False)
        .agg(venta_neta=("venta_neta", "sum"), ordenes=("orden", "count"))
        .sort_values("venta_neta", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    sns.barplot(data=data, x="canal", y="venta_neta", hue="canal", legend=False, ax=ax)
    ax.set_title("Venta neta por canal", fontsize=12, weight="bold")
    ax.set_xlabel("Canal")
    ax.set_ylabel("Venta neta")
    ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        labels = [compact_money(value) for value in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=8, padding=3)
    fig.tight_layout()
    return fig


def monthly_sales_chart(df: pd.DataFrame) -> plt.Figure:
    data = (
        df.groupby("mes", as_index=False)
        .agg(venta_neta=("venta_neta", "sum"), ordenes=("orden", "count"))
        .sort_values("mes")
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(data["mes"], data["venta_neta"], marker="o", linewidth=2.4, color="#2a9d8f")
    ax.set_title("Evolucion mensual de ventas", fontsize=12, weight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Venta neta")
    ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def top_communes_chart(df: pd.DataFrame) -> plt.Figure:
    data = commune_summary(df).head(12)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(data=data, y="comuna", x="venta_neta", color="#e76f51", ax=ax)
    ax.set_title("Top 12 comunas por venta neta", fontsize=12, weight="bold")
    ax.set_xlabel("Venta neta")
    ax.set_ylabel("Comuna")
    ax.xaxis.set_major_formatter(FuncFormatter(money_axis))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def synthesis_chart(df: pd.DataFrame) -> plt.Figure:
    data = commune_summary(df)
    sale_range = data["venta_neta"].max() - data["venta_neta"].min()
    if sale_range == 0:
        sizes = [420] * len(data)
    else:
        sizes = 90 + 1250 * (data["venta_neta"] - data["venta_neta"].min()) / sale_range

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    scatter = ax.scatter(
        data["ordenes"],
        data["ticket_promedio"],
        s=sizes,
        c=data["kms_promedio"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, row in data.head(6).iterrows():
        ax.annotate(
            row["comuna"],
            (row["ordenes"], row["ticket_promedio"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Sintesis territorial por comuna", fontsize=12, weight="bold")
    ax.set_xlabel("Ordenes")
    ax.set_ylabel("Ticket promedio")
    ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Km promedio")
    fig.tight_layout()
    return fig


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    min_date = df["fecha_compra"].min().date()
    max_date = df["fecha_compra"].max().date()
    date_value = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_value, tuple) and len(date_value) == 2:
        start_date, end_date = date_value
    else:
        start_date = end_date = date_value

    channels = sorted(df["canal"].dropna().unique())
    selected_channels = st.sidebar.multiselect("Canal", channels, default=channels)

    centers = sorted(df["centro_dist"].dropna().unique())
    selected_centers = st.sidebar.multiselect(
        "Centro de distribucion", centers, default=centers
    )

    communes = sorted(df["comuna"].dropna().unique())
    selected_communes = st.sidebar.multiselect("Comuna", communes, default=communes)

    filtered = df[
        (df["fecha_compra"].dt.date >= start_date)
        & (df["fecha_compra"].dt.date <= end_date)
        & (df["canal"].isin(selected_channels))
        & (df["centro_dist"].isin(selected_centers))
        & (df["comuna"].isin(selected_communes))
    ].copy()

    return filtered


def render_findings(df: pd.DataFrame) -> None:
    communes = commune_summary(df)
    centers = center_summary(df)
    channel = (
        df.groupby("canal", as_index=False)["venta_neta"]
        .sum()
        .sort_values("venta_neta", ascending=False)
    )

    top_commune = communes.iloc[0]
    top_center = centers.iloc[0]
    top_channel = channel.iloc[0]
    channel_share = top_channel["venta_neta"] / df["venta_neta"].sum()

    st.markdown(
        f"""
        **Lectura ejecutiva.** La mayor venta filtrada se concentra en **{top_commune['comuna']}**
        con {money(top_commune['venta_neta'])}. El canal dominante es **{top_channel['canal']}**
        ({channel_share:.1%} de la venta), y el centro con mayor aporte es
        **{top_center['centro_dist']}**. Las comunas con mayor ticket y baja distancia promedio
        sugieren zonas prioritarias para retencion; las comunas lejanas con baja venta son candidatas
        para revisar cobertura, surtido o campanas territoriales.
        """
    )


def main() -> None:
    df = load_data()
    geojson = load_geojson()
    filtered = apply_filters(df)

    st.title("Análisis geoespacial de ventas")
    st.caption("Cadena de tiendas de comestibles | Región Metropolitana")

    if filtered.empty:
        st.warning("No hay registros para la combinacion de filtros seleccionada.")
        return

    kpi_cols = st.columns(6)
    for col, row in zip(kpi_cols, metric_table(filtered).itertuples(index=False)):
        col.metric(row.Indicador, row.Valor)

    st.divider()

    left, right = st.columns((1.05, 1))
    with left:
        st.subheader("Panorama general")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.pyplot(sales_by_channel_chart(filtered), width="stretch")
        with chart_cols[1]:
            st.pyplot(monthly_sales_chart(filtered), width="stretch")
        st.pyplot(top_communes_chart(filtered), width="stretch")

    with right:
        st.subheader("Mapa interactivo")
        map_type = st.radio(
            "Tipo de mapa",
            ["Red logistica", "Mapa de calor", "Coropleta comunal"],
            horizontal=True,
        )

        if map_type == "Red logistica":
            mapa = logistics_map(filtered)
        elif map_type == "Mapa de calor":
            heat_weight_label = st.selectbox(
                "Ponderacion",
                ["Venta neta", "Unidades", "Ordenes"],
            )
            heat_weight = {
                "Venta neta": "venta_neta",
                "Unidades": "unidades",
                "Ordenes": "ordenes",
            }[heat_weight_label]
            mapa = heat_map(filtered.assign(ordenes=1), heat_weight)
        else:
            metric_label = st.selectbox(
                "Metrica de color",
                [
                    "Venta neta",
                    "Ordenes",
                    "Unidades",
                    "Ticket promedio",
                    "Distancia promedio",
                ],
            )
            metric_column = {
                "Venta neta": "venta_neta",
                "Ordenes": "ordenes",
                "Unidades": "unidades",
                "Ticket promedio": "ticket_promedio",
                "Distancia promedio": "kms_promedio",
            }[metric_label]
            mapa = choropleth_map(filtered, geojson, metric_column)

        st_folium(mapa, height=650, use_container_width=True, returned_objects=[])

    st.divider()

    lower_left, lower_right = st.columns((1, 1))
    with lower_left:
        st.subheader("Sintesis territorial")
        st.pyplot(synthesis_chart(filtered), width="stretch")
    with lower_right:
        st.subheader("Hallazgos")
        render_findings(filtered)
        st.dataframe(
            commune_summary(filtered)
            .head(12)
            .assign(
                venta_neta=lambda x: x["venta_neta"].map(money),
                ticket_promedio=lambda x: x["ticket_promedio"].map(money),
                kms_promedio=lambda x: x["kms_promedio"].map(
                    lambda value: f"{value:.1f} km".replace(".", ",")
                ),
            )[
                [
                    "comuna",
                    "ordenes",
                    "venta_neta",
                    "unidades",
                    "ticket_promedio",
                    "kms_promedio",
                ]
            ],
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
