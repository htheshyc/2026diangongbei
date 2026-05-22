#!/usr/bin/env python3
"""Solve Question 1 of B problem and generate paper-ready artifacts."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
Q1_DIR = ROOT / "问题1"
SRC_DIR = Q1_DIR / "src"
IMG_DIR = Q1_DIR / "img"
DOC_DIR = Q1_DIR / "docs"
TABLE_DIR = DOC_DIR / "tables"
DATA_DIR = ROOT / "2026年电工杯竞赛赛题" / "2026年电工杯竞赛赛题" / "B题"

ATTACHMENT_1 = DATA_DIR / "附件1：小区基础数据.xlsx"
ATTACHMENT_2 = DATA_DIR / "附件2：服务需求数据.xlsx"

COMMUNITY_COL = "小区编号"
TYPE_LABELS = ["自理", "半失能", "失能"]
TYPE_COLS = {
    "自理": "自理老人",
    "半失能": "半失能老人",
    "失能": "失能老人",
}
SERVICE_ORDER = ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"]

DEATH_RATE = 0.05
NEW_ELDER_RATE = 0.07
DAYS_PER_MONTH = 30


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_number(value: object) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text.startswith("0"):
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else math.nan


def read_inputs() -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, float]]:
    population = pd.read_excel(ATTACHMENT_1, sheet_name="人口与老人结构", header=1)
    population = clean_columns(population).dropna(subset=[COMMUNITY_COL])
    population[COMMUNITY_COL] = population[COMMUNITY_COL].astype(str).str.strip()

    transitions = pd.read_excel(ATTACHMENT_1, sheet_name="转移概率", header=1)
    transitions = clean_columns(transitions).dropna(how="all")
    transitions["转移类型"] = transitions["转移类型"].astype(str).str.strip()
    transition_probs = {}
    for _, row in transitions.iterrows():
        key = row["转移类型"]
        prob = to_number(row["年度转移概率参考区间"])
        transition_probs[key] = prob

    demand = pd.read_excel(ATTACHMENT_2, sheet_name="每位老人月均服务需求次数", header=1)
    demand = clean_columns(demand).dropna(subset=["服务项目"])
    demand["服务项目"] = demand["服务项目"].astype(str).str.strip()
    demand = demand.rename(columns={"自理": "自理", "半自理": "半失能", "失能": "失能"})
    for col in TYPE_LABELS:
        demand[col] = demand[col].map(to_number)
    demand = demand.set_index("服务项目").loc[SERVICE_ORDER].reset_index()

    revenue = pd.read_excel(ATTACHMENT_2, sheet_name="服务营收及支出", header=1)
    revenue = clean_columns(revenue).dropna(subset=["服务项目"])
    revenue["服务项目"] = revenue["服务项目"].astype(str).str.strip()
    revenue["基准价格"] = revenue["单次服务营收（元）"].map(to_number)
    revenue["直接支出"] = revenue["单次服务直接支出（元）（基准价格）"].map(to_number)
    revenue = revenue[["服务项目", "基准价格", "直接支出"]].set_index("服务项目").loc[SERVICE_ORDER].reset_index()

    cap_ratio = {
        "自理": 0.20,
        "半失能": 0.25,
        "失能": 0.30,
    }
    return population, transition_probs, demand, revenue, cap_ratio


def largest_remainder_round(values: np.ndarray, target_total: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target_total is None:
        target_total = int(round(values.sum()))
    floors = np.floor(values).astype(int)
    remainder_count = int(target_total - floors.sum())
    if remainder_count > 0:
        order = np.argsort(-(values - floors))
        floors[order[:remainder_count]] += 1
    elif remainder_count < 0:
        order = np.argsort(values - floors)
        for idx in order[: abs(remainder_count)]:
            if floors[idx] > 0:
                floors[idx] -= 1
    return floors


def forecast_population(population: pd.DataFrame, transition_probs: dict[str, float]) -> pd.DataFrame:
    rho_ah = transition_probs["自理 → 半失能"]
    rho_hd = transition_probs["半失能 → 失能"]

    records: list[dict[str, object]] = []
    for _, row in population.iterrows():
        community = row[COMMUNITY_COL]
        state = np.array([row[TYPE_COLS[t]] for t in TYPE_LABELS], dtype=float)
        for year in range(0, 6):
            records.append(
                {
                    "年份": year,
                    "小区": community,
                    "自理": int(state[0]),
                    "半失能": int(state[1]),
                    "失能": int(state[2]),
                    "老人总数": int(state.sum()),
                }
            )
            if year == 5:
                break
            total = state.sum()
            raw_next = np.array(
                [
                    (1 - DEATH_RATE) * (1 - rho_ah) * state[0] + NEW_ELDER_RATE * total,
                    (1 - DEATH_RATE) * rho_ah * state[0] + (1 - DEATH_RATE) * (1 - rho_hd) * state[1],
                    (1 - DEATH_RATE) * rho_hd * state[1] + (1 - DEATH_RATE) * state[2],
                ],
                dtype=float,
            )
            state = largest_remainder_round(raw_next)

    return pd.DataFrame(records)


def build_demand_tables(
    forecast: pd.DataFrame,
    population: pd.DataFrame,
    demand: pd.DataFrame,
    revenue: pd.DataFrame,
    cap_ratio: dict[str, float],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    year5 = forecast[forecast["年份"] == 5].copy()
    q = demand.set_index("服务项目")[TYPE_LABELS]
    prices = revenue.set_index("服务项目")["基准价格"]
    income = population.set_index(COMMUNITY_COL)["人均月收入(元)"]

    theoretical_records: list[dict[str, object]] = []
    constrained_records: list[dict[str, object]] = []
    scaling_records: list[dict[str, object]] = []

    for _, row in year5.iterrows():
        community = row["小区"]
        for elder_type in TYPE_LABELS:
            count = int(row[elder_type])
            per_capita = q[elder_type]
            theoretical_cost = float((per_capita * prices).sum())
            cap = float(income.loc[community] * cap_ratio[elder_type])
            scale = 1.0 if theoretical_cost <= 0 else min(1.0, cap / theoretical_cost)

            scaling_records.append(
                {
                    "小区": community,
                    "老人类型": elder_type,
                    "人均月收入": float(income.loc[community]),
                    "理论月费用": theoretical_cost,
                    "消费上限": cap,
                    "削减系数": scale,
                    "预算是否约束": "是" if scale < 0.999999 else "否",
                }
            )

            for service in SERVICE_ORDER:
                q0 = float(per_capita.loc[service])
                q_adj = q0 * scale
                theoretical_records.append(
                    {
                        "小区": community,
                        "老人类型": elder_type,
                        "第5年人数": count,
                        "服务项目": service,
                        "理论人均月需求": q0,
                        "理论月需求总次数": int(round(count * q0)),
                    }
                )
                constrained_records.append(
                    {
                        "小区": community,
                        "老人类型": elder_type,
                        "第5年人数": count,
                        "服务项目": service,
                        "消费约束后人均月需求": q_adj,
                        "消费约束后月需求总次数": int(round(count * q_adj)),
                    }
                )

    theoretical_detail = pd.DataFrame(theoretical_records)
    constrained_detail = pd.DataFrame(constrained_records)
    scaling = pd.DataFrame(scaling_records)

    theoretical_by_community = (
        theoretical_detail.pivot_table(
            index="小区",
            columns="服务项目",
            values="理论月需求总次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=SERVICE_ORDER)
        .reset_index()
    )
    theoretical_by_type = (
        theoretical_detail.pivot_table(
            index=["服务项目"],
            columns="老人类型",
            values="理论月需求总次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=SERVICE_ORDER, columns=TYPE_LABELS)
        .reset_index()
    )
    theoretical_by_type["合计"] = theoretical_by_type[TYPE_LABELS].sum(axis=1)

    constrained_by_community = (
        constrained_detail.pivot_table(
            index="小区",
            columns="服务项目",
            values="消费约束后月需求总次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=SERVICE_ORDER)
        .reset_index()
    )
    constrained_by_type = (
        constrained_detail.pivot_table(
            index=["服务项目"],
            columns="老人类型",
            values="消费约束后月需求总次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=SERVICE_ORDER, columns=TYPE_LABELS)
        .reset_index()
    )
    constrained_by_type["合计"] = constrained_by_type[TYPE_LABELS].sum(axis=1)

    per_capita_constrained = (
        constrained_detail.pivot_table(
            index=["小区", "老人类型"],
            columns="服务项目",
            values="消费约束后人均月需求",
            aggfunc="first",
            fill_value=0,
        )
        .reindex(columns=SERVICE_ORDER)
        .reset_index()
    )

    theoretical_pivot_detail = (
        theoretical_detail.pivot_table(
            index=["小区", "老人类型"],
            columns="服务项目",
            values="理论月需求总次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=SERVICE_ORDER)
        .reset_index()
    )
    constrained_pivot_detail = (
        constrained_detail.pivot_table(
            index=["小区", "老人类型"],
            columns="服务项目",
            values="消费约束后月需求总次数",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=SERVICE_ORDER)
        .reset_index()
    )

    return (
        theoretical_by_community,
        theoretical_by_type,
        theoretical_pivot_detail,
        constrained_by_community,
        constrained_by_type,
        constrained_pivot_detail,
        per_capita_constrained,
        scaling,
    )


def choose_font() -> None:
    preferred = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    for font in preferred:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def plot_outputs(
    forecast: pd.DataFrame,
    theoretical_by_type: pd.DataFrame,
    constrained_by_type: pd.DataFrame,
    scaling: pd.DataFrame,
) -> None:
    choose_font()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    palette = {
        "自理": "#4C78A8",
        "半失能": "#F58518",
        "失能": "#E45756",
    }

    totals_by_year = forecast.groupby("年份")[TYPE_LABELS + ["老人总数"]].sum().reset_index()
    years = totals_by_year["年份"].to_numpy()
    shares = totals_by_year[TYPE_LABELS].div(totals_by_year["老人总数"], axis=0) * 100

    fig, (ax_total, ax_share) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.26},
    )
    ax_total.plot(
        years,
        totals_by_year["老人总数"],
        color="#172033",
        linewidth=2.4,
        marker="o",
        markersize=5.8,
    )
    ax_total.fill_between(years, totals_by_year["老人总数"], color="#4C78A8", alpha=0.13)
    for year, total in zip(years, totals_by_year["老人总数"]):
        ax_total.annotate(
            f"{int(total)}",
            xy=(year, total),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#374151",
        )
    y_min = totals_by_year["老人总数"].min() - 180
    y_max = totals_by_year["老人总数"].max() + 220
    ax_total.set_ylim(y_min, y_max)
    ax_total.set_ylabel("老人总数")
    ax_total.set_title("未来五年老人总量与类型占比预测", fontsize=15, fontweight="bold", pad=12)
    ax_total.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax_total.tick_params(axis="x", labelbottom=False)
    ax_total.spines[["top", "right"]].set_visible(False)

    for elder_type in TYPE_LABELS:
        ax_share.plot(
            years,
            shares[elder_type],
            color=palette[elder_type],
            linewidth=2.2,
            marker="o",
            markersize=5,
        )
        ax_share.annotate(
            f"{elder_type} {shares[elder_type].iloc[-1]:.1f}%",
            xy=(years[-1], shares[elder_type].iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=palette[elder_type],
        )
    ax_share.set_ylim(0, 75)
    ax_share.set_yticks([0, 20, 40, 60])
    ax_share.set_xlim(years.min() - 0.15, years.max() + 0.7)
    ax_share.set_xticks(years)
    ax_share.set_xlabel("年份")
    ax_share.set_ylabel("占比（%）")
    ax_share.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax_share.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.92)
    fig.savefig(IMG_DIR / "q1_elderly_structure_trend.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    year5 = forecast[forecast["年份"] == 5].copy()
    year5 = year5.sort_values("老人总数", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    left = np.zeros(len(year5))
    y_pos = np.arange(len(year5))
    for elder_type in TYPE_LABELS:
        vals = year5[elder_type].to_numpy()
        ax.barh(y_pos, vals, left=left, color=palette[elder_type], label=elder_type, height=0.68)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(year5["小区"])
    ax.set_xlabel("第5年末老人数量")
    ax.set_ylabel("小区")
    ax.set_title("第5年末各小区老人类型结构", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    ax.legend(ncol=3, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q1_year5_community_structure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    theoretical_total = theoretical_by_type.set_index("服务项目")["合计"].reindex(SERVICE_ORDER)
    constrained_total = constrained_by_type.set_index("服务项目")["合计"].reindex(SERVICE_ORDER)
    x = np.arange(len(SERVICE_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.bar(x - width / 2, theoretical_total, width=width, color="#4C78A8", label="理论需求")
    ax.bar(x + width / 2, constrained_total, width=width, color="#54A24B", label="消费约束后需求")
    ax.set_xticks(x)
    ax.set_xticklabels(SERVICE_ORDER, rotation=20, ha="right")
    ax.set_ylabel("月需求总次数")
    ax.set_title("第5年末各服务月需求：理论值与消费约束后对比", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q1_service_demand_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    heat = scaling.pivot(index="小区", columns="老人类型", values="削减系数").reindex(columns=TYPE_LABELS)
    cmap = LinearSegmentedColormap.from_list(
        "constraint_scale",
        ["#B91C1C", "#F59E0B", "#FDE68A", "#3F8F5F"],
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    values = heat.to_numpy()
    mesh = ax.pcolormesh(
        np.arange(values.shape[1] + 1),
        np.arange(values.shape[0] + 1),
        values,
        cmap=cmap,
        vmin=0.65,
        vmax=1.0,
        edgecolors="white",
        linewidth=1.4,
    )
    ax.invert_yaxis()
    ax.set_xticks(np.arange(values.shape[1]) + 0.5)
    ax.set_xticklabels(TYPE_LABELS)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5)
    ax.set_yticklabels(heat.index)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, labelsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val < 0.78 else "#111827"
            ax.text(j + 0.5, i + 0.5, f"{val:.0%}", ha="center", va="center", color=color, fontsize=9.5)
    ax.set_title("消费约束削减系数", fontsize=14, fontweight="bold", pad=18)
    fig.text(
        0.10,
        0.03,
        "注：数值越低表示需求被压缩越明显；100%表示不受消费上限约束。",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#4B5563",
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, shrink=0.82)
    cbar.set_label("保留比例")
    cbar.set_ticks([0.65, 0.75, 0.85, 0.95, 1.0])
    cbar.set_ticklabels(["65%", "75%", "85%", "95%", "100%"])
    ax.spines[:].set_visible(False)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.12, right=0.88)
    fig.savefig(IMG_DIR / "q1_consumption_scaling_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_number(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if abs(float(value) - round(float(value))) < 1e-9:
            return f"{int(round(float(value)))}"
        return f"{float(value):.2f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    headers = [str(c) for c in data.columns]
    rows = [[format_number(v) for v in row] for row in data.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def export_tables(
    forecast: pd.DataFrame,
    theoretical_by_community: pd.DataFrame,
    theoretical_by_type: pd.DataFrame,
    theoretical_pivot_detail: pd.DataFrame,
    constrained_by_community: pd.DataFrame,
    constrained_by_type: pd.DataFrame,
    constrained_pivot_detail: pd.DataFrame,
    per_capita_constrained: pd.DataFrame,
    scaling: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "q1_population_forecast.csv": forecast,
        "q1_theoretical_demand_by_community.csv": theoretical_by_community,
        "q1_theoretical_demand_by_type.csv": theoretical_by_type,
        "q1_theoretical_demand_detail.csv": theoretical_pivot_detail,
        "q1_constrained_demand_by_community.csv": constrained_by_community,
        "q1_constrained_demand_by_type.csv": constrained_by_type,
        "q1_constrained_demand_detail.csv": constrained_pivot_detail,
        "q1_constrained_per_capita.csv": per_capita_constrained,
        "q1_consumption_scaling.csv": scaling,
    }
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")


def build_paper(
    forecast: pd.DataFrame,
    theoretical_by_community: pd.DataFrame,
    theoretical_by_type: pd.DataFrame,
    theoretical_pivot_detail: pd.DataFrame,
    constrained_by_community: pd.DataFrame,
    constrained_by_type: pd.DataFrame,
    per_capita_constrained: pd.DataFrame,
    scaling: pd.DataFrame,
) -> str:
    totals_by_year = forecast.groupby("年份")[TYPE_LABELS + ["老人总数"]].sum().reset_index()
    year5 = forecast[forecast["年份"] == 5][["小区", "自理", "半失能", "失能", "老人总数"]].copy()
    scaling_short = scaling.copy()
    scaling_short["理论月费用"] = scaling_short["理论月费用"].round(2)
    scaling_short["消费上限"] = scaling_short["消费上限"].round(2)
    scaling_short["削减系数"] = scaling_short["削减系数"].round(4)
    per_capita_show = per_capita_constrained.copy()
    for service in SERVICE_ORDER:
        per_capita_show[service] = per_capita_show[service].round(2)

    type_detail_sample = theoretical_pivot_detail.copy()
    constrained_community = constrained_by_community.copy()
    start_row = totals_by_year[totals_by_year["年份"] == 0].iloc[0]
    end_row = totals_by_year[totals_by_year["年份"] == 5].iloc[0]
    start_total = int(start_row["老人总数"])
    end_total = int(end_row["老人总数"])
    end_self = int(end_row["自理"])
    end_half = int(end_row["半失能"])
    end_disabled = int(end_row["失能"])

    paper = rf"""# 第一问：未来五年老人数量与服务需求量预测

## 摘要

针对 B 题第一问，本文建立三状态老人数量递推模型，将老人状态划分为自理、半失能、失能三类，并结合自然死亡率、新增老年人口比例与状态转移概率，预测未来五年各小区各类老人数量。在此基础上，利用服务需求矩阵计算第5年末六类养老服务的理论月需求，再引入人均收入与月消费上限约束，对服务需求进行等比例削减，得到第5年末各小区各类型老人的可支付月均服务需求。所有结果由 `src/solve_question1.py` 自动生成，图表输出在 `img/`，附表输出在 `docs/tables/`。

## 1 数据来源与符号约定

第一问主要使用附件1和附件2。

| 数据表 | 主要字段 | 模型符号 | 用途 |
|---|---|---|---|
| 附件1-人口与老人结构 | 小区编号、总人口、60+老人数、自理老人、半失能老人、失能老人、人均月收入 | \(N_{{i,0}}^a,N_{{i,0}}^h,N_{{i,0}}^d,M_i\) | 初始人口状态与消费约束 |
| 附件1-转移概率 | 自理到半失能、半失能到失能 | \(\rho_{{ah}},\rho_{{hd}}\) | 状态转移参数 |
| 附件2-每位老人月均服务需求次数 | 服务项目、自理、半自理、失能 | \(q_{{s,t}}^0\) | 理论月需求频次 |
| 附件2-服务营收及支出 | 单次服务营收、单次服务直接支出 | \(p_s^0,c_s\) | 理论服务费用与后续成本口径 |
| 附件2-月服务消费上限 | 老人类型、收入占比 | \(\alpha_t\) | 消费能力约束 |

设小区集合为
\[
I=\\{{A,B,C,D,E,F,G,H,I,J\\}},
\]
老人类型集合为
\[
T=\\{{a,h,d\\}},
\]
其中 \(a,h,d\) 分别表示自理、半失能、失能。服务集合为
\[
S=\\{{\\text{{助餐}},\\text{{日间照料}},\\text{{上门护理}},\\text{{康复理疗}},\\text{{助浴}},\\text{{紧急救助}}\\}}.
\]

基准参数取值为：自然死亡率 \(\mu=5\\%\)，新增老人比例 \(\eta=7\\%\)，自理转半失能概率 \(\rho_{{ah}}=0.045\)，半失能转失能概率 \(\rho_{{hd}}=0.10\)。

## 2 问题1.1：老人数量递推预测模型

### 2.1 模型假设

1. 新增刚满60岁老人默认进入自理老人状态；
2. 自理老人可能转为半失能，半失能老人可能转为失能，失能老人不恢复；
3. 自然死亡率对三类老人均适用；
4. 五年内人口结构、收入水平、服务需求频次等参数保持稳定；
5. 每年末人数采用最大余数法整数化，以保持各类型人数之和与总人数一致。

### 2.2 递推模型

记小区 \(i\) 在第 \(y\) 年末的老人状态向量为
\[
\\boldsymbol{{N}}_{{i,y}}=
\\begin{{bmatrix}}
N_{{i,y}}^a\\\\
N_{{i,y}}^h\\\\
N_{{i,y}}^d
\\end{{bmatrix}},
\qquad
N_{{i,y}}=N_{{i,y}}^a+N_{{i,y}}^h+N_{{i,y}}^d.
\]

逐年递推为
\[
\\begin{{aligned}}
\\widetilde N_{{i,y+1}}^a
&=(1-\mu)(1-\rho_{{ah}})N_{{i,y}}^a+\eta N_{{i,y}},\\\\
\\widetilde N_{{i,y+1}}^h
&=(1-\mu)\rho_{{ah}}N_{{i,y}}^a+(1-\mu)(1-\rho_{{hd}})N_{{i,y}}^h,\\\\
\\widetilde N_{{i,y+1}}^d
&=(1-\mu)\rho_{{hd}}N_{{i,y}}^h+(1-\mu)N_{{i,y}}^d.
\\end{{aligned}}
\]

写成矩阵形式：
\[
\\boldsymbol{{N}}_{{i,y+1}}
=
\\boldsymbol{{A}}\\boldsymbol{{N}}_{{i,y}}
+\eta N_{{i,y}}\\boldsymbol{{e}}_a,
\qquad
\\boldsymbol{{e}}_a=(1,0,0)^\\top,
\]
其中
\[
\\boldsymbol{{A}}=(1-\mu)
\\begin{{bmatrix}}
1-\rho_{{ah}} & 0 & 0\\\\
\rho_{{ah}} & 1-\rho_{{hd}} & 0\\\\
0 & \rho_{{hd}} & 1
\\end{{bmatrix}}.
\]

注意，健康状态转移只改变老人类型，不改变老人总量。因此对三类老人求和可得
\[
N_{{i,y+1}}\approx(1-\mu)N_{{i,y}}+\eta N_{{i,y}}
=(1+\eta-\mu)N_{{i,y}}=1.02N_{{i,y}}.
\]
也就是说，在本题基准参数下老人总数是年增长率约 \(2\%\) 的几何增长；由于预测期只有5年且增长率较小，折线图在视觉上会接近线性，但数学上并非线性递推。

### 2.3 预测结果

全街道老人总量和结构的五年预测如下。

{markdown_table(totals_by_year)}

第5年末各小区老人数量预测如下。

{markdown_table(year5)}

![未来五年街道老人数量及结构预测](../img/q1_elderly_structure_trend.png)

![第5年末各小区老人类型结构](../img/q1_year5_community_structure.png)

## 3 问题1.2：第5年末理论月服务需求预测

### 3.1 模型表达

设 \(q_{{s,t}}^0\) 为类型 \(t\) 老人对服务 \(s\) 的理论月均需求次数。第5年末，小区 \(i\)、类型 \(t\)、服务 \(s\) 的理论月需求为
\[
Q_{{i,s,t,5}}^0=N_{{i,5}}^t q_{{s,t}}^0.
\]
小区 \(i\) 对服务 \(s\) 的理论月需求总量为
\[
Q_{{i,s,5}}^0=\sum_{{t\in T}}Q_{{i,s,t,5}}^0.
\]

### 3.2 计算结果

按服务项目和老人类型汇总的全街道理论月需求如下。

{markdown_table(theoretical_by_type)}

按小区和服务项目汇总的第5年末理论月需求如下。

{markdown_table(theoretical_by_community)}

按“小区-老人类型-服务项目”展开的明细表较长，已完整导出为 `docs/tables/q1_theoretical_demand_detail.csv`。下表给出明细前若干行用于核验口径。

{markdown_table(type_detail_sample, max_rows=12)}

## 4 问题1.3：消费约束下的月均服务需求

### 4.1 消费约束模型

设小区 \(i\) 人均月收入为 \(M_i\)，类型 \(t\) 老人的月服务消费上限比例为 \(\alpha_t\)。理论月服务费用为
\[
E_{{i,t}}^0=\sum_{{s\in S}}p_s^0q_{{s,t}}^0,
\]
月消费上限为
\[
L_{{i,t}}=\alpha_tM_i.
\]
若 \(E_{{i,t}}^0>L_{{i,t}}\)，则按附件说明对各服务次数等比例削减。定义削减系数
\[
\lambda_{{i,t}}=\min\\left\\{{1,\\frac{{L_{{i,t}}}}{{E_{{i,t}}^0}}\\right\\}}.
\]
于是消费约束后的单个老人月均服务需求为
\[
\\bar q_{{i,s,t}}=\lambda_{{i,t}}q_{{s,t}}^0,
\]
对应小区总需求为
\[
Q_{{i,s,t,5}}=N_{{i,5}}^t\\bar q_{{i,s,t}}.
\]

由于原始需求表中紧急救助为 0.15 次/月等小数频次，本文在人均需求表中保留两位小数；在小区月需求总次数表中进行四舍五入取整。

### 4.2 消费约束强度

各小区、各类型老人的消费削减系数如下。系数为1表示消费能力不构成约束，系数小于1表示需要同比例压缩服务次数。

{markdown_table(scaling_short[["小区", "老人类型", "人均月收入", "理论月费用", "消费上限", "削减系数", "预算是否约束"]])}

![消费约束削减系数热力图](../img/q1_consumption_scaling_heatmap.png)

### 4.3 消费约束后的需求结果

按服务项目和老人类型汇总的全街道消费约束后月需求如下。

{markdown_table(constrained_by_type)}

按小区和服务项目汇总的消费约束后月需求如下。

{markdown_table(constrained_community)}

![第5年末各服务月需求：理论值与消费约束后对比](../img/q1_service_demand_comparison.png)

第5年末每个小区、各类型老人的消费约束后人均月服务需求如下。

{markdown_table(per_capita_show)}

## 5 算法步骤与复杂度

第一问采用确定性递推和矩阵乘法计算，步骤如下：

1. 读取附件1中的小区老人初始结构、收入和状态转移概率；
2. 对每个小区建立三状态向量 \(\\boldsymbol{{N}}_{{i,y}}\)；
3. 按年度状态递推式计算 \(y=1,\dots,5\) 的老人数量，并用最大余数法整数化；
4. 读取附件2中的服务需求矩阵，计算第5年末理论需求 \(Q_{{i,s,t,5}}^0\)；
5. 根据收入和消费上限计算削减系数 \(\lambda_{{i,t}}\)，得到消费约束后的需求；
6. 导出结果表和学术图。

若小区数为 \(|I|\)，老人类型数为 \(|T|\)，服务项目数为 \(|S|\)，预测年数为 \(Y\)，则人口递推复杂度为
\[
O(|I|Y|T|^2),
\]
需求计算复杂度为
\[
O(|I||T||S|).
\]
本题中 \(|I|=10, |T|=3, |S|=6, Y=5\)，计算规模很小，算法可精确、稳定复现。

## 6 结论

在基准参数下，街道 60 岁以上老人总量从当前 {start_total} 人增长到第5年末 {end_total} 人，其中自理老人 {end_self} 人、半失能老人 {end_half} 人、失能老人 {end_disabled} 人。消费约束主要影响半失能和失能老人，尤其失能老人由于护理、日间照料、康复理疗和助浴需求较高，理论费用普遍超过收入上限，因此需要明显压缩需求。消费约束后，助餐和日间照料仍是总需求最高的两类服务；上门护理、康复理疗、助浴等服务的需求对收入约束更敏感，应在后续站点选址和补贴定价模型中重点考虑。
"""
    return paper.replace("\\\\", "\\")


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    population, transition_probs, demand, revenue, cap_ratio = read_inputs()
    forecast = forecast_population(population, transition_probs)
    (
        theoretical_by_community,
        theoretical_by_type,
        theoretical_pivot_detail,
        constrained_by_community,
        constrained_by_type,
        constrained_pivot_detail,
        per_capita_constrained,
        scaling,
    ) = build_demand_tables(forecast, population, demand, revenue, cap_ratio)

    export_tables(
        forecast,
        theoretical_by_community,
        theoretical_by_type,
        theoretical_pivot_detail,
        constrained_by_community,
        constrained_by_type,
        constrained_pivot_detail,
        per_capita_constrained,
        scaling,
    )
    plot_outputs(forecast, theoretical_by_type, constrained_by_type, scaling)
    paper = build_paper(
        forecast,
        theoretical_by_community,
        theoretical_by_type,
        theoretical_pivot_detail,
        constrained_by_community,
        constrained_by_type,
        per_capita_constrained,
        scaling,
    )
    (DOC_DIR / "第一问论文.md").write_text(paper, encoding="utf-8")
    print(f"Generated paper: {DOC_DIR / '第一问论文.md'}")
    print(f"Generated images: {IMG_DIR}")
    print(f"Generated tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
