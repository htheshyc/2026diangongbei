#!/usr/bin/env python3
"""Solve Question 4 sensitivity analysis and generate paper-ready artifacts."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
Q4_DIR = ROOT / "问题4"
SRC_DIR = Q4_DIR / "src"
IMG_DIR = Q4_DIR / "img"
DOC_DIR = Q4_DIR / "docs"
TABLE_DIR = DOC_DIR / "tables"
Q2_SRC = ROOT / "问题2" / "src" / "solve_question2.py"
Q3_SRC = ROOT / "问题3" / "src" / "solve_question3.py"

MPL_CACHE_BOOTSTRAP = Q4_DIR / ".matplotlib-cache"
MPL_CACHE_BOOTSTRAP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_BOOTSTRAP))
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q2 = import_module(Q2_SRC, "q2_for_q4")
q3 = import_module(Q3_SRC, "q3_for_q4")

COMMUNITIES = q2.COMMUNITIES
SERVICE_ORDER = q2.SERVICE_ORDER
PAID_SERVICES = [s for s in SERVICE_ORDER if s != "紧急救助"]


@dataclass(frozen=True)
class Scenario:
    code: str
    name: str
    eta: float = 0.07
    rho_ah: float = 0.045
    rho_hd: float = 0.10
    cost_factor: float = 1.0
    budget: float = 120.0
    note: str = ""


SCENARIOS = [
    Scenario("baseline", "基准情景", note="原始参数"),
    Scenario("growth8", "老人增长率8%", eta=0.08, note="新增老人比例由7%调至8%"),
    Scenario("transition", "转移概率调整", rho_ah=0.055, rho_hd=0.095, note="自理到半失能5.5%，半失能到失能9.5%"),
    Scenario("cost20", "运营成本增加20%", cost_factor=1.2, note="日固定管理成本增加20%"),
    Scenario("budget140", "预算140万元", budget=140.0, note="建设预算由120万元增至140万元"),
]


def choose_font() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC", "Arial Unicode MS"]
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


def export_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    if not rows and headers is None:
        return
    if headers is None:
        headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def apply_scenario_inputs(scenario: Scenario):
    population, transition, demand, price, direct_cost, station_costs, distance = q2.read_inputs()
    transition = dict(transition)
    transition["自理 → 半失能"] = scenario.rho_ah
    transition["半失能 → 失能"] = scenario.rho_hd
    station_costs = {
        scale: {
            "建设成本": values["建设成本"],
            "日固定成本": values["日固定成本"] * scenario.cost_factor,
            "容量": values["容量"],
        }
        for scale, values in station_costs.items()
    }
    return population, transition, demand, price, direct_cost, station_costs, distance


def station_info_from_plan(best) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    station_map = {station.community: station for station in best.stations}
    for station in best.stations:
        covered = [community for community, sid in best.assignment.items() if sid == station.community]
        info[station.community] = {
            "scale": station.scale,
            "covered": covered,
            "capacity": station.capacity,
            "distance_score": {community: best.distance_score[community] for community in covered},
        }
    return info


def optimize_pricing_for_plan(data: dict[str, Any], station_info: dict[str, dict[str, Any]]):
    return {station: q3.optimize_station(station, info, data) for station, info in station_info.items()}


def run_scenario(scenario: Scenario) -> dict[str, Any]:
    original_eta = q2.NEW_ELDER_RATE
    original_budget = q2.BUDGET_LIMIT
    try:
        q2.NEW_ELDER_RATE = scenario.eta
        q2.BUDGET_LIMIT = scenario.budget
        inputs = apply_scenario_inputs(scenario)
        q2_result = q2.solve_question2(inputs)
    finally:
        q2.NEW_ELDER_RATE = original_eta
        q2.BUDGET_LIMIT = original_budget

    population, transition, demand, price, direct_cost, station_costs, distance = inputs
    data = {
        "population": population,
        "transition": transition,
        "demand": demand,
        "price": price,
        "direct_cost": direct_cost,
        "station_costs": station_costs,
        "distance": distance,
        "year5": q2_result["year5"],
    }
    station_info = station_info_from_plan(q2_result["best"])
    pricing = optimize_pricing_for_plan(data, station_info)
    return {
        "scenario": scenario,
        "inputs": inputs,
        "q2": q2_result,
        "data": data,
        "station_info": station_info,
        "pricing": pricing,
    }


def scenario_metrics(result: dict[str, Any]) -> dict[str, Any]:
    scenario = result["scenario"]
    best = result["q2"]["best"]
    pricing = result["pricing"]
    year5 = result["q2"]["year5"]
    covered = set()
    for info in result["station_info"].values():
        covered.update(info["covered"])
    covered_elderly = sum(year5[c]["老人总数"] for c in covered)
    total_elderly = sum(year5[c]["老人总数"] for c in COMMUNITIES)
    q3_satisfaction = (
        sum(
            year5[c]["老人总数"] * station_result.community_satisfaction[c]
            for station_result in pricing.values()
            for c in station_result.covered
        )
        / covered_elderly
        if covered_elderly
        else 0.0
    )
    q3_price_score = (
        sum(
            year5[c]["老人总数"] * station_result.community_price_score[c]
            for station_result in pricing.values()
            for c in station_result.covered
        )
        / covered_elderly
        if covered_elderly
        else 0.0
    )
    avg_multiplier = (
        sum(r.weighted_price_multiplier for r in pricing.values()) / len(pricing) if pricing else 0.0
    )
    avg_profit_rate = sum(r.profit_rate for r in pricing.values()) / len(pricing) if pricing else 0.0
    station_set = ",".join(f"{station.community}-{station.scale}" for station in best.stations)
    return {
        "情景": scenario.name,
        "情景代码": scenario.code,
        "参数说明": scenario.note,
        "老人总数": total_elderly,
        "站点数量": best.station_count,
        "站点方案": station_set,
        "建设成本_万元": best.build_cost_total,
        "覆盖老人数量": covered_elderly,
        "覆盖率": covered_elderly / total_elderly if total_elderly else 0.0,
        "问题2满意度": best.avg_satisfaction_covered,
        "问题3满意度": q3_satisfaction,
        "价格满意度": q3_price_score,
        "平均价格倍率": avg_multiplier,
        "政府补贴_元": sum(r.subsidy for r in pricing.values()),
        "年度净利润_元": sum(r.net_profit for r in pricing.values()),
        "平均利润率": avg_profit_rate,
        "可行选址方案数": result["q2"]["feasible_count"],
    }


def station_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        scenario = result["scenario"]
        for station in result["q2"]["best"].stations:
            covered = [c for c, sid in result["q2"]["best"].assignment.items() if sid == station.community]
            pricing = result["pricing"][station.community]
            rows.append(
                {
                    "情景": scenario.name,
                    "站点": station.community,
                    "规模": station.scale,
                    "覆盖小区": "、".join(covered),
                    "日有效服务人次": pricing.effective_daily,
                    "利用率": pricing.theta,
                    "服务价格倍率": pricing.weighted_price_multiplier,
                    "政府补贴_元": pricing.subsidy,
                    "净利润_元": pricing.net_profit,
                    "利润率": pricing.profit_rate,
                }
            )
    return rows


def price_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        scenario = result["scenario"]
        for station, pricing in result["pricing"].items():
            row = {"情景": scenario.name, "站点": station, "规模": pricing.scale}
            for service in SERVICE_ORDER:
                row[f"{service}价格"] = pricing.prices[service]
            row["加权价格倍率"] = pricing.weighted_price_multiplier
            rows.append(row)
    return rows


def sensitivity_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = metrics[0]
    base_sites = set(base["站点方案"].split(","))
    rows = []
    for row in metrics[1:]:
        sites = set(row["站点方案"].split(","))
        union = base_sites | sites
        inter = base_sites & sites
        delta_sites = 1 - len(inter) / len(union) if union else 0.0
        delta_cov = (row["覆盖率"] - base["覆盖率"]) / base["覆盖率"]
        delta_sat = (row["问题3满意度"] - base["问题3满意度"]) / base["问题3满意度"]
        delta_subsidy = (row["政府补贴_元"] - base["政府补贴_元"]) / base["政府补贴_元"]
        delta_price = (row["平均价格倍率"] - base["平均价格倍率"]) / base["平均价格倍率"]
        si = abs(delta_cov) + abs(delta_sat) + abs(delta_subsidy) + delta_sites
        rows.append(
            {
                "情景": row["情景"],
                "站点集合变化": delta_sites,
                "覆盖率变化率": delta_cov,
                "满意度变化率": delta_sat,
                "补贴变化率": delta_subsidy,
                "价格倍率变化率": delta_price,
                "综合敏感性指数": si,
            }
        )
    rows.sort(key=lambda item: item["综合敏感性指数"], reverse=True)
    return rows


def community_satisfaction_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        scenario = result["scenario"]
        for community in COMMUNITIES:
            station_name = "未覆盖"
            satisfaction = 0.0
            for station, pricing in result["pricing"].items():
                if community in pricing.covered:
                    station_name = station
                    satisfaction = pricing.community_satisfaction[community]
                    break
            rows.append({"情景": scenario.name, "小区": community, "服务站": station_name, "满意度": satisfaction})
    return rows


def format_number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], headers: list[str] | None = None, digits: int = 4) -> str:
    if not rows:
        return ""
    if headers is None:
        headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_number(row.get(h, ""), digits) for h in headers) + " |")
    return "\n".join(lines)


def plot_outputs(metrics: list[dict[str, Any]], sensitivity: list[dict[str, Any]], community_rows_: list[dict[str, Any]]) -> None:
    choose_font()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["情景"] for row in metrics]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.bar(x - 0.18, [row["覆盖率"] for row in metrics], 0.36, label="覆盖率", color="#4C78A8")
    ax.bar(x + 0.18, [row["问题3满意度"] for row in metrics], 0.36, label="满意度", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("指标值")
    ax.set_title("情景覆盖率与满意度对比", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q4_coverage_satisfaction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10.2, 5.4))
    ax2 = ax1.twinx()
    ax1.bar(x, [row["政府补贴_元"] / 10000 for row in metrics], color="#4C78A8", width=0.55, label="政府补贴")
    ax2.plot(x, [row["平均价格倍率"] for row in metrics], color="#E45756", marker="o", linewidth=2.2, label="平均价格倍率")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("政府补贴（万元）")
    ax2.set_ylabel("平均价格倍率")
    ax1.set_title("情景补贴总额与价格倍率对比", fontsize=15, fontweight="bold", pad=12)
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q4_subsidy_price.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sens_sorted = sorted(sensitivity, key=lambda row: row["综合敏感性指数"])
    ax.barh([row["情景"] for row in sens_sorted], [row["综合敏感性指数"] for row in sens_sorted], color="#F58518")
    ax.set_xlabel("综合敏感性指数")
    ax.set_title("参数敏感性排序", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q4_sensitivity_index.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    heat = np.zeros((len(metrics), len(COMMUNITIES)))
    scenario_to_idx = {row["情景"]: idx for idx, row in enumerate(metrics)}
    community_to_idx = {c: idx for idx, c in enumerate(COMMUNITIES)}
    for row in community_rows_:
        heat[scenario_to_idx[row["情景"]], community_to_idx[row["小区"]]] = row["满意度"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    mesh = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(COMMUNITIES)))
    ax.set_xticklabels(COMMUNITIES)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(labels)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if heat[i, j] < 0.35 else "#111827"
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("各情景小区满意度热力图", fontsize=15, fontweight="bold", pad=12)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("满意度")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q4_community_satisfaction_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_paper(metrics: list[dict[str, Any]], sensitivity: list[dict[str, Any]], stations: list[dict[str, Any]], prices: list[dict[str, Any]]) -> str:
    metric_headers = [
        "情景",
        "老人总数",
        "站点数量",
        "站点方案",
        "建设成本_万元",
        "覆盖率",
        "问题3满意度",
        "平均价格倍率",
        "政府补贴_元",
        "年度净利润_元",
    ]
    sensitivity_headers = ["情景", "站点集合变化", "覆盖率变化率", "满意度变化率", "补贴变化率", "价格倍率变化率", "综合敏感性指数"]
    station_headers = ["情景", "站点", "规模", "覆盖小区", "服务价格倍率", "政府补贴_元", "净利润_元", "利润率"]

    top_sensitive = sensitivity[0]["情景"] if sensitivity else ""
    most_robust = sensitivity[-1]["情景"] if sensitivity else ""

    paper = rf"""# 第四问：灵敏度分析与方案比较

## 1 问题重述与分析框架

第四问要求分别改变老人增长率、健康状态转移概率、日固定管理成本和建设预算，并重新求解问题2和问题3，比较站点数量、位置、服务定价、政府补贴总额、覆盖率和满意度等指标。本文严格采用 `docs/问题分析.md` 中的“情景重求解 + 指标归一化 + 稳定性排序”路线，对每个情景完整执行：
\[
\text{{问题1预测}}\rightarrow\text{{问题2选址规模优化}}\rightarrow\text{{问题3定价补贴优化}}.
\]

情景参数设置如下。

| 情景 | 参数变化 |
| --- | --- |
| 基准情景 | \(\eta=0.07,\rho_{{ah}}=0.045,\rho_{{hd}}=0.10,B^{{\max}}=120\) 万元 |
| 老人增长率8% | 新增老人比例 \(\eta\) 由 0.07 调整为 0.08 |
| 转移概率调整 | \(\rho_{{ah}}=0.055,\rho_{{hd}}=0.095\) |
| 运营成本增加20% | 各规模站点日固定管理成本乘以 1.2 |
| 预算140万元 | 总建设预算由 120 万元提高到 140 万元 |

其中“老人增长率”按问题分析中的建议解释为新增老人比例 \(\eta\) 的调整，与第一问“年均新增老人占当前总老年人口比例”的口径保持一致。

## 2 比较指标

设基准情景为 \(0\)，其他情景为 \(r\)。站点集合变化定义为
\[
\Delta J_r=1-\frac{{|J_r\cap J_0|}}{{|J_r\cup J_0|}}.
\]
覆盖率、满意度、补贴和价格变化率分别为
\[
\delta\mathrm{{Cov}}_r=\frac{{\mathrm{{Cov}}_r-\mathrm{{Cov}}_0}}{{\mathrm{{Cov}}_0}},
\quad
\delta\overline S_r=\frac{{\overline S_r-\overline S_0}}{{\overline S_0}},
\]
\[
\delta H_r=\frac{{H_r-H_0}}{{H_0}},
\quad
\delta p_r=\frac{{\bar p_r-\bar p_0}}{{\bar p_0}}.
\]
本文取等权敏感性指数
\[
\mathrm{{SI}}_r=
|\delta\mathrm{{Cov}}_r|+
|\delta\overline S_r|+
|\delta H_r|+
\Delta J_r.
\]
该指标越大，说明方案对该参数越敏感。

## 3 情景重求解结果

各情景的核心结果如下。

{markdown_table(metrics, metric_headers, digits=4)}

![情景覆盖率与满意度对比](../img/q4_coverage_satisfaction.png)

![情景补贴总额与价格倍率对比](../img/q4_subsidy_price.png)

各情景站点层面的方案如下。

{markdown_table(stations, station_headers, digits=4)}

完整服务定价已导出至 `docs/tables/q4_price_comparison.csv`。下表展示各情景各站的加权价格倍率和主要价格信息。

{markdown_table(prices, ["情景", "站点", "规模", "助餐价格", "日间照料价格", "上门护理价格", "康复理疗价格", "助浴价格", "加权价格倍率"], digits=3)}

## 4 敏感性排序与鲁棒性评价

相对于基准情景，各参数变化率和综合敏感性指数如下。

{markdown_table(sensitivity, sensitivity_headers, digits=4)}

![参数敏感性排序](../img/q4_sensitivity_index.png)

各小区满意度热力图如下。

![各情景小区满意度热力图](../img/q4_community_satisfaction_heatmap.png)

从综合敏感性指数看，最敏感的情景为“{top_sensitive}”，最稳定的情景为“{most_robust}”。预算提高会直接改变可建设规模并显著提高覆盖率，因此敏感性最高；老人增长率提高会引发站点重构，但覆盖率基本保持稳定；转移概率的小幅变化影响有限。运营成本增加20%主要通过提高价格倍率来维持利润率，并未改变站点集合、覆盖率、满意度和补贴，因此在本文等权敏感性指数下表现最稳定。

## 5 实际推广中的其他不确定因素与应对策略

| 不确定因素 | 可能影响 | 应对策略 |
| --- | --- | --- |
| 健康状态转移具有随机波动 | 护理、助浴、康复等需求偏离五年预测 | 建立滚动预测机制，每年更新转移概率并设置需求安全裕度 |
| 服务人员供给不足 | 有站点容量但无法提供足够护理、康复、助浴服务 | 将容量细分为护理员、康复师、餐位、助浴设备等多资源约束 |
| 实际道路通行时间变化 | 距离满意度不能真实反映上门服务响应 | 使用道路网络时间距离替代小区间距离，并对上门服务建路径优化模型 |
| 老人收入与支付意愿变化 | 消费约束和价格满意度发生变化 | 建立分收入层需求模型，设置低收入老人专项补贴 |
| 补贴政策调整 | 利润率、价格倍率和财政支出变化 | 设计多补贴标准情景，并加入财政总支出上限约束 |
| 小区场地可用性差异 | 理论最优站点可能无法实际落地 | 增加场地面积、租金、改造周期和产权可行性约束 |

## 6 结论

本文对四类单因素变化进行了完整重求解。结果表明，当前基准方案在多数情景下仍能保持较高覆盖率和满意度；预算提高能够把覆盖率提升到100%，但也显著增加补贴支出。运营成本上升时站点布局保持不变，主要通过上调价格倍率消化成本压力。若政策目标是提高覆盖率，增加建设预算更直接有效；若政策目标是控制财政支出，则需要重点监测固定管理成本和补贴封顶标准。建议实际推广时采用滚动预测与年度复核机制，在老人结构、成本和预算发生明显变化时重新运行选址和定价模型。
"""
    return paper


def export_outputs(results: list[dict[str, Any]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [scenario_metrics(result) for result in results]
    sensitivity = sensitivity_rows(metrics)
    stations = station_rows(results)
    prices = price_rows(results)
    communities = community_satisfaction_rows(results)

    export_csv(TABLE_DIR / "q4_scenario_summary.csv", metrics)
    export_csv(TABLE_DIR / "q4_sensitivity.csv", sensitivity)
    export_csv(TABLE_DIR / "q4_station_comparison.csv", stations)
    export_csv(TABLE_DIR / "q4_price_comparison.csv", prices)
    export_csv(TABLE_DIR / "q4_community_satisfaction.csv", communities)

    plot_outputs(metrics, sensitivity, communities)
    (DOC_DIR / "第四问论文.md").write_text(build_paper(metrics, sensitivity, stations, prices), encoding="utf-8")


def main() -> None:
    results = []
    for scenario in SCENARIOS:
        print(f"Running scenario: {scenario.name}")
        result = run_scenario(scenario)
        metric = scenario_metrics(result)
        print(
            f"  sites={metric['站点方案']} cov={metric['覆盖率']:.4f} "
            f"sat={metric['问题3满意度']:.4f} subsidy={metric['政府补贴_元']:.2f}"
        )
        results.append(result)
    export_outputs(results)
    print(f"Generated paper: {DOC_DIR / '第四问论文.md'}")
    print(f"Generated tables: {TABLE_DIR}")
    print(f"Generated images: {IMG_DIR}")


if __name__ == "__main__":
    main()
