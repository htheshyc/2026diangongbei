#!/usr/bin/env python3
"""Solve Question 2 of B problem and generate paper-ready artifacts."""

from __future__ import annotations

import csv
import itertools
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

MPL_CACHE_BOOTSTRAP = Path.cwd() / "问题2" / ".matplotlib-cache"
MPL_CACHE_BOOTSTRAP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_BOOTSTRAP))

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
Q2_DIR = ROOT / "问题2"
SRC_DIR = Q2_DIR / "src"
IMG_DIR = Q2_DIR / "img"
DOC_DIR = Q2_DIR / "docs"
TABLE_DIR = DOC_DIR / "tables"
DATA_DIR = ROOT / "2026年电工杯竞赛赛题" / "2026年电工杯竞赛赛题" / "B题"
MPL_CACHE_DIR = Q2_DIR / ".matplotlib-cache"

ATTACHMENT_1 = DATA_DIR / "附件1：小区基础数据.xlsx"
ATTACHMENT_2 = DATA_DIR / "附件2：服务需求数据.xlsx"
ATTACHMENT_3 = DATA_DIR / "附件3：服务站建设与运营成本.xlsx"
ATTACHMENT_4 = DATA_DIR / "附件4：小区间距离矩阵.xlsx"

COMMUNITIES = list("ABCDEFGHIJ")
TYPE_LABELS = ["自理", "半失能", "失能"]
TYPE_COLS = {
    "自理": "自理老人",
    "半失能": "半失能老人",
    "失能": "失能老人",
}
SERVICE_ORDER = ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"]
SCALE_ORDER = ["小型", "中型", "大型"]

DEATH_RATE = 0.05
NEW_ELDER_RATE = 0.07
BUDGET_LIMIT = 120.0
SERVICE_RADIUS = 1000.0
DAYS_PER_MONTH = 30.0


@dataclass(frozen=True)
class Station:
    community: str
    scale: str
    build_cost: float
    fixed_daily_cost: float
    capacity: float

    @property
    def annual_fixed_cost(self) -> float:
        return 365.0 * self.fixed_daily_cost + 10000.0 * self.build_cost / 20.0


@dataclass
class PlanResult:
    state: tuple[int, ...]
    stations: list[Station]
    assignment: dict[str, str]
    satisfaction: dict[str, float]
    distance_score: dict[str, float]
    response_score: dict[str, float]
    theta: dict[str, float]
    theoretical_daily_load: dict[str, float]
    effective_daily_load: dict[str, float]
    revenue: dict[str, float]
    direct_cost: dict[str, float]
    fixed_cost: dict[str, float]
    profit: dict[str, float]
    coverage_rate: float
    covered_elderly: int
    avg_satisfaction_covered: float
    avg_satisfaction_all: float
    total_profit: float
    build_cost_total: float
    station_count: int


def xlsx_sheets(path: Path) -> dict[str, list[list[str]]]:
    """Read small xlsx sheets using only the standard library."""
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    def col_index(cell_ref: str) -> int:
        match = re.match(r"([A-Z]+)", cell_ref)
        if not match:
            return 1
        col = 0
        for ch in match.group(1):
            col = col * 26 + ord(ch) - 64
        return col

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        result: dict[str, list[list[str]]] = {}

        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            name = sheet.attrib["name"]
            rid = sheet.attrib[f"{{{ns['r']}}}id"]
            target = rid_to_target[rid]
            sheet_path = target[1:] if target.startswith("/") else f"xl/{target}"
            sheet_root = ET.fromstring(zf.read(sheet_path))
            rows: list[list[str]] = []
            for row in sheet_root.findall("m:sheetData/m:row", ns):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", ns):
                    col = col_index(cell.attrib.get("r", "A1"))
                    cell_type = cell.attrib.get("t")
                    value = ""
                    v = cell.find("m:v", ns)
                    if v is not None:
                        value = v.text or ""
                        if cell_type == "s":
                            value = shared_strings[int(value)]
                    inline = cell.find("m:is", ns)
                    if inline is not None:
                        value = "".join(t.text or "" for t in inline.findall(".//m:t", ns))
                    values[col] = value.strip()
                if values:
                    width = max(values)
                    rows.append([values.get(i, "") for i in range(1, width + 1)])
            result[name] = rows
        return result


def to_number(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", text)
    return float(match.group()) if match else math.nan


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def rows_to_dicts(rows: list[list[str]], header_row: int = 1) -> list[dict[str, str]]:
    header = [cell.strip() for cell in rows[header_row]]
    records = []
    for row in rows[header_row + 1 :]:
        if not any(str(cell).strip() for cell in row):
            continue
        padded = row + [""] * (len(header) - len(row))
        records.append({header[i]: padded[i].strip() for i in range(len(header))})
    return records


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


def read_inputs():
    population_rows = rows_to_dicts(xlsx_sheets(ATTACHMENT_1)["人口与老人结构"], header_row=1)
    transition_rows = rows_to_dicts(xlsx_sheets(ATTACHMENT_1)["转移概率"], header_row=1)
    demand_rows = rows_to_dicts(xlsx_sheets(ATTACHMENT_2)["每位老人月均服务需求次数"], header_row=1)
    revenue_rows = rows_to_dicts(xlsx_sheets(ATTACHMENT_2)["服务营收及支出"], header_row=1)
    station_rows = rows_to_dicts(xlsx_sheets(ATTACHMENT_3)["服务站建设与运营成本"], header_row=1)
    distance_rows = rows_to_dicts(xlsx_sheets(ATTACHMENT_4)["小区间距离矩阵"], header_row=1)

    population: dict[str, dict[str, float]] = {}
    for row in population_rows:
        community = row["小区编号"]
        if community not in COMMUNITIES:
            continue
        population[community] = {
            "总人口": to_number(row["总人口"]),
            "60+老人数": to_number(row["60+老人数"]),
            "自理老人": to_number(row["自理老人"]),
            "半失能老人": to_number(row["半失能老人"]),
            "失能老人": to_number(row["失能老人"]),
            "人均月收入": to_number(row["人均月收入(元)"]),
        }

    transition = {row["转移类型"]: to_number(row["年度转移概率参考区间"]) for row in transition_rows}

    demand: dict[str, dict[str, float]] = {}
    for row in demand_rows:
        service = row["服务项目"]
        if service not in SERVICE_ORDER:
            continue
        demand[service] = {
            "自理": to_number(row["自理"]),
            "半失能": to_number(row["半自理"]),
            "失能": to_number(row["失能"]),
        }

    price: dict[str, float] = {}
    direct_cost: dict[str, float] = {}
    for row in revenue_rows:
        service = row["服务项目"]
        if service not in SERVICE_ORDER:
            continue
        price[service] = to_number(row["单次服务营收（元）"])
        direct_cost[service] = to_number(row["单次服务直接支出（元）（基准价格）"])

    station_costs: dict[str, dict[str, float]] = {}
    for row in station_rows:
        scale = row["站点规模"]
        if scale not in SCALE_ORDER:
            continue
        station_costs[scale] = {
            "建设成本": to_number(row["一次性建设成本（万元）"]),
            "日固定成本": to_number(row["日均固定管理成本（元/日）"]),
            "容量": to_number(row["日最大服务人次"]),
        }

    distance: dict[str, dict[str, float]] = {}
    for row in distance_rows:
        community = row["组别"]
        if community not in COMMUNITIES:
            continue
        distance[community] = {target: to_number(row[target]) for target in COMMUNITIES}

    return population, transition, demand, price, direct_cost, station_costs, distance


def forecast_population(population, transition) -> dict[str, dict[str, int]]:
    rho_ah = transition["自理 → 半失能"]
    rho_hd = transition["半失能 → 失能"]
    year5: dict[str, dict[str, int]] = {}
    for community in COMMUNITIES:
        state = np.array([population[community][TYPE_COLS[t]] for t in TYPE_LABELS], dtype=float)
        for year in range(5):
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
        year5[community] = {
            "自理": int(state[0]),
            "半失能": int(state[1]),
            "失能": int(state[2]),
            "老人总数": int(state.sum()),
        }
    return year5


def build_constrained_demand(population, year5, demand, price):
    cap_ratio = {"自理": 0.20, "半失能": 0.25, "失能": 0.30}
    demand_detail: dict[str, dict[str, dict[str, int]]] = {
        community: {service: {t: 0 for t in TYPE_LABELS} for service in SERVICE_ORDER}
        for community in COMMUNITIES
    }
    monthly_total: dict[str, float] = {community: 0.0 for community in COMMUNITIES}
    monthly_revenue_base: dict[str, float] = {community: 0.0 for community in COMMUNITIES}
    monthly_direct_base: dict[str, float] = {community: 0.0 for community in COMMUNITIES}
    scaling_rows: list[dict[str, object]] = []

    for community in COMMUNITIES:
        income = population[community]["人均月收入"]
        for elder_type in TYPE_LABELS:
            theoretical_cost = sum(price[s] * demand[s][elder_type] for s in SERVICE_ORDER)
            limit = cap_ratio[elder_type] * income
            scale = min(1.0, limit / theoretical_cost) if theoretical_cost > 0 else 1.0
            scaling_rows.append(
                {
                    "小区": community,
                    "老人类型": elder_type,
                    "理论月费用": theoretical_cost,
                    "消费上限": limit,
                    "削减系数": scale,
                }
            )
            count = year5[community][elder_type]
            for service in SERVICE_ORDER:
                q_adj = demand[service][elder_type] * scale
                q_total = round_half_up(count * q_adj)
                demand_detail[community][service][elder_type] = q_total
                monthly_total[community] += q_total
                monthly_revenue_base[community] += q_total * price[service]
    return demand_detail, monthly_total, monthly_revenue_base, scaling_rows


def distance_satisfaction(distance_m: float) -> float:
    if distance_m <= 300:
        return 1.00
    if distance_m <= 500:
        return 0.90
    if distance_m <= 650:
        return 0.75
    if distance_m <= 1000:
        return 0.60
    return 0.0


def response_satisfaction(theta: float) -> float:
    if theta <= 0.60:
        return 1.00
    if theta <= 0.75:
        return 0.93
    if theta <= 0.85:
        return 0.85
    if theta <= 0.95:
        return 0.72
    if theta <= 1.00:
        return 0.60
    return 0.60


def build_stations(state: tuple[int, ...], station_costs) -> list[Station]:
    stations: list[Station] = []
    for idx, code in enumerate(state):
        if code == 0:
            continue
        scale = SCALE_ORDER[code - 1]
        cost = station_costs[scale]
        stations.append(
            Station(
                community=COMMUNITIES[idx],
                scale=scale,
                build_cost=cost["建设成本"],
                fixed_daily_cost=cost["日固定成本"],
                capacity=cost["容量"],
            )
        )
    return stations


def evaluate_plan(
    state: tuple[int, ...],
    station_costs,
    year5,
    demand_detail,
    monthly_total,
    price,
    direct_cost,
    distance,
) -> PlanResult | None:
    stations = build_stations(state, station_costs)
    if not stations:
        return None
    build_cost_total = sum(station.build_cost for station in stations)
    if build_cost_total > BUDGET_LIMIT + 1e-9:
        return None

    station_ids = [station.community for station in stations]
    station_map = {station.community: station for station in stations}
    available = {
        community: [sid for sid in station_ids if distance[community][sid] <= SERVICE_RADIUS]
        for community in COMMUNITIES
    }
    if not any(available.values()):
        return None

    def ordered_communities(mode: str, response: dict[str, float]) -> list[str]:
        def best_satisfaction(community: str) -> float:
            if not available[community]:
                return 0.0
            return max(
                0.2 * distance_satisfaction(distance[community][sid]) + 0.3 * response[sid] + 0.5
                for sid in available[community]
            )

        if mode == "elderly_desc":
            return sorted(COMMUNITIES, key=lambda c: (year5[c]["老人总数"], best_satisfaction(c)), reverse=True)
        if mode == "demand_asc":
            return sorted(COMMUNITIES, key=lambda c: (monthly_total[c], -best_satisfaction(c)))
        if mode == "efficiency_desc":
            return sorted(
                COMMUNITIES,
                key=lambda c: (
                    year5[c]["老人总数"] / max(monthly_total[c] / DAYS_PER_MONTH * best_satisfaction(c), 1e-9),
                    best_satisfaction(c),
                ),
                reverse=True,
            )
        if mode == "satisfaction_desc":
            return sorted(COMMUNITIES, key=lambda c: (best_satisfaction(c), year5[c]["老人总数"]), reverse=True)
        return COMMUNITIES[:]

    def settle_assignment(candidate_assignment: dict[str, str], start_response: dict[str, float]):
        response = dict(start_response)
        seen: list[dict[str, float]] = []
        last_payload = None

        for _ in range(30):
            sat: dict[str, float] = {}
            dist_score: dict[str, float] = {}
            theoretical = {sid: 0.0 for sid in station_ids}
            effective = {sid: 0.0 for sid in station_ids}
            for community, sid in candidate_assignment.items():
                s1 = distance_satisfaction(distance[community][sid])
                sij = 0.2 * s1 + 0.3 * response[sid] + 0.5
                daily = monthly_total[community] / DAYS_PER_MONTH
                dist_score[community] = s1
                sat[community] = sij
                theoretical[sid] += daily
                effective[sid] += daily * sij
            theta_local = {
                sid: effective[sid] / station_map[sid].capacity if station_map[sid].capacity else float("inf")
                for sid in station_ids
            }
            new_response = {sid: response_satisfaction(theta_local[sid]) for sid in station_ids}
            last_payload = (sat, dist_score, theoretical, effective, new_response, theta_local)
            if all(abs(new_response[sid] - response[sid]) < 1e-12 for sid in station_ids):
                return last_payload
            signature = tuple(new_response[sid] for sid in station_ids)
            if signature in [tuple(item[sid] for sid in station_ids) for item in seen]:
                conservative = {
                    sid: min([response[sid], new_response[sid], *[item[sid] for item in seen]])
                    for sid in station_ids
                }
                sat = {}
                dist_score = {}
                theoretical = {sid: 0.0 for sid in station_ids}
                effective = {sid: 0.0 for sid in station_ids}
                for community, sid in candidate_assignment.items():
                    s1 = distance_satisfaction(distance[community][sid])
                    sij = 0.2 * s1 + 0.3 * conservative[sid] + 0.5
                    daily = monthly_total[community] / DAYS_PER_MONTH
                    dist_score[community] = s1
                    sat[community] = sij
                    theoretical[sid] += daily
                    effective[sid] += daily * sij
                theta_local = {
                    sid: effective[sid] / station_map[sid].capacity if station_map[sid].capacity else float("inf")
                    for sid in station_ids
                }
                return sat, dist_score, theoretical, effective, conservative, theta_local
            seen.append(dict(response))
            response = new_response
        return last_payload

    assignment_candidates = []
    for mode in ["efficiency_desc", "elderly_desc", "demand_asc", "satisfaction_desc", "natural"]:
        response = {sid: 1.0 for sid in station_ids}
        assignment: dict[str, str] = {}
        satisfaction: dict[str, float] = {}
        distance_score: dict[str, float] = {}
        effective_daily_load: dict[str, float] = {sid: 0.0 for sid in station_ids}
        theoretical_daily_load: dict[str, float] = {sid: 0.0 for sid in station_ids}

        for _ in range(30):
            remaining = {sid: station_map[sid].capacity for sid in station_ids}
            new_assignment: dict[str, str] = {}
            new_satisfaction: dict[str, float] = {}
            new_distance_score: dict[str, float] = {}

            for community in ordered_communities(mode, response):
                choices = available[community]
                if not choices:
                    continue
                daily = monthly_total[community] / DAYS_PER_MONTH
                ranked = []
                for sid in choices:
                    s1 = distance_satisfaction(distance[community][sid])
                    sij = 0.2 * s1 + 0.3 * response[sid] + 0.5
                    effective_need = daily * sij
                    if effective_need <= remaining[sid] + 1e-9:
                        ranked.append((sij, s1, remaining[sid] - effective_need, -distance[community][sid], sid))
                if not ranked:
                    continue
                ranked.sort(reverse=True)
                sid = ranked[0][4]
                new_assignment[community] = sid
                new_satisfaction[community] = ranked[0][0]
                new_distance_score[community] = ranked[0][1]
                remaining[sid] -= daily * ranked[0][0]

            settled = settle_assignment(new_assignment, response)
            if settled is None:
                continue
            (
                new_satisfaction,
                new_distance_score,
                new_theoretical,
                new_effective,
                new_response,
                theta,
            ) = settled

            if new_assignment == assignment and all(abs(new_response[sid] - response[sid]) < 1e-12 for sid in station_ids):
                assignment = new_assignment
                satisfaction = new_satisfaction
                distance_score = new_distance_score
                theoretical_daily_load = new_theoretical
                effective_daily_load = new_effective
                response = new_response
                break

            assignment = new_assignment
            satisfaction = new_satisfaction
            distance_score = new_distance_score
            theoretical_daily_load = new_theoretical
            effective_daily_load = new_effective
            response = new_response

        if assignment and all(effective_daily_load[sid] <= station_map[sid].capacity + 1e-9 for sid in station_ids):
            covered_elderly_tmp = sum(year5[c]["老人总数"] for c in assignment)
            avg_sat_tmp = (
                sum(year5[c]["老人总数"] * satisfaction[c] for c in assignment) / covered_elderly_tmp
                if covered_elderly_tmp
                else 0.0
            )
            assignment_candidates.append(
                (
                    covered_elderly_tmp,
                    avg_sat_tmp,
                    sum(effective_daily_load.values()),
                    assignment,
                    satisfaction,
                    distance_score,
                    theoretical_daily_load,
                    effective_daily_load,
                    response,
                    {
                        sid: effective_daily_load[sid] / station_map[sid].capacity
                        if station_map[sid].capacity
                        else float("inf")
                        for sid in station_ids
                    },
                )
            )

    if not assignment_candidates:
        return None

    assignment_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    (
        _covered_elderly_tmp,
        _avg_sat_tmp,
        _effective_sum_tmp,
        assignment,
        satisfaction,
        distance_score,
        theoretical_daily_load,
        effective_daily_load,
        response,
        theta,
    ) = assignment_candidates[0]

    if not assignment:
        return None

    covered_elderly = sum(year5[c]["老人总数"] for c in assignment)
    total_elderly = sum(year5[c]["老人总数"] for c in COMMUNITIES)
    coverage_rate = covered_elderly / total_elderly
    avg_satisfaction_covered = (
        sum(year5[c]["老人总数"] * satisfaction[c] for c in assignment) / covered_elderly
        if covered_elderly
        else 0.0
    )
    avg_satisfaction_all = sum(year5[c]["老人总数"] * satisfaction.get(c, 0.0) for c in COMMUNITIES) / total_elderly

    revenue = {sid: 0.0 for sid in station_ids}
    variable = {sid: 0.0 for sid in station_ids}
    for community, sid in assignment.items():
        s = satisfaction[community]
        for service in SERVICE_ORDER:
            monthly_service = sum(demand_detail[community][service][t] for t in TYPE_LABELS)
            revenue[sid] += 12.0 * monthly_service * s * price[service]
            variable[sid] += 12.0 * monthly_service * s * direct_cost[service]
    fixed = {sid: station_map[sid].annual_fixed_cost for sid in station_ids}
    profit = {sid: revenue[sid] - variable[sid] - fixed[sid] for sid in station_ids}
    total_profit = sum(profit.values())

    return PlanResult(
        state=state,
        stations=stations,
        assignment=assignment,
        satisfaction=satisfaction,
        distance_score=distance_score,
        response_score=response,
        theta=theta,
        theoretical_daily_load=theoretical_daily_load,
        effective_daily_load=effective_daily_load,
        revenue=revenue,
        direct_cost=variable,
        fixed_cost=fixed,
        profit=profit,
        coverage_rate=coverage_rate,
        covered_elderly=covered_elderly,
        avg_satisfaction_covered=avg_satisfaction_covered,
        avg_satisfaction_all=avg_satisfaction_all,
        total_profit=total_profit,
        build_cost_total=build_cost_total,
        station_count=len(stations),
    )


def solve_question2(inputs):
    population, transition, demand, price, direct_cost, station_costs, distance = inputs
    year5 = forecast_population(population, transition)
    demand_detail, monthly_total, monthly_revenue_base, scaling_rows = build_constrained_demand(
        population, year5, demand, price
    )

    best: PlanResult | None = None
    evaluated: list[dict[str, object]] = []
    feasible_count = 0

    for state in itertools.product(range(4), repeat=len(COMMUNITIES)):
        stations = build_stations(state, station_costs)
        if not stations:
            continue
        build_cost = sum(station.build_cost for station in stations)
        if build_cost > BUDGET_LIMIT + 1e-9:
            continue
        result = evaluate_plan(state, station_costs, year5, demand_detail, monthly_total, price, direct_cost, distance)
        if result is None:
            continue
        feasible_count += 1
        evaluated.append(
            {
                "方案": "".join(str(x) for x in state),
                "站点数": result.station_count,
                "建设成本_万元": result.build_cost_total,
                "覆盖率": result.coverage_rate,
                "覆盖人口满意度": result.avg_satisfaction_covered,
                "全体人口满意度": result.avg_satisfaction_all,
                "年度总利润_元": result.total_profit,
            }
        )
        key = (
            result.coverage_rate,
            result.avg_satisfaction_covered,
            result.total_profit,
            -result.build_cost_total,
            -result.station_count,
        )
        if best is None:
            best = result
        else:
            best_key = (
                best.coverage_rate,
                best.avg_satisfaction_covered,
                best.total_profit,
                -best.build_cost_total,
                -best.station_count,
            )
            if key > best_key:
                best = result

    if best is None:
        raise RuntimeError("No feasible plan found.")

    evaluated.sort(
        key=lambda row: (
            float(row["覆盖率"]),
            float(row["覆盖人口满意度"]),
            float(row["年度总利润_元"]),
        ),
        reverse=True,
    )
    return {
        "population": population,
        "transition": transition,
        "demand": demand,
        "price": price,
        "direct_cost": direct_cost,
        "station_costs": station_costs,
        "distance": distance,
        "year5": year5,
        "demand_detail": demand_detail,
        "monthly_total": monthly_total,
        "scaling_rows": scaling_rows,
        "best": best,
        "evaluated": evaluated,
        "feasible_count": feasible_count,
    }


def choose_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Arial Unicode MS",
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


def classical_mds(distance: dict[str, dict[str, float]]) -> dict[str, tuple[float, float]]:
    dmat = np.array([[distance[i][j] for j in COMMUNITIES] for i in COMMUNITIES], dtype=float)
    n = dmat.shape[0]
    jmat = np.eye(n) - np.ones((n, n)) / n
    bmat = -0.5 * jmat @ (dmat**2) @ jmat
    eigvals, eigvecs = np.linalg.eigh(bmat)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    return {COMMUNITIES[i]: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}


def export_csv(path: Path, rows: list[dict[str, object]], headers: list[str] | None = None) -> None:
    if not rows and headers is None:
        return
    if headers is None:
        headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], headers: list[str] | None = None, digits: int = 3) -> str:
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


def station_plan_rows(best: PlanResult) -> list[dict[str, object]]:
    rows = []
    for station in best.stations:
        covered = [c for c, sid in best.assignment.items() if sid == station.community]
        rows.append(
            {
                "站点小区": station.community,
                "规模": station.scale,
                "覆盖小区": "、".join(covered),
                "建设成本_万元": station.build_cost,
                "日容量_人次": station.capacity,
                "日理论需求_人次": best.theoretical_daily_load[station.community],
                "日有效服务_人次": best.effective_daily_load[station.community],
                "利用率": best.theta[station.community],
                "响应满意度": best.response_score[station.community],
            }
        )
    return rows


def community_rows(best: PlanResult, year5, monthly_total, distance) -> list[dict[str, object]]:
    rows = []
    for community in COMMUNITIES:
        sid = best.assignment.get(community, "")
        rows.append(
            {
                "小区": community,
                "第5年老人总数": year5[community]["老人总数"],
                "月需求总人次": monthly_total[community],
                "分配站点": sid if sid else "未覆盖",
                "距离_米": distance[community][sid] if sid else "",
                "距离满意度": best.distance_score.get(community, 0.0),
                "综合满意度": best.satisfaction.get(community, 0.0),
            }
        )
    return rows


def financial_rows(best: PlanResult) -> list[dict[str, object]]:
    rows = []
    for station in best.stations:
        sid = station.community
        rows.append(
            {
                "站点小区": sid,
                "规模": station.scale,
                "年收入_元": best.revenue[sid],
                "年直接支出_元": best.direct_cost[sid],
                "年固定成本_元": best.fixed_cost[sid],
                "年度利润_元": best.profit[sid],
            }
        )
    rows.append(
        {
            "站点小区": "合计",
            "规模": "",
            "年收入_元": sum(best.revenue.values()),
            "年直接支出_元": sum(best.direct_cost.values()),
            "年固定成本_元": sum(best.fixed_cost.values()),
            "年度利润_元": best.total_profit,
        }
    )
    return rows


def plot_outputs(solution) -> None:
    choose_font()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    best: PlanResult = solution["best"]
    year5 = solution["year5"]
    monthly_total = solution["monthly_total"]
    distance = solution["distance"]
    evaluated = solution["evaluated"]

    coords = classical_mds(distance)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for community in COMMUNITIES:
        x, y = coords[community]
        ax.scatter(x, y, s=160, color="#E5E7EB", edgecolor="#374151", linewidth=1.0, zorder=2)
        ax.text(x, y, community, ha="center", va="center", fontsize=10, fontweight="bold", color="#111827", zorder=3)
    scale_color = {"小型": "#54A24B", "中型": "#F58518", "大型": "#E45756"}
    scale_size = {"小型": 330, "中型": 460, "大型": 590}
    for station in best.stations:
        x, y = coords[station.community]
        ax.scatter(
            x,
            y,
            s=scale_size[station.scale],
            color=scale_color[station.scale],
            alpha=0.88,
            edgecolor="#111827",
            linewidth=1.1,
            zorder=1,
            label=station.scale,
        )
    for community, sid in best.assignment.items():
        if community == sid:
            continue
        x1, y1 = coords[community]
        x2, y2 = coords[sid]
        ax.plot([x1, x2], [y1, y2], color="#6B7280", linewidth=1.1, alpha=0.55, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", frameon=False, title="服务站规模")
    ax.set_title("问题二最优服务站选址与覆盖关系", fontsize=15, fontweight="bold", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q2_station_layout.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    community_data = community_rows(best, year5, monthly_total, distance)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    vals = [row["综合满意度"] for row in community_data]
    colors = ["#4C78A8" if row["分配站点"] != "未覆盖" else "#D1D5DB" for row in community_data]
    ax.bar(COMMUNITIES, vals, color=colors, width=0.68)
    ax.axhline(best.avg_satisfaction_covered, color="#E45756", linewidth=2, linestyle="--", label="覆盖人口加权平均")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("满意度")
    ax.set_title("各小区老人综合满意度", fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q2_community_satisfaction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    station_rows_ = station_plan_rows(best)
    xlabels = [row["站点小区"] for row in station_rows_]
    load = [row["利用率"] for row in station_rows_]
    profit = [best.profit[row["站点小区"]] / 10000.0 for row in station_rows_]
    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
    ax2 = ax1.twinx()
    bars = ax1.bar(np.arange(len(xlabels)) - 0.18, load, width=0.36, color="#4C78A8", label="利用率")
    ax2.bar(np.arange(len(xlabels)) + 0.18, profit, width=0.36, color="#F58518", label="年度利润")
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_xticklabels(xlabels)
    ax1.set_ylim(0, max(1.0, max(load) * 1.15))
    ax1.set_ylabel("利用率")
    ax2.set_ylabel("年度利润（万元）")
    ax1.set_title("服务站利用率与年度利润", fontsize=15, fontweight="bold", pad=12)
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q2_station_load_profit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_plot = evaluated[:: max(1, len(evaluated) // 25000)]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sc = ax.scatter(
        [row["覆盖率"] for row in top_plot],
        [row["覆盖人口满意度"] for row in top_plot],
        c=[row["建设成本_万元"] for row in top_plot],
        s=18,
        cmap="viridis",
        alpha=0.58,
        linewidth=0,
    )
    ax.scatter(
        [best.coverage_rate],
        [best.avg_satisfaction_covered],
        color="#E45756",
        s=90,
        marker="*",
        label="最终方案",
        zorder=4,
    )
    ax.set_xlabel("覆盖率")
    ax.set_ylabel("覆盖人口加权满意度")
    ax.set_title("可行方案覆盖率-满意度散点", fontsize=15, fontweight="bold", pad=12)
    ax.grid(color="#E5E7EB", linewidth=0.9)
    ax.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("建设成本（万元）")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "q2_feasible_solution_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_outputs(solution) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    best: PlanResult = solution["best"]
    year5 = solution["year5"]
    monthly_total = solution["monthly_total"]
    distance = solution["distance"]

    export_csv(TABLE_DIR / "q2_station_plan.csv", station_plan_rows(best))
    export_csv(TABLE_DIR / "q2_community_assignment.csv", community_rows(best, year5, monthly_total, distance))
    export_csv(TABLE_DIR / "q2_station_financials.csv", financial_rows(best))
    export_csv(TABLE_DIR / "q2_top20_candidate_plans.csv", solution["evaluated"][:20])

    summary_rows = [
        {"指标": "可行方案数", "数值": solution["feasible_count"]},
        {"指标": "站点数量", "数值": best.station_count},
        {"指标": "建设成本（万元）", "数值": best.build_cost_total},
        {"指标": "服务覆盖率", "数值": best.coverage_rate},
        {"指标": "覆盖人口加权满意度", "数值": best.avg_satisfaction_covered},
        {"指标": "全体老人加权满意度", "数值": best.avg_satisfaction_all},
        {"指标": "年度总利润（元）", "数值": best.total_profit},
    ]
    export_csv(TABLE_DIR / "q2_summary.csv", summary_rows)

    paper = build_paper(solution, summary_rows)
    (DOC_DIR / "第二问论文.md").write_text(paper, encoding="utf-8")


def build_paper(solution, summary_rows: list[dict[str, object]]) -> str:
    best: PlanResult = solution["best"]
    year5 = solution["year5"]
    monthly_total = solution["monthly_total"]
    distance = solution["distance"]
    station_rows_ = station_plan_rows(best)
    community_rows_ = community_rows(best, year5, monthly_total, distance)
    financial_rows_ = financial_rows(best)

    total_elderly = sum(year5[c]["老人总数"] for c in COMMUNITIES)
    total_monthly_demand = sum(monthly_total.values())
    selected_desc = "、".join(f"{s.community}({s.scale})" for s in best.stations)

    paper = rf"""# 第二问：服务站选址与规模优化

## 1 问题重述与建模思路

第二问要求在总建设预算不超过 120 万元、服务半径不超过 1000 米、服务站日容量受规模限制的条件下，确定养老服务站的数量、位置和规模，使第 5 年末老人服务覆盖率和满意度尽可能高。本文沿用第一问得到的第 5 年末老人数量 \(N_{{i,5}}\) 与消费约束后的月需求 \(Q_{{i,s,t,5}}\)，将 10 个小区均视为候选站点，建立带容量约束的最大覆盖选址模型。

由于候选点只有 10 个，每个点只有“不建、小型、中型、大型”4 种状态，全部方案数为
\\[
4^{{10}}=1,048,576.
\\]
因此本文采用精确枚举而不是启发式搜索。对每个满足预算的站点规模方案，按“容量允许时优先选择综合满意度最高站点”的原则进行固定点分配，并计算覆盖率、满意度、容量、利润等指标，最后按词典序
\\[
\\max \\left(\\mathrm{{Cov}},\\overline S,\\sum_j \\Pi_j\\right)
\\]
筛选最优方案，即先最大化覆盖率，再最大化覆盖人口加权满意度，若仍并列则选择年度利润更高的方案。

## 2 模型建立

### 2.1 决策变量

设 \(I=\\{{A,B,C,D,E,F,G,H,I,J\\}}\) 为小区集合，\(K=\\{{\\text{{小型}},\\text{{中型}},\\text{{大型}}\\}}\) 为规模集合。定义
\\[
x_{{j,k}}=
\\begin{{cases}}
1,&\\text{{在小区 }}j\\text{{ 建设规模 }}k\\text{{ 的服务站}},\\\\
0,&\\text{{否则}},
\\end{{cases}}
\\]
以及
\\[
u_{{i,j}}=
\\begin{{cases}}
1,&\\text{{小区 }}i\\text{{ 分配给站点 }}j,\\\\
0,&\\text{{否则}}.
\\end{{cases}}
\\]
每个小区最多建设一个服务站：
\\[
\\sum_{{k\\in K}}x_{{j,k}}\\le 1,\\quad j\\in I.
\\]

### 2.2 预算、半径与容量约束

设 \(B_k\) 为规模 \(k\) 的一次性建设成本，单位为万元。预算约束为
\\[
\\sum_{{j\\in I}}\\sum_{{k\\in K}}B_kx_{{j,k}}\\le 120.
\\]

设 \(d_{{ij}}\) 为小区 \(i\) 到候选站点 \(j\) 的距离。超出 1000 米不产生有效服务需求，因此有
\\[
u_{{i,j}}=0,\\quad d_{{ij}}>1000,
\\]
且小区只能分配给已建站点：
\\[
u_{{i,j}}\\le \\sum_{{k\\in K}}x_{{j,k}}.
\\]
每个小区最多选择一个站点：
\\[
\\sum_{{j\\in I}}u_{{i,j}}\\le 1.
\\]

第一问给出的消费约束后月需求总人次记为
\\[
D_i^m=\\sum_{{s\\in S}}\\sum_{{t\\in T}}Q_{{i,s,t,5}},
\\]
按每月 30 天折算为日理论需求
\\[
D_i^d=\\frac{{D_i^m}}{{30}}.
\\]

### 2.3 满意度与实际有效服务人次

距离满意度 \(S_{{1,ij}}\) 由附件5的分段规则给出：
\\[
S_{{1,ij}}=
\\begin{{cases}}
1.00,& d_{{ij}}\\le 300,\\\\
0.90,& 300<d_{{ij}}\\le 500,\\\\
0.75,& 500<d_{{ij}}\\le 650,\\\\
0.60,& 650<d_{{ij}}\\le 1000.
\\end{{cases}}
\\]

问题2不优化价格，默认按附件2基准价格收费，因此价格满意度取 \(S_3=1\)。设站点 \(j\) 的利用率为
\\[
\\theta_j=\\frac{{V_j}}{{C_{{k(j)}}}},
\\]
响应满意度 \(S_{{2,j}}\) 按附件5规则由 \(\theta_j\) 分段给出。小区 \(i\) 分配到站点 \(j\) 的综合满意度为
\\[
S_{{ij}}=0.2S_{{1,ij}}+0.3S_{{2,j}}+0.5S_3.
\\]
实际有效服务人次等于理论需求人次乘以满意度：
\\[
V_j=\\sum_{{i\\in I}}u_{{i,j}}D_i^dS_{{ij}}.
\\]
容量约束为
\\[
V_j\\le C_{{k(j)}}.
\\]

由于 \(S_{{2,j}}\) 依赖利用率，而利用率又依赖分配和满意度，本文对每个站点方案进行固定点迭代：先令所有建成站点 \(S_2=1\)，据此在不突破站点容量的前提下按最高综合满意度准入覆盖小区；再计算实际有效服务人次、利用率和新的 \(S_2\)，重复直到分配和响应满意度稳定。若某小区在所有半径内站点上都会导致容量超限，则该小区暂不计入覆盖。

### 2.4 评价指标与利润核算

服务覆盖率定义为
\\[
\\mathrm{{Cov}}=
\\frac{{\\sum_{{i\\in I}}N_{{i,5}}\\mathbf 1(\\sum_j u_{{i,j}}\\ge1)}}
{{\\sum_{{i\\in I}}N_{{i,5}}}}.
\\]
覆盖人口加权平均满意度定义为
\\[
\\overline S=
\\frac{{\\sum_{{i\\in I}}N_{{i,5}}\\sum_j u_{{i,j}}S_{{ij}}}}
{{\\sum_{{i\\in I}}N_{{i,5}}\\sum_j u_{{i,j}}}}.
\\]

设 \(p_s^0\) 为服务 \(s\) 基准价格，\(c_s\) 为直接支出。站点 \(j\) 的年度收入、直接支出和固定成本分别为
\\[
R_j=12\\sum_{{i,s,t}}u_{{i,j}}Q_{{i,s,t,5}}S_{{ij}}p_s^0,
\\]
\\[
G_j=12\\sum_{{i,s,t}}u_{{i,j}}Q_{{i,s,t,5}}S_{{ij}}c_s,
\\]
\\[
A_j=365F_{{k(j)}}+\\frac{{10000B_{{k(j)}}}}{{20}}.
\\]
预计年度利润为
\\[
\\Pi_j=R_j-G_j-A_j.
\\]

## 3 求解算法

精确枚举算法如下。

1. 对 10 个候选小区生成“不建、小型、中型、大型”的全部组合；
2. 删除建设成本超过 120 万元的方案；
3. 对每个预算可行方案，找出每个小区 1000 米内可选站点；
4. 初始化各站点响应满意度 \(S_2=1\)；
5. 按 \(S_{{ij}}=0.2S_1+0.3S_2+0.5\) 为每个可覆盖小区选择满意度最高且仍有容量余量的站点；
6. 根据分配结果计算各站点理论需求、有效服务人次、利用率和新的响应满意度；
7. 重复第 5-6 步直至分配和响应满意度稳定；
8. 若小区无法在容量约束下分配，则视为未覆盖；对可行分配计算覆盖率、满意度、年度利润；
9. 按 \((\\mathrm{{Cov}},\\overline S,\\sum_j\\Pi_j)\) 的词典序选择最终方案。

若小区数为 \(n\)，固定点迭代次数为 \(L\)，则枚举复杂度为
\\[
O(4^n\\cdot L\\cdot n^2).
\\]
本题 \(n=10\)，实际可行方案数为 {solution["feasible_count"]}，因此可在普通计算机上直接完成精确搜索。

## 4 求解结果

第 5 年末全街道老人总数为 {total_elderly} 人，消费约束后全街道月服务需求总人次为 {total_monthly_demand:.0f} 次。最优方案为：{selected_desc}。

核心指标如下。

{markdown_table(summary_rows, ["指标", "数值"])}

### 4.1 站点位置、规模与覆盖关系

{markdown_table(station_rows_, ["站点小区", "规模", "覆盖小区", "建设成本_万元", "日容量_人次", "日理论需求_人次", "日有效服务_人次", "利用率", "响应满意度"])}

![问题二最优服务站选址与覆盖关系](../img/q2_station_layout.png)

### 4.2 小区分配与满意度

{markdown_table(community_rows_, ["小区", "第5年老人总数", "月需求总人次", "分配站点", "距离_米", "距离满意度", "综合满意度"])}

![各小区老人综合满意度](../img/q2_community_satisfaction.png)

### 4.3 年度利润

{markdown_table(financial_rows_, ["站点小区", "规模", "年收入_元", "年直接支出_元", "年固定成本_元", "年度利润_元"], digits=2)}

![服务站利用率与年度利润](../img/q2_station_load_profit.png)

可行方案的覆盖率-满意度分布如下图所示，红色星标为最终选定方案。

![可行方案覆盖率-满意度散点](../img/q2_feasible_solution_scatter.png)

## 5 结果解释

最优方案在 120 万元预算内使用 {best.build_cost_total:.0f} 万元，建设 {best.station_count} 个服务站，实现服务覆盖率 {best.coverage_rate:.2%}，覆盖人口加权满意度 {best.avg_satisfaction_covered:.4f}。由于容量约束较紧，最优方案选择优先覆盖老人规模大且单位容量收益较高的小区；部分小区即使位于某个服务半径内，若会导致站点容量超限，也不会被计入覆盖。

年度利润核算中，本文采用问题2基准价格口径，尚未引入问题3的政府补贴和自主定价机制。因此该利润仅作为站点方案的辅助评价指标；第三问还需要继续通过补贴与定价优化满足“保本微利、利润率不超过 8%”的政策目标。

## 6 模型局限性与改进方向

模型局限性如下。

1. 将每个小区抽象为一个点，距离矩阵无法反映小区内部楼栋分布、步行路径和道路通达差异；
2. 将服务能力统一折算为总人次，未区分助餐、护理、康复、助浴等服务对人员技能和场地设备的不同要求；
3. 假设老人总是选择综合满意度最高的服务站，忽略习惯、信息不对称、亲友陪同和站点口碑等行为因素；
4. 满意度采用分段常数函数，距离或利用率在阈值附近的小变化会导致评分跳变；
5. 仅以第 5 年末需求作静态规划，没有刻画建设爬坡、年度扩容和短期高峰需求。

改进方向为：在后续研究中可引入道路网络时间距离替代小区间距离，并将容量约束细分为餐位、护理员工时、康复设备、助浴设施等多资源约束；同时可使用 Logit 离散选择模型表示老人选站概率，从“确定性选择最高满意度站点”推广为“多因素概率选择”，使模型更贴近实际运营。
"""
    return paper.replace("\\\\", "\\")


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    solution = solve_question2(read_inputs())
    plot_outputs(solution)
    export_outputs(solution)
    best: PlanResult = solution["best"]
    print(f"Generated paper: {DOC_DIR / '第二问论文.md'}")
    print(f"Generated tables: {TABLE_DIR}")
    print(f"Generated images: {IMG_DIR}")
    print(
        "Best plan:",
        ", ".join(f"{station.community}-{station.scale}" for station in best.stations),
        f"coverage={best.coverage_rate:.4f}",
        f"satisfaction={best.avg_satisfaction_covered:.4f}",
        f"profit={best.total_profit:.2f}",
    )


if __name__ == "__main__":
    main()
