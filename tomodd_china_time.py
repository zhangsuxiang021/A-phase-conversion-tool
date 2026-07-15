#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新旧地震台网 DBO/DPB 或 JOPENS 观测报告转换为 hypoDD/tomoDD输入格式。

输入文件、输出文件与起始事件编号既可以在“用户配置区”直接修改，也可以通过命令行覆盖。

核心规则
--------
1. 台站标签固定为“台网代码 + 台站代码”，例如 AH + CHZ -> AHCHZ。
2. Pg/Pn/P 归入 P-family；Sg/Sn/S 归入 S-family。
3. 同一事件、同一台站：只保留最早的 P-family 到时和最早的 S-family 到时。
4. PmP、SmS、SME、SMN 等反射相/振幅记录不作为 P/S 写出。
5. 若同一台站最早 S 不晚于最早 P，则寻找下一条晚于 P 的有效 S；
   若不存在，则删除该异常 S，并在日志中记录。
6. 原始报告时间按中国标准时间（北京时间，UTC+8）解释；输出事件头同样保留中国标准时间；走时由同一时区下的到时减发震时刻得到。

推荐用法
--------
直接使用脚本顶部配置：
    python tomodd_integrated.py

命令行指定一个或多个输入文件：
    python tomodd_integrated.py phase.dat new_report.txt -o phaseps_dp.dat

其他常用参数：
    python tomodd_integrated.py phase.dat -o phaseps_dp.dat --start-id 1
    python tomodd_integrated.py "reports/*.txt" -o phaseps_dp.dat
    python tomodd_integrated.py phase.dat --max-travel-time 1800

Python 版本：3.8+
仅使用标准库。

开发作者：Suxiang Zhang(1041573288@qq.com)
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# 用户配置区：不使用命令行参数时，程序采用这里的设置
# =============================================================================

# 可写多个文件，也可写通配符，例如 ["data/*.dat", "reports/*.txt"]。
INPUT_FILES = ["phase.dat"]

# hypoDD/tomoDD 绝对走时震相输出文件。
OUTPUT_FILE = "phaseps_dp_china.dat"

# 输出事件的起始编号。
START_EVENT_ID = 1

# 区域目录的最大允许走时。1800 s 足以覆盖常见区域地震，同时可排除明显的
# 日期/时钟错误；如确需远震资料，可在此处或命令行中调大。
MAX_TRAVEL_TIME = 1800.0

# 最小正走时。
MIN_TRAVEL_TIME = 0.001

# 无显式日期的到时若比发震时刻早超过该阈值，按次日到时处理。
CROSS_DAY_THRESHOLD_SEC = 6.0 * 3600.0

# tomoDD/hypoDD 本项目采用的最长台站标签长度：2 位台网 + 5 位台站。
MAX_NETWORK_LEN = 2
MAX_STATION_LEN = 5
STRICT_STATION_LENGTH = True

# 若为 True，最早 S 必须晚于最早 P；不满足时尝试下一条 S。
ENFORCE_S_AFTER_P = True
MIN_S_MINUS_P = 0.0

# 是否写出完全没有有效震相的事件。通常应为 False。
WRITE_EVENTS_WITHOUT_PICKS = False

# 固定输出权重。选择“首波”时首先比较到时；权重只用于完全同到时的破同。
DPB_PHASE_WEIGHTS = {
    "Pg": (1.00, "P"),
    "Pn": (1.00, "P"),
    "P":  (0.50, "P"),
    "Sg": (0.60, "S"),
    "Sn": (0.60, "S"),
    "S":  (0.25, "S"),
}

JOPENS_PHASE_WEIGHTS = {
    "Pg": (1.00, "P"),
    "Pn": (1.00, "P"),
    "P":  (0.50, "P"),
    "Sg": (0.60, "S"),
    "Sn": (0.60, "S"),
    "S":  (0.25, "S"),
}


# =============================================================================
# 常量与数据结构
# =============================================================================

CHINA_TZ = timezone(timedelta(hours=8), name="UTC+08:00")

ACCEPTED_PHASES = frozenset(DPB_PHASE_WEIGHTS)
MAGNITUDE_TYPES = frozenset({
    "ML", "Ml", "ml", "MS", "Ms", "ms", "mb", "mB", "Mb", "MB",
    "MW", "Mw", "mw", "Md", "MD", "mbLg", "MLv",
})
DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
TIME_RE = re.compile(r"^(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d*)?)$")


@dataclass
class Config:
    input_patterns: List[str]
    output_file: Path
    start_event_id: int = START_EVENT_ID
    max_travel_time: float = MAX_TRAVEL_TIME
    min_travel_time: float = MIN_TRAVEL_TIME
    cross_day_threshold_sec: float = CROSS_DAY_THRESHOLD_SEC
    strict_station_length: bool = STRICT_STATION_LENGTH
    enforce_s_after_p: bool = ENFORCE_S_AFTER_P
    min_s_minus_p: float = MIN_S_MINUS_P
    write_events_without_picks: bool = WRITE_EVENTS_WITHOUT_PICKS


@dataclass
class Event:
    origin_local: datetime
    lat: float
    lon: float
    dep: float
    mag: float
    source_file: str
    source_line: int
    dbo_variant: str = ""


@dataclass
class Pick:
    station: str
    travel_time: float
    output_weight: float
    family: str
    original_phase: str
    line_no: int
    source_file: str
    arrival_local: datetime
    residual: Optional[float] = None
    input_weight: Optional[float] = None
    channel: str = ""
    raw_line: str = ""


@dataclass
class Stats:
    files_requested: int = 0
    files_processed: int = 0
    files_missing: int = 0
    files_unknown_format: int = 0
    events_read: int = 0
    events_written: int = 0
    events_without_picks: int = 0
    raw_valid_picks: int = 0
    picks_written: int = 0
    duplicate_groups: Counter = field(default_factory=Counter)
    duplicate_picks_removed: Counter = field(default_factory=Counter)
    phase_in: Counter = field(default_factory=Counter)
    phase_kept: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    formats: Counter = field(default_factory=Counter)
    dbo_variants: Counter = field(default_factory=Counter)
    encodings: Counter = field(default_factory=Counter)
    station_truncations: Counter = field(default_factory=Counter)
    station_collision_map: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    examples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_example(self, key: str, text: str, limit: int = 8) -> None:
        if len(self.examples[key]) < limit:
            self.examples[key].append(text)


# =============================================================================
# 通用解析工具
# =============================================================================


def detect_text_encoding(path: Path) -> str:
    """检测常见中文文本编码；震相字段均为 ASCII，失败时仍可安全忽略坏字符。"""
    sample = path.read_bytes()[:131072]
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def parse_float(token: str) -> Optional[float]:
    try:
        value = float(token)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_date_token(token: str) -> Optional[Tuple[int, int, int]]:
    match = DATE_RE.match(token.strip())
    if not match:
        return None
    y, m, d = map(int, match.groups())
    try:
        datetime(y, m, d)
    except ValueError:
        return None
    return y, m, d


def parse_time_token(token: str) -> Optional[Tuple[int, int, float]]:
    match = TIME_RE.match(token.strip())
    if not match:
        return None
    h = int(match.group(1))
    minute = int(match.group(2))
    sec = float(match.group(3))
    if not (0 <= h <= 23 and 0 <= minute <= 59 and 0.0 <= sec < 61.0):
        return None
    return h, minute, sec


def make_local_datetime(year: int, month: int, day: int,
                        hour: int, minute: int, second: float) -> datetime:
    """稳健处理浮点秒及 59.999999 舍入进位。"""
    base = datetime(year, month, day, hour, minute, 0, tzinfo=CHINA_TZ)
    total_microseconds = int(round(second * 1_000_000.0))
    return base + timedelta(microseconds=total_microseconds)


def parse_explicit_datetime(date_token: str, time_token: str) -> Optional[datetime]:
    date_parts = parse_date_token(date_token)
    time_parts = parse_time_token(time_token)
    if date_parts is None or time_parts is None:
        return None
    return make_local_datetime(*date_parts, *time_parts)


def infer_arrival_datetime(origin_local: datetime, time_token: str,
                           cross_day_threshold_sec: float) -> Optional[datetime]:
    time_parts = parse_time_token(time_token)
    if time_parts is None:
        return None
    arrival = make_local_datetime(
        origin_local.year, origin_local.month, origin_local.day, *time_parts
    )
    if arrival < origin_local:
        backward = (origin_local - arrival).total_seconds()
        if backward >= cross_day_threshold_sec:
            arrival += timedelta(days=1)
    return arrival


def is_jopens_header(line: str) -> bool:
    """旧 JOPENS 事件头的日期斜杠位于固定列。"""
    return len(line) > 10 and line[7:8] == "/" and line[10:11] == "/"


def expand_input_patterns(patterns: Sequence[str], output_file: Path) -> List[Path]:
    """展开输入文件和通配符，保持用户给定顺序并去重。"""
    result: List[Path] = []
    seen = set()
    output_resolved = output_file.resolve()

    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            matches = [Path(pattern)]
        for path in sorted(matches, key=lambda p: str(p)):
            resolved = path.expanduser().resolve()
            if resolved == output_resolved:
                continue
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
    return result


def detect_file_format(path: Path, encoding: str) -> Optional[str]:
    """返回 DBO 或 JOPENS；新旧 DBO/DPB 在逐行解析时进一步区分。"""
    with path.open("r", encoding=encoding, errors="ignore") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            line = raw.rstrip("\r\n")
            if line.lstrip().startswith("DBO"):
                return "DBO"
            if is_jopens_header(line) or is_jopens_header(line.strip()):
                return "JOPENS"
    return None


def make_station_label(network: str, station: str,
                       cfg: Config, stats: Stats,
                       source_file: str, line_no: int) -> Optional[str]:
    network_raw = network.strip()
    station_raw = station.strip()
    if not network_raw or not station_raw:
        stats.skipped["missing_station_code"] += 1
        return None

    too_long = len(network_raw) > MAX_NETWORK_LEN or len(station_raw) > MAX_STATION_LEN
    if too_long and cfg.strict_station_length:
        stats.skipped["station_code_too_long"] += 1
        stats.add_example(
            "station_code_too_long",
            f"{source_file}:{line_no}: network={network_raw!r}, station={station_raw!r}",
        )
        return None

    network_out = network_raw[:MAX_NETWORK_LEN]
    station_out = station_raw[:MAX_STATION_LEN]
    label = network_out + station_out

    if too_long:
        original = network_raw + station_raw
        stats.station_truncations[original] += 1
        stats.station_collision_map[label].add(original)

    return label


# =============================================================================
# DBO/DPB 解析
# =============================================================================


def parse_dbo_event_line(line: str, source_file: str, line_no: int,
                         stats: Stats) -> Optional[Event]:
    """
    同时解析两种 DBO：

    旧格式：
        DBO AH 2026-01-26 17:07:40.11 31.982 117.586 12 ML 1.7 ...

    新格式：
        DBO CB 2026/02/05 05:13:46.0 32.879 120.818 10 2.1 1 ...
    """
    parts = line.split()
    if len(parts) < 8:
        stats.skipped["short_dbo_event"] += 1
        return None

    origin_local = parse_explicit_datetime(parts[2], parts[3])
    lat = parse_float(parts[4])
    lon = parse_float(parts[5])
    dep = parse_float(parts[6])
    if origin_local is None or lat is None or lon is None or dep is None:
        stats.skipped["bad_dbo_core_fields"] += 1
        stats.add_example("bad_dbo_core_fields", f"{source_file}:{line_no}: {line}")
        return None

    mag = 0.0
    variant = "unknown"

    # 旧格式：深度后为震级类型，再跟震级值。
    if len(parts) >= 9 and parts[7] in MAGNITUDE_TYPES:
        parsed_mag = parse_float(parts[8])
        if parsed_mag is not None:
            mag = parsed_mag
        variant = "typed_magnitude_hyphen" if "-" in parts[2] else "typed_magnitude_slash"
    else:
        # 新格式：深度后的第一个数值就是震级。
        parsed_mag = parse_float(parts[7])
        if parsed_mag is not None:
            mag = parsed_mag
            variant = "direct_magnitude_slash" if "/" in parts[2] else "direct_magnitude_hyphen"
        else:
            # 兜底：在后续字段中搜索震级类型 + 数值。
            for idx, token in enumerate(parts[7:], start=7):
                if token in MAGNITUDE_TYPES and idx + 1 < len(parts):
                    parsed_mag = parse_float(parts[idx + 1])
                    if parsed_mag is not None:
                        mag = parsed_mag
                        variant = "searched_typed_magnitude"
                        break

    if mag < 0.0:
        stats.skipped["negative_magnitude_reset_to_zero"] += 1
        mag = 0.0

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        stats.skipped["invalid_event_coordinates"] += 1
        stats.add_example("invalid_event_coordinates", f"{source_file}:{line_no}: {line}")
        return None

    stats.dbo_variants[variant] += 1
    return Event(
        origin_local=origin_local,
        lat=lat,
        lon=lon,
        dep=dep,
        mag=mag,
        source_file=source_file,
        source_line=line_no,
        dbo_variant=variant,
    )


def find_phase_token(parts: Sequence[str]) -> Optional[Tuple[int, str]]:
    """在 DPB 前部字段中寻找目标首波相，兼容可选 U/R/C/D 等字段。"""
    upper_bound = min(len(parts), 12)
    for idx in range(3, upper_bound):
        token = parts[idx]
        if token in ACCEPTED_PHASES:
            return idx, token
    return None


def find_arrival_in_dpb(parts: Sequence[str], origin_local: datetime,
                        cfg: Config) -> Tuple[Optional[datetime], bool, Optional[int]]:
    """返回（到时、日期是否推断、时间字段索引）。"""
    # 优先使用显式 YYYY-MM-DD/HH:MM:SS 或 YYYY/MM/DD HH:MM:SS。
    for idx in range(len(parts) - 1):
        if parse_date_token(parts[idx]) is not None:
            arrival = parse_explicit_datetime(parts[idx], parts[idx + 1])
            if arrival is not None:
                return arrival, False, idx + 1

    # 兜底：只找到时间字段时，使用发震日期并处理跨日。
    for idx, token in enumerate(parts):
        if parse_time_token(token) is not None:
            arrival = infer_arrival_datetime(
                origin_local, token, cfg.cross_day_threshold_sec
            )
            return arrival, True, idx

    return None, False, None


def parse_dpb_pick(line: str, event: Event, cfg: Config,
                   stats: Stats, source_file: str, line_no: int) -> Optional[Pick]:
    parts = line.split()
    if len(parts) < 6:
        stats.skipped["short_dpb"] += 1
        return None

    phase_info = find_phase_token(parts)
    if phase_info is None:
        # PmP/SmS/SME/SMN 等均在此作为非目标记录忽略。
        stats.skipped["non_first_arrival_phase_or_amplitude"] += 1
        return None
    phase_idx, phase = phase_info

    station_label = make_station_label(
        parts[1], parts[2], cfg, stats, source_file, line_no
    )
    if station_label is None:
        return None

    channel = parts[3]
    input_weight = parse_float(parts[phase_idx + 1]) if phase_idx + 1 < len(parts) else None

    arrival_local, inferred_date, time_idx = find_arrival_in_dpb(
        parts, event.origin_local, cfg
    )
    if arrival_local is None:
        stats.skipped["missing_dpb_arrival_time"] += 1
        stats.add_example("missing_dpb_arrival_time", f"{source_file}:{line_no}: {line}")
        return None

    travel_time = (arrival_local - event.origin_local).total_seconds()
    if travel_time < 0.0:
        # 仅在输入没有显式日期时允许按跨日规则修正。
        if inferred_date and abs(travel_time) >= cfg.cross_day_threshold_sec:
            travel_time += 86400.0
        else:
            stats.skipped["negative_travel_time"] += 1
            stats.add_example(
                "negative_travel_time",
                f"{source_file}:{line_no}: tt={travel_time:.3f}; {line}",
            )
            return None

    if not (cfg.min_travel_time <= travel_time <= cfg.max_travel_time):
        stats.skipped["travel_time_out_of_range"] += 1
        stats.add_example(
            "travel_time_out_of_range",
            f"{source_file}:{line_no}: tt={travel_time:.3f}; {line}",
        )
        return None

    residual: Optional[float] = None
    if time_idx is not None and time_idx + 1 < len(parts):
        residual = parse_float(parts[time_idx + 1])

    output_weight, family = DPB_PHASE_WEIGHTS[phase]
    stats.raw_valid_picks += 1
    stats.phase_in[phase] += 1

    return Pick(
        station=station_label,
        travel_time=travel_time,
        output_weight=output_weight,
        family=family,
        original_phase=phase,
        line_no=line_no,
        source_file=source_file,
        arrival_local=arrival_local,
        residual=residual,
        input_weight=input_weight,
        channel=channel,
        raw_line=line,
    )


# =============================================================================
# JOPENS 解析
# =============================================================================


def parse_jopens_event_line(line: str, source_file: str, line_no: int,
                            stats: Stats) -> Optional[Event]:
    try:
        chars = list(line)
        for pos in (7, 10, 16, 19):
            if pos < len(chars):
                chars[pos] = " "
        normalized = "".join(chars)
        parts = normalized[:58].split()
        if len(parts) < 11:
            raise ValueError("event header has too few fields")

        year, month, day = map(int, parts[1:4])
        hour, minute = int(parts[4]), int(parts[5])
        second = float(parts[6])
        lat, lon, dep, mag = map(float, parts[7:11])
        if mag < 0.0:
            mag = 0.0

        origin_local = make_local_datetime(
            year, month, day, hour, minute, second
        )
        return Event(
            origin_local=origin_local,
            lat=lat,
            lon=lon,
            dep=dep,
            mag=mag,
            source_file=source_file,
            source_line=line_no,
            dbo_variant="JOPENS",
        )
    except Exception as exc:
        stats.skipped["bad_jopens_event"] += 1
        stats.add_example(
            "bad_jopens_event", f"{source_file}:{line_no}: {exc}; {line}"
        )
        return None


def parse_jopens_pick(line: str, event: Event,
                       previous_station: Tuple[str, str],
                       cfg: Config, stats: Stats,
                       source_file: str, line_no: int
                       ) -> Tuple[Optional[Pick], Tuple[str, str]]:
    chars = list(line)
    for pos in (34, 37):
        if pos < len(chars):
            chars[pos] = " "
    normalized = "".join(chars)
    fields = normalized[17:44].split()
    if len(fields) < 6:
        stats.skipped["short_jopens_pick"] += 1
        return None, previous_station

    phase = fields[0]
    if phase not in JOPENS_PHASE_WEIGHTS:
        stats.skipped["non_first_arrival_phase_or_amplitude"] += 1
        return None, previous_station

    try:
        hour = int(float(fields[3]))
        minute = int(float(fields[4]))
        second = float(fields[5])
    except (TypeError, ValueError):
        stats.skipped["bad_jopens_arrival_time"] += 1
        return None, previous_station

    network, station = previous_station
    if normalized[:3] != "   ":
        network = normalized[0:2].strip()
        station = normalized[2:7].strip()

    station_label = make_station_label(
        network, station, cfg, stats, source_file, line_no
    )
    updated_station = (network, station)
    if station_label is None:
        return None, updated_station

    arrival_local = make_local_datetime(
        event.origin_local.year,
        event.origin_local.month,
        event.origin_local.day,
        hour,
        minute,
        second,
    )
    if arrival_local < event.origin_local:
        backward = (event.origin_local - arrival_local).total_seconds()
        if backward >= cfg.cross_day_threshold_sec:
            arrival_local += timedelta(days=1)

    travel_time = (arrival_local - event.origin_local).total_seconds()
    if not (cfg.min_travel_time <= travel_time <= cfg.max_travel_time):
        stats.skipped["travel_time_out_of_range"] += 1
        return None, updated_station

    output_weight, family = JOPENS_PHASE_WEIGHTS[phase]
    stats.raw_valid_picks += 1
    stats.phase_in[phase] += 1

    pick = Pick(
        station=station_label,
        travel_time=travel_time,
        output_weight=output_weight,
        family=family,
        original_phase=phase,
        line_no=line_no,
        source_file=source_file,
        arrival_local=arrival_local,
        raw_line=line,
    )
    return pick, updated_station


# =============================================================================
# 首波选择、物理一致性与输出
# =============================================================================


def pick_sort_key(pick: Pick) -> Tuple[float, float, float, int]:
    """
    首要标准严格为最早走时；仅在走时完全相同时依次使用：
    较小绝对残差、较高输出权重、较早原始行号。
    """
    abs_residual = abs(pick.residual) if pick.residual is not None else math.inf
    return (
        pick.travel_time,
        abs_residual,
        -pick.output_weight,
        pick.line_no,
    )


def select_first_arrivals(picks: Sequence[Pick], cfg: Config,
                          stats: Stats) -> List[Pick]:
    """同一事件、同一台站分别选择最早 P-family 与最早 S-family。"""
    by_station_family: Dict[Tuple[str, str], List[Pick]] = defaultdict(list)
    by_station: Dict[str, Dict[str, List[Pick]]] = defaultdict(lambda: defaultdict(list))

    for pick in picks:
        by_station_family[(pick.station, pick.family)].append(pick)
        by_station[pick.station][pick.family].append(pick)

    selected: Dict[Tuple[str, str], Pick] = {}
    for (station, family), candidates in by_station_family.items():
        candidates_sorted = sorted(candidates, key=pick_sort_key)
        selected[(station, family)] = candidates_sorted[0]

        if len(candidates_sorted) > 1:
            stats.duplicate_groups[family] += 1
            stats.duplicate_picks_removed[family] += len(candidates_sorted) - 1
            kept = candidates_sorted[0]
            description = ", ".join(
                f"{p.original_phase}/{p.channel}:{p.travel_time:.3f}s"
                for p in candidates_sorted
            )
            stats.add_example(
                "first_arrival_selection",
                f"{station} {family}: keep "
                f"{kept.original_phase}/{kept.channel}:{kept.travel_time:.3f}s; "
                f"candidates=[{description}]",
            )

    # P-S 物理顺序检查。保留最早 P，并从所有 S 候选中寻找最早的 S>P。
    if cfg.enforce_s_after_p:
        for station, family_map in by_station.items():
            p_pick = selected.get((station, "P"))
            s_pick = selected.get((station, "S"))
            if p_pick is None or s_pick is None:
                continue
            if s_pick.travel_time > p_pick.travel_time + cfg.min_s_minus_p:
                continue

            valid_s = sorted(
                (
                    p for p in family_map.get("S", [])
                    if p.travel_time > p_pick.travel_time + cfg.min_s_minus_p
                ),
                key=pick_sort_key,
            )
            if valid_s:
                replacement = valid_s[0]
                selected[(station, "S")] = replacement
                stats.skipped["earliest_s_replaced_for_ps_order"] += 1
                stats.add_example(
                    "ps_order_correction",
                    f"{station}: P={p_pick.travel_time:.3f}s; "
                    f"replace S={s_pick.travel_time:.3f}s with "
                    f"{replacement.travel_time:.3f}s",
                )
            else:
                del selected[(station, "S")]
                stats.skipped["s_removed_not_after_p"] += 1
                stats.add_example(
                    "ps_order_correction",
                    f"{station}: P={p_pick.travel_time:.3f}s; "
                    f"remove invalid S={s_pick.travel_time:.3f}s",
                )

    final_picks = sorted(
        selected.values(),
        key=lambda p: (p.station, 0 if p.family == "P" else 1, p.travel_time),
    )
    for pick in final_picks:
        stats.phase_kept[pick.original_phase] += 1
    return final_picks


def write_event_block(handle, event: Event, event_id: int,
                      picks: Sequence[Pick]) -> None:
    """
    写出一个 hypoDD/tomoDD 事件块。

    event.origin_local 在解析输入时已被明确赋予中国标准时间（UTC+8）时区。
    本版本直接写出该中国时，不再转换为 UTC。震相走时仍是在中国时坐标下由
    到时减发震时刻得到，因此与是否转换事件头时区无关。
    """
    china_origin = event.origin_local
    china_second = (
        china_origin.second + china_origin.microsecond / 1_000_000.0
    )

    handle.write(
        f"#{china_origin.year:5d} {china_origin.month:2d} {china_origin.day:2d} "
        f"{china_origin.hour:2d} {china_origin.minute:2d} {china_second:5.2f} "
        f"{event.lat:8.4f} {event.lon:9.4f} {event.dep:7.2f}  {event.mag:5.2f}"
        f"{0.0:6.2f}{0.0:6.2f}{0.0:6.2f} {event_id:10d}\n"
    )
    for pick in picks:
        handle.write(
            f"{pick.station:<7}  {pick.travel_time:8.2f}  "
            f"{pick.output_weight:5.2f}  {pick.family}\n"
        )


def flush_event(handle, event: Optional[Event], raw_picks: Sequence[Pick],
                next_event_id: int, cfg: Config, stats: Stats) -> int:
    if event is None:
        return next_event_id

    selected = select_first_arrivals(raw_picks, cfg, stats)
    if not selected:
        stats.events_without_picks += 1
        if not cfg.write_events_without_picks:
            return next_event_id

    write_event_block(handle, event, next_event_id, selected)
    stats.events_written += 1
    stats.picks_written += len(selected)
    return next_event_id + 1


# =============================================================================
# 文件处理
# =============================================================================


def process_dbo_file(path: Path, encoding: str, output_handle,
                     next_event_id: int, cfg: Config, stats: Stats) -> int:
    current_event: Optional[Event] = None
    current_picks: List[Pick] = []
    source_file = str(path)

    with path.open("r", encoding=encoding, errors="ignore") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue

            record_type = stripped.split(maxsplit=1)[0]
            if record_type == "DBO":
                next_event_id = flush_event(
                    output_handle, current_event, current_picks,
                    next_event_id, cfg, stats
                )
                current_picks = []
                current_event = parse_dbo_event_line(
                    stripped, source_file, line_no, stats
                )
                if current_event is not None:
                    stats.events_read += 1
                continue

            if record_type == "DPB" and current_event is not None:
                pick = parse_dpb_pick(
                    stripped, current_event, cfg, stats, source_file, line_no
                )
                if pick is not None:
                    current_picks.append(pick)

    return flush_event(
        output_handle, current_event, current_picks,
        next_event_id, cfg, stats
    )


def process_jopens_file(path: Path, encoding: str, output_handle,
                        next_event_id: int, cfg: Config, stats: Stats) -> int:
    current_event: Optional[Event] = None
    current_picks: List[Pick] = []
    previous_station = ("", "")
    previous_j2 = False
    source_file = str(path)

    with path.open("r", encoding=encoding, errors="ignore") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            slash = len(line) >= 8 and line[7] == "/"
            region_nonblank = (
                any(char != " " for char in line[25:42])
                if len(line) >= 42 else False
            )

            if slash:
                previous_j2 = region_nonblank

            if slash and previous_j2:
                next_event_id = flush_event(
                    output_handle, current_event, current_picks,
                    next_event_id, cfg, stats
                )
                current_picks = []
                previous_station = ("", "")
                current_event = parse_jopens_event_line(
                    line, source_file, line_no, stats
                )
                if current_event is not None:
                    stats.events_read += 1
                continue

            if (not slash and previous_j2 and region_nonblank
                    and current_event is not None):
                pick, previous_station = parse_jopens_pick(
                    line, current_event, previous_station,
                    cfg, stats, source_file, line_no
                )
                if pick is not None:
                    current_picks.append(pick)

    return flush_event(
        output_handle, current_event, current_picks,
        next_event_id, cfg, stats
    )


# =============================================================================
# 日志与命令行
# =============================================================================


def write_conversion_log(log_path: Path, cfg: Config,
                         input_files: Sequence[Path], stats: Stats) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("tomoDD/hypoDD phase conversion QC log\n")
        handle.write("===================================\n")
        handle.write("Time basis: input = China Standard Time (UTC+8); event output = China Standard Time (UTC+8)\n")
        handle.write("Station label: network + station (maximum 2 + 5 characters)\n")
        handle.write("Selection: earliest P-family and earliest S-family per event/station\n")
        handle.write("P-family: Pg, Pn, P; S-family: Sg, Sn, S\n")
        handle.write("Excluded: PmP, SmS, SME, SMN and other non-target records\n")
        handle.write(f"Enforce S after P: {cfg.enforce_s_after_p}\n")
        handle.write(f"Travel-time range: {cfg.min_travel_time} to {cfg.max_travel_time} s\n")
        handle.write(f"Start event id: {cfg.start_event_id}\n")
        handle.write(f"Output file: {cfg.output_file}\n")
        handle.write("\nInput files:\n")
        for path in input_files:
            handle.write(f"  {path}\n")

        handle.write("\nSummary:\n")
        for key, value in (
            ("files_requested", stats.files_requested),
            ("files_processed", stats.files_processed),
            ("files_missing", stats.files_missing),
            ("files_unknown_format", stats.files_unknown_format),
            ("events_read", stats.events_read),
            ("events_written", stats.events_written),
            ("events_without_picks", stats.events_without_picks),
            ("raw_valid_picks_before_selection", stats.raw_valid_picks),
            ("picks_written_after_selection", stats.picks_written),
        ):
            handle.write(f"  {key}: {value}\n")

        def write_counter(title: str, counter: Counter) -> None:
            handle.write(f"\n{title}:\n")
            if not counter:
                handle.write("  none\n")
                return
            for key, value in counter.most_common():
                handle.write(f"  {key}: {value}\n")

        write_counter("detected_formats", stats.formats)
        write_counter("encodings", stats.encodings)
        write_counter("dbo_variants", stats.dbo_variants)
        write_counter("input_phase_counts", stats.phase_in)
        write_counter("kept_original_phase_counts", stats.phase_kept)
        write_counter("duplicate_groups", stats.duplicate_groups)
        write_counter("duplicate_picks_removed", stats.duplicate_picks_removed)
        write_counter("skipped_or_corrected_counts", stats.skipped)
        write_counter("station_truncations", stats.station_truncations)

        collisions = {
            label: originals
            for label, originals in stats.station_collision_map.items()
            if len(originals) > 1
        }
        handle.write("\nstation_label_collisions_after_truncation:\n")
        if not collisions:
            handle.write("  none\n")
        else:
            for label, originals in sorted(collisions.items()):
                handle.write(f"  {label}: {sorted(originals)}\n")

        handle.write("\nExamples:\n")
        if not stats.examples:
            handle.write("  none\n")
        else:
            for key, values in stats.examples.items():
                handle.write(f"[{key}]\n")
                for value in values:
                    handle.write(f"  {value}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert old/new DBO-DPB or JOPENS reports to hypoDD/tomoDD phase.dat, "
            "keeping the earliest P and S arrival per event/station."
        )
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="Input files or glob patterns. Defaults to INPUT_FILES in the script.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output phase file. Defaults to OUTPUT_FILE in the script.",
    )
    parser.add_argument(
        "--start-id", type=int, default=None,
        help="Starting event id. Defaults to START_EVENT_ID in the script.",
    )
    parser.add_argument(
        "--max-travel-time", type=float, default=None,
        help="Maximum accepted travel time in seconds.",
    )
    parser.add_argument(
        "--allow-long-station-code", action="store_true",
        help=(
            "Truncate network/station to 2+5 characters instead of rejecting long codes. "
            "Any truncation/collision is written to the QC log."
        ),
    )
    parser.add_argument(
        "--no-ps-order-check", action="store_true",
        help="Do not enforce S arrival later than P at the same station.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_patterns = list(args.inputs) if args.inputs else list(INPUT_FILES)
    output_file = Path(args.output if args.output is not None else OUTPUT_FILE).expanduser()
    start_event_id = args.start_id if args.start_id is not None else START_EVENT_ID
    max_travel_time = (
        args.max_travel_time
        if args.max_travel_time is not None else MAX_TRAVEL_TIME
    )

    if start_event_id < 0:
        raise ValueError("start event id must be >= 0")
    if max_travel_time <= MIN_TRAVEL_TIME:
        raise ValueError("max travel time must be larger than min travel time")

    cfg = Config(
        input_patterns=input_patterns,
        output_file=output_file.resolve(),
        start_event_id=start_event_id,
        max_travel_time=max_travel_time,
        strict_station_length=not args.allow_long_station_code,
        enforce_s_after_p=not args.no_ps_order_check,
    )

    input_files = expand_input_patterns(input_patterns, cfg.output_file)
    stats = Stats(files_requested=len(input_files))
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)

    next_event_id = cfg.start_event_id
    with cfg.output_file.open("w", encoding="utf-8", newline="\n") as output_handle:
        for path in input_files:
            if not path.exists() or not path.is_file():
                stats.files_missing += 1
                print(f"[WARNING] Missing input file: {path}", file=sys.stderr)
                continue

            encoding = detect_text_encoding(path)
            file_format = detect_file_format(path, encoding)
            if file_format is None:
                stats.files_unknown_format += 1
                print(f"[WARNING] Unknown file format: {path}", file=sys.stderr)
                continue

            stats.files_processed += 1
            stats.formats[file_format] += 1
            stats.encodings[encoding] += 1
            print(f"[INFO] Processing {path} as {file_format}, encoding={encoding}")

            if file_format == "DBO":
                next_event_id = process_dbo_file(
                    path, encoding, output_handle,
                    next_event_id, cfg, stats
                )
            else:
                next_event_id = process_jopens_file(
                    path, encoding, output_handle,
                    next_event_id, cfg, stats
                )

    log_path = Path(str(cfg.output_file) + ".convert.log")
    write_conversion_log(log_path, cfg, input_files, stats)

    print("[DONE] Conversion completed")
    print("  event_header_time_system          = China Standard Time (UTC+8)")
    print(f"  files_processed                  = {stats.files_processed}")
    print(f"  events_read                     = {stats.events_read}")
    print(f"  events_written                  = {stats.events_written}")
    print(f"  raw_valid_picks_before_selection= {stats.raw_valid_picks}")
    print(f"  picks_written_after_selection   = {stats.picks_written}")
    print(f"  output                           = {cfg.output_file}")
    print(f"  QC log                           = {log_path}")

    if stats.files_processed == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
