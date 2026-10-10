from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


EVALUATOR_PROTOCOL_VERSION = "travelplanner-local-eval-v2"


@dataclass
class DayPlan:
    day_number: int = 0
    current_city: str = ""
    from_city: str = ""
    to_city: str = ""
    transportation: str = ""
    flight_number: str = ""
    breakfast: str = ""
    lunch: str = ""
    dinner: str = ""
    attraction: str = ""
    accommodation: str = ""


@dataclass
class ParsedPlan:
    days: list[DayPlan] = field(default_factory=list)
    raw_text: str = ""
    parse_success: bool = False


@dataclass
class ConstraintResult:
    is_not_absent: bool = False
    is_valid_in_sandbox: bool = False
    is_valid_in_current_city: bool = True
    is_valid_restaurants: bool = False
    is_valid_attractions: bool = False
    is_valid_accommodation: bool = False
    is_valid_transportation: bool = False
    valid_cost: bool | None = None
    valid_room_rule: bool | None = None
    valid_cuisine: bool | None = None
    valid_room_type: bool | None = None
    valid_transportation_type: bool | None = None
    commonsense_pass: bool = False
    hard_pass: bool = False
    final_pass: bool = False
    absent_fields: list[str] = field(default_factory=list)
    sandbox_violations: list[str] = field(default_factory=list)
    duplicate_restaurants: list[str] = field(default_factory=list)
    duplicate_attractions: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    budget: float = 0.0
    unavailable_identifier_mentions: list[str] = field(default_factory=list)
    unavailable_identifier_hits: list[str] = field(default_factory=list)
    uses_unavailable_identifier: bool = False


def evaluate_plan(
    plan_text: str,
    ref_info: dict[str, Any],
    query_data: dict[str, Any],
    unavailable_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Parse and score a TravelPlanner response with dependency-free rules."""
    parsed = parse_plan(plan_text)
    result = check_constraints(parsed, ref_info, query_data)
    unavailable = unavailable_ids or extract_unavailable_ids(ref_info)
    result.unavailable_identifier_mentions = find_identifier_hits(plan_text, unavailable)
    result.unavailable_identifier_hits = find_selected_identifier_hits(parsed, unavailable)
    result.uses_unavailable_identifier = bool(result.unavailable_identifier_hits)
    data = asdict(result)
    data["parse_success"] = parsed.parse_success
    data["day_count"] = len(parsed.days)
    data["unavailable_id_count"] = len(unavailable)
    data["evaluator_protocol_version"] = EVALUATOR_PROTOCOL_VERSION
    data["unavailable_usage_semantics"] = "selected_itinerary_fields"
    return data


def parse_plan(plan_text: str) -> ParsedPlan:
    if not plan_text or len(plan_text.strip()) < 20:
        return ParsedPlan(raw_text=plan_text or "", parse_success=False)

    json_days = _parse_json_plan(plan_text)
    if json_days:
        return ParsedPlan(days=json_days, raw_text=plan_text, parse_success=True)

    normalized_text = _normalize_text_plan(plan_text)
    days: list[DayPlan] = []
    day_blocks = re.split(
        r"(?im)^\s*(?:\d+[.)]\s*)?Day\s+(\d+)\b"
        r"(?:\s*(?:[:\-(\u2010-\u2015][^\n]*))?[ \t]*$",
        normalized_text,
    )
    for i in range(1, len(day_blocks) - 1, 2):
        try:
            day_num = int(day_blocks[i])
        except ValueError:
            continue
        days.append(_parse_day_block(day_blocks[i + 1], day_num))
    return ParsedPlan(days=days, raw_text=plan_text, parse_success=bool(days))


def _parse_json_plan(plan_text: str) -> list[DayPlan]:
    for candidate in _json_candidates(plan_text):
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        rows = _json_day_rows(payload)
        if not rows:
            continue
        days = [_day_from_mapping(row, fallback_day=idx) for idx, row in enumerate(rows, 1)]
        days = [day for day in days if day.day_number > 0]
        if days:
            return days
    return []


def _json_candidates(plan_text: str) -> list[str]:
    text = plan_text.strip()
    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
    )
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _json_day_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    normalized = {_normalized_key(key): value for key, value in payload.items()}
    for key in ("plan", "travelplan", "itinerary", "days", "schedule"):
        value = normalized.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            nested = _json_day_rows(value)
            if nested:
                return nested

    keyed_days: list[dict[str, Any]] = []
    for key, value in payload.items():
        match = re.fullmatch(r"\s*day\s*(\d+)\s*", str(key), re.I)
        if match and isinstance(value, dict):
            row = dict(value)
            row.setdefault("day", int(match.group(1)))
            keyed_days.append(row)
    if keyed_days:
        return keyed_days
    if any(key in normalized for key in ("day", "daynumber", "currentcity")):
        return [payload]
    return []


def _day_from_mapping(row: dict[str, Any], *, fallback_day: int) -> DayPlan:
    values = {_normalized_key(key): value for key, value in row.items()}
    day = DayPlan(
        day_number=_day_number(values.get("day") or values.get("daynumber"), fallback_day)
    )
    city_text = _plan_value(values.get("currentcity") or values.get("city"))
    from_to = re.match(r"from\s+(.+?)\s+to\s+(.+)", city_text, re.I)
    if from_to:
        day.from_city = from_to.group(1).strip()
        day.to_city = from_to.group(2).strip()
        day.current_city = day.to_city
    else:
        day.current_city = city_text
    day.transportation = _plan_value(values.get("transportation") or values.get("transport"))
    flight_match = re.search(r"F\d{5,}", day.transportation)
    if flight_match:
        day.flight_number = flight_match.group(0)
    day.breakfast = _plan_value(values.get("breakfast"))
    day.lunch = _plan_value(values.get("lunch"))
    day.dinner = _plan_value(values.get("dinner"))
    day.attraction = _plan_value(
        values.get("attraction") or values.get("attractions"), separator=";"
    )
    day.accommodation = _plan_value(
        values.get("accommodation") or values.get("hotel") or values.get("lodging")
    )
    return day


def _normalize_text_plan(plan_text: str) -> str:
    text = plan_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```(?:[a-zA-Z0-9_-]+)?", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    labels = (
        "Day\\s+\\d+|Current\\s+City|Transportation|Breakfast|Attractions?|"
        "Lunch|Dinner|Accommodation"
    )
    text = re.sub(rf"(?im)^\s*[-+*]\s+(?=(?:{labels})\b)", "", text)
    text = text.replace("\uff1a", ":")
    return text


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _day_number(value: Any, fallback: int) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else fallback


def _plan_value(value: Any, *, separator: str = ", ") -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return separator.join(_plan_value(item) for item in value if _plan_value(item))
    if isinstance(value, dict):
        for key in ("name", "value", "description"):
            if key in value:
                return _plan_value(value[key])
        return ", ".join(f"{key}: {_plan_value(item)}" for key, item in value.items())
    return str(value).strip().strip("*_")


def check_constraints(
    parsed: ParsedPlan,
    ref_info: dict[str, Any],
    query_data: dict[str, Any],
) -> ConstraintResult:
    result = ConstraintResult()
    if not parsed.parse_success or not parsed.days:
        return result

    catalog = _build_catalog(ref_info)
    _check_absent_fields(parsed, result)
    _check_sandbox(parsed, catalog, result)
    _check_duplicates_and_transport(parsed, result)
    _check_cost(parsed, catalog, query_data, result)
    _check_local_constraints(parsed, catalog, query_data, result)

    commonsense_checks = [
        result.is_not_absent,
        result.is_valid_in_sandbox,
        result.is_valid_in_current_city,
        result.is_valid_restaurants,
        result.is_valid_attractions,
        result.is_valid_accommodation,
        result.is_valid_transportation,
    ]
    result.commonsense_pass = all(commonsense_checks)
    hard_checks = [
        v
        for v in [
            result.valid_cost,
            result.valid_room_rule,
            result.valid_cuisine,
            result.valid_room_type,
            result.valid_transportation_type,
        ]
        if v is not None
    ]
    result.hard_pass = all(hard_checks) if hard_checks else True
    result.final_pass = result.commonsense_pass and result.hard_pass
    return result


def extract_unavailable_ids(ref_info: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key, value in ref_info.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict) or not item.get("Status"):
                continue
            if key.startswith("Flight"):
                ids.append(str(item.get("Flight Number", "")).strip())
            elif key.startswith("Accommodations"):
                ids.append(str(item.get("NAME", "")).strip())
            elif key.startswith("Restaurants") or key.startswith("Attractions"):
                ids.append(str(item.get("Name", "")).strip())
    return sorted({item for item in ids if item})


def find_identifier_hits(text: str, identifiers: list[str]) -> list[str]:
    haystack = _normalize(text).casefold()
    hits = []
    for identifier in identifiers:
        normalized = _normalize(identifier).casefold()
        if normalized and normalized in haystack:
            hits.append(identifier)
    return sorted(set(hits))


def find_selected_identifier_hits(
    parsed: ParsedPlan,
    identifiers: list[str],
) -> list[str]:
    """Return unavailable identifiers selected in structured itinerary fields.

    Full-response matching is retained separately as mention telemetry. Selection
    must be derived from parsed fields so an explanation such as "F123 was
    cancelled; use F456" does not count F123 as an attempted resource use.
    """
    selected = [
        value
        for day in parsed.days
        for value in _selected_day_identifiers(day)
        if _normalize(value)
    ]
    return sorted(
        {
            identifier
            for identifier in identifiers
            if any(_selected_field_matches_identifier(value, identifier) for value in selected)
        }
    )


def _selected_day_identifiers(day: DayPlan) -> list[str]:
    values = [
        day.flight_number,
        day.breakfast,
        day.lunch,
        day.dinner,
        day.accommodation,
    ]
    values.extend(part.strip() for part in re.split(r"[;]", day.attraction))
    return values


def _selected_field_matches_identifier(value: str, identifier: str) -> bool:
    selected = _normalize(value).casefold()
    target = _normalize(identifier).casefold()
    if not target or not selected.startswith(target):
        return False
    if selected == target:
        return True
    return selected[len(target)] in {",", "(", "[", "-", "\u2013", "\u2014", ":"}


def _parse_day_block(block: str, day_number: int) -> DayPlan:
    day = DayPlan(day_number=day_number)
    city_match = re.search(r"(?im)^\s*Current\s+City\s*:\s*(.+?)\s*$", block)
    if city_match:
        city_text = city_match.group(1).strip()
        from_to = re.match(r"from\s+(.+?)\s+to\s+(.+)", city_text, re.IGNORECASE)
        if from_to:
            day.from_city = from_to.group(1).strip()
            day.to_city = from_to.group(2).strip()
            day.current_city = day.to_city
        else:
            day.current_city = city_text

    trans_match = re.search(r"(?im)^\s*Transportation\s*:\s*(.+?)\s*$", block)
    if trans_match:
        day.transportation = trans_match.group(1).strip()
        flight_match = re.search(r"F\d{5,}", day.transportation)
        if flight_match:
            day.flight_number = flight_match.group(0)

    for attr, pattern in [
        ("breakfast", r"(?im)^\s*Breakfast\s*:\s*(.+?)\s*$"),
        ("lunch", r"(?im)^\s*Lunch\s*:\s*(.+?)\s*$"),
        ("dinner", r"(?im)^\s*Dinner\s*:\s*(.+?)\s*$"),
        ("attraction", r"(?im)^\s*Attractions?\s*:\s*(.+?)\s*$"),
        ("accommodation", r"(?im)^\s*Accommodation\s*:\s*(.+?)\s*$"),
    ]:
        match = re.search(pattern, block)
        if match:
            setattr(day, attr, match.group(1).strip())
    return day


def _build_catalog(ref_info: dict[str, Any]) -> dict[str, Any]:
    catalog: dict[str, Any] = {
        "valid_flights": set(),
        "valid_accommodations": set(),
        "valid_restaurants": set(),
        "valid_attractions": set(),
        "flight_prices": {},
        "accommodation_prices": {},
        "restaurant_costs": {},
        "accommodation_rules": {},
        "accommodation_types": {},
        "restaurant_cuisines": {},
    }
    for key, value in ref_info.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict) or item.get("Status"):
                continue
            if key.startswith("Flight"):
                flight_number = str(item.get("Flight Number", "")).strip()
                if flight_number:
                    catalog["valid_flights"].add(flight_number)
                    catalog["flight_prices"][flight_number] = _to_float(item.get("Price"), 0.0)
            elif key.startswith("Accommodations"):
                name = str(item.get("NAME", "")).strip()
                if name:
                    catalog["valid_accommodations"].add(name)
                    catalog["accommodation_prices"][name] = _to_float(item.get("price"), 0.0)
                    catalog["accommodation_rules"][name] = str(item.get("house_rules", ""))
                    catalog["accommodation_types"][name] = str(item.get("room type", ""))
            elif key.startswith("Restaurants"):
                name = str(item.get("Name", "")).strip()
                if name:
                    catalog["valid_restaurants"].add(name)
                    catalog["restaurant_costs"][name] = _to_float(item.get("Average Cost"), 0.0)
                    catalog["restaurant_cuisines"][name] = str(item.get("Cuisines", ""))
            elif key.startswith("Attractions"):
                name = str(item.get("Name", "")).strip()
                if name:
                    catalog["valid_attractions"].add(name)
    return catalog


def _check_absent_fields(parsed: ParsedPlan, result: ConstraintResult) -> None:
    absent = []
    for day in parsed.days:
        is_last_day = day == parsed.days[-1]
        if not day.current_city:
            absent.append(f"Day {day.day_number}: missing current city")
        for field_name in ["breakfast", "lunch", "dinner", "attraction"]:
            value = getattr(day, field_name)
            if not value or value == "-":
                absent.append(f"Day {day.day_number}: missing {field_name}")
        if not is_last_day and (not day.accommodation or day.accommodation == "-"):
            absent.append(f"Day {day.day_number}: missing accommodation")
    result.is_not_absent = not absent
    result.absent_fields = absent


def _check_sandbox(parsed: ParsedPlan, catalog: dict[str, Any], result: ConstraintResult) -> None:
    violations = []
    for day in parsed.days:
        if day.flight_number and day.flight_number not in catalog["valid_flights"]:
            violations.append(f"Day {day.day_number}: flight {day.flight_number} not available")
        for meal_name, meal_value in [
            ("breakfast", day.breakfast),
            ("lunch", day.lunch),
            ("dinner", day.dinner),
        ]:
            name = _item_name(meal_value)
            if name and name not in catalog["valid_restaurants"]:
                violations.append(f"Day {day.day_number}: {meal_name} '{name}' not available")
        name = _item_name(day.accommodation)
        if name and name not in catalog["valid_accommodations"]:
            violations.append(f"Day {day.day_number}: accommodation '{name}' not available")
        for attraction in _split_attractions(day.attraction):
            if attraction and attraction not in catalog["valid_attractions"]:
                violations.append(f"Day {day.day_number}: attraction '{attraction}' not available")
    result.is_valid_in_sandbox = not violations
    result.sandbox_violations = violations


def _check_duplicates_and_transport(parsed: ParsedPlan, result: ConstraintResult) -> None:
    restaurants = [
        _item_name(meal)
        for day in parsed.days
        for meal in [day.breakfast, day.lunch, day.dinner]
        if _item_name(meal)
    ]
    attractions = [
        attraction
        for day in parsed.days
        for attraction in _split_attractions(day.attraction)
        if attraction
    ]
    result.duplicate_restaurants = sorted(
        {name for name in restaurants if restaurants.count(name) > 1}
    )
    result.duplicate_attractions = sorted(
        {name for name in attractions if attractions.count(name) > 1}
    )
    result.is_valid_restaurants = not result.duplicate_restaurants
    result.is_valid_attractions = not result.duplicate_attractions

    accommodations_by_city: dict[str, set[str]] = {}
    for day in parsed.days[:-1]:
        accommodation = _item_name(day.accommodation)
        if not accommodation:
            continue
        city = day.current_city or day.to_city or "unknown"
        accommodations_by_city.setdefault(city, set()).add(accommodation)
    result.is_valid_accommodation = bool(accommodations_by_city) and all(
        len(names) == 1 for names in accommodations_by_city.values()
    )

    first_transport = bool(parsed.days[0].transportation and parsed.days[0].transportation != "-")
    last_transport = (
        bool(parsed.days[-1].transportation and parsed.days[-1].transportation != "-")
        if len(parsed.days) > 1
        else True
    )
    result.is_valid_transportation = first_transport and last_transport


def _check_cost(
    parsed: ParsedPlan,
    catalog: dict[str, Any],
    query_data: dict[str, Any],
    result: ConstraintResult,
) -> None:
    budget = _to_float(query_data.get("budget"), 0.0)
    people = int(_to_float(query_data.get("people_number"), 1.0) or 1)
    result.budget = budget
    if budget <= 0:
        return

    total = 0.0
    for day in parsed.days:
        if day.flight_number:
            total += catalog["flight_prices"].get(day.flight_number, 0.0) * people
        for meal in [day.breakfast, day.lunch, day.dinner]:
            name = _item_name(meal)
            if name:
                total += catalog["restaurant_costs"].get(name, 0.0) * people
        accommodation = _item_name(day.accommodation)
        if accommodation:
            total += catalog["accommodation_prices"].get(accommodation, 0.0)
    result.estimated_cost = total
    result.valid_cost = total <= budget


def _check_local_constraints(
    parsed: ParsedPlan,
    catalog: dict[str, Any],
    query_data: dict[str, Any],
    result: ConstraintResult,
) -> None:
    local_constraint = _parse_local_constraint(query_data.get("local_constraint"))

    required_cuisines = _as_list(local_constraint.get("cuisine"))
    if required_cuisines:
        found = set()
        for day in parsed.days:
            for meal in [day.breakfast, day.lunch, day.dinner]:
                cuisine_text = catalog["restaurant_cuisines"].get(_item_name(meal), "").casefold()
                for cuisine in required_cuisines:
                    if cuisine.casefold() in cuisine_text:
                        found.add(cuisine)
        result.valid_cuisine = all(cuisine in found for cuisine in required_cuisines)

    required_room_types = _as_list(local_constraint.get("room type"))
    if required_room_types:
        result.valid_room_type = True
        for day in parsed.days[:-1]:
            accommodation = _item_name(day.accommodation)
            if not accommodation:
                continue
            actual = catalog["accommodation_types"].get(accommodation, "").casefold()
            if not any(room.casefold() in actual for room in required_room_types):
                result.valid_room_type = False
                break

    required_room_rules = _as_list(local_constraint.get("house rule"))
    if required_room_rules:
        result.valid_room_rule = True
        for day in parsed.days[:-1]:
            accommodation = _item_name(day.accommodation)
            rules = catalog["accommodation_rules"].get(accommodation, "").casefold()
            if _violates_room_rules(required_room_rules, rules):
                result.valid_room_rule = False
                break

    required_transport = _as_list(local_constraint.get("transportation"))
    if required_transport:
        result.valid_transportation_type = True
        required = [item.casefold() for item in required_transport]
        for day in parsed.days:
            transport = day.transportation.casefold()
            if "no flight" in required and "flight" in transport:
                result.valid_transportation_type = False
            if "no self-driving" in required and "self-driving" in transport:
                result.valid_transportation_type = False


def _parse_local_constraint(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _violates_room_rules(required_rules: list[str], actual_rules: str) -> bool:
    required = [rule.casefold().removeprefix("no ").strip() for rule in required_rules]
    return any(
        [
            "smoking" in required and "no smoking" in actual_rules,
            "pets" in required and "no pets" in actual_rules,
            "parties" in required and "no parties" in actual_rules,
            "children under 10" in required and "no children under 10" in actual_rules,
            "visitors" in required and "no visitors" in actual_rules,
        ]
    )


def _split_attractions(text: str) -> list[str]:
    if not text or text == "-":
        return []
    return [_item_name(part) for part in re.split(r"[;]", text) if _item_name(part)]


def _item_name(text: str) -> str:
    if not text or text == "-":
        return ""
    return text.split(",")[0].strip().rstrip(".")


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
