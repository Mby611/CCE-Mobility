from cce_mobility.benchmarks.travelplanner_eval import evaluate_plan, parse_plan


def test_travelplanner_eval_accepts_valid_structured_plan():
    ref_info = {
        "Flight from A to B on 2022-03-01": [
            {"Flight Number": "F100001", "Price": 100, "Status": ""},
        ],
        "Flight from B to A on 2022-03-02": [
            {"Flight Number": "F100002", "Price": 110, "Status": ""},
        ],
        "Restaurants in B": [
            {"Name": f"Restaurant {idx}", "Average Cost": 10, "Cuisines": "Chinese", "Status": ""}
            for idx in range(1, 7)
        ],
        "Attractions in B": [
            {"Name": "Museum", "Status": ""},
            {"Name": "Gallery", "Status": ""},
        ],
        "Accommodations in B": [
            {
                "NAME": "Hotel B",
                "price": 80,
                "room type": "Private room",
                "house_rules": "No smoking",
                "Status": "",
            }
        ],
    }
    query_data = {
        "budget": "1000",
        "people_number": "1",
        "local_constraint": "{'house rule': None, 'cuisine': 'Chinese', 'room type': 'Private room', 'transportation': None}",
    }
    plan = """Travel Plan:
Day 1:
Current City: from A to B
Transportation: Flight Number: F100001, from A to B
Breakfast: Restaurant 1, B
Attraction: Museum, B
Lunch: Restaurant 2, B
Dinner: Restaurant 3, B
Accommodation: Hotel B, B

Day 2:
Current City: from B to A
Transportation: Flight Number: F100002, from B to A
Breakfast: Restaurant 4, B
Attraction: Gallery, B
Lunch: Restaurant 5, B
Dinner: Restaurant 6, B
Accommodation: -
"""

    metrics = evaluate_plan(plan, ref_info, query_data)

    assert metrics["parse_success"] is True
    assert metrics["commonsense_pass"] is True
    assert metrics["hard_pass"] is True
    assert metrics["final_pass"] is True


def test_travelplanner_eval_flags_unavailable_identifier_usage():
    ref_info = {
        "Flight from A to B on 2022-03-01": [
            {"Flight Number": "F100001", "Price": 100, "Status": "CANCELLED"},
            {"Flight Number": "F100003", "Price": 120, "Status": ""},
        ],
        "Restaurants in B": [],
        "Attractions in B": [],
        "Accommodations in B": [],
    }
    plan = """Travel Plan:
Day 1:
Current City: from A to B
Transportation: Flight Number: F100001, from A to B
Breakfast: -
Attraction: -
Lunch: -
Dinner: -
Accommodation: -
"""

    metrics = evaluate_plan(plan, ref_info, {"budget": 1000}, unavailable_ids=["F100001"])

    assert metrics["uses_unavailable_identifier"] is True
    assert metrics["unavailable_identifier_hits"] == ["F100001"]
    assert metrics["unavailable_identifier_mentions"] == ["F100001"]
    assert metrics["evaluator_protocol_version"] == "travelplanner-local-eval-v2"
    assert metrics["unavailable_usage_semantics"] == "selected_itinerary_fields"


def test_travelplanner_eval_does_not_treat_cancelled_resource_notice_as_usage():
    ref_info = {
        "Flight from A to B on 2022-03-01": [
            {"Flight Number": "F100001", "Price": 100, "Status": "CANCELLED"},
            {"Flight Number": "F100003", "Price": 120, "Status": ""},
        ],
        "Restaurants in B": [],
        "Attractions in B": [],
        "Accommodations in B": [],
    }
    plan = """Travel Plan:
Day 1 (March 1, 2022): A to B
Current City: from A to B
Transportation: Flight Number: F100003, from A to B (F100001 was cancelled.)
Breakfast: -
Attraction: -
Lunch: -
Dinner: -
Accommodation: -
"""

    metrics = evaluate_plan(
        plan,
        ref_info,
        {"budget": 1000},
        unavailable_ids=["F100001"],
    )

    assert metrics["parse_success"] is True
    assert metrics["unavailable_identifier_mentions"] == ["F100001"]
    assert metrics["unavailable_identifier_hits"] == []
    assert metrics["uses_unavailable_identifier"] is False


def test_travelplanner_eval_detects_selected_resource_name_containing_commas():
    unavailable = "Large room, close to downtown"
    plan = f"""Travel Plan:
Day 1: March 1, 2022 - A to B
Current City: from A to B
Transportation: Self-driving, from A to B
Breakfast: -
Attraction: -
Lunch: -
Dinner: -
Accommodation: {unavailable}, B (Private room)
"""

    metrics = evaluate_plan(
        plan,
        {},
        {"budget": 1000},
        unavailable_ids=[unavailable],
    )

    assert metrics["unavailable_identifier_mentions"] == [unavailable]
    assert metrics["unavailable_identifier_hits"] == [unavailable]
    assert metrics["uses_unavailable_identifier"] is True


def test_travelplanner_parser_accepts_markdown_plan_without_day_colons():
    plan = """### Travel Plan
**Day 1**
- **Current City:** from A to B
- **Transportation:** Flight Number: F100001, from A to B
- **Breakfast:** Restaurant 1, B
- **Attractions:** Museum, B; Gallery, B
- **Lunch:** Restaurant 2, B
- **Dinner:** Restaurant 3, B
- **Accommodation:** Hotel B, B
"""

    parsed = parse_plan(plan)

    assert parsed.parse_success is True
    assert len(parsed.days) == 1
    assert parsed.days[0].from_city == "A"
    assert parsed.days[0].to_city == "B"
    assert parsed.days[0].attraction == "Museum, B; Gallery, B"


def test_travelplanner_parser_accepts_json_plan_wrappers():
    plan = """```json
{
  "travel_plan": [
    {
      "day": "Day 1",
      "current_city": "from A to B",
      "transportation": "Flight Number: F100001, from A to B",
      "breakfast": "Restaurant 1, B",
      "attractions": ["Museum, B", "Gallery, B"],
      "lunch": "Restaurant 2, B",
      "dinner": "Restaurant 3, B",
      "accommodation": "Hotel B, B"
    }
  ]
}
```"""

    parsed = parse_plan(plan)

    assert parsed.parse_success is True
    assert len(parsed.days) == 1
    assert parsed.days[0].day_number == 1
    assert parsed.days[0].flight_number == "F100001"
    assert parsed.days[0].attraction == "Museum, B;Gallery, B"


def test_travelplanner_parser_accepts_parenthesized_dates_in_day_headers():
    plan = """Day 1 (March 13th):
Current City: from Washington to Myrtle Beach
Transportation: Flight Number: F3927581, from Washington to Myrtle Beach
Breakfast: First Eat, Myrtle Beach
Attraction: SkyWheel Myrtle Beach, Myrtle Beach
Lunch: Nagai, Myrtle Beach
Dinner: Twigly, Myrtle Beach
Accommodation: Hotel A, Myrtle Beach

Day 2 (March 14, 2022):
Current City: Myrtle Beach
Transportation: -
Breakfast: First Eat, Myrtle Beach
Attraction: WonderWorks Myrtle Beach, Myrtle Beach
Lunch: Nagai, Myrtle Beach
Dinner: Twigly, Myrtle Beach
Accommodation: Hotel A, Myrtle Beach
"""

    parsed = parse_plan(plan)

    assert parsed.parse_success is True
    assert [day.day_number for day in parsed.days] == [1, 2]
    assert parsed.days[0].flight_number == "F3927581"
    assert parsed.days[1].current_city == "Myrtle Beach"


def test_travelplanner_parser_accepts_dash_separated_dates_in_day_headers():
    plan = """Travel Plan:
Day 1 \u2014 March 13, 2022:
Current City: from Washington to Myrtle Beach
Transportation: Flight Number: F3927581, from Washington to Myrtle Beach
Breakfast: -
Attraction: SkyWheel Myrtle Beach, Myrtle Beach
Lunch: First Eat, Myrtle Beach
Dinner: Nagai, Myrtle Beach
Accommodation: Hotel A, Myrtle Beach

Day 2 - March 14, 2022:
Current City: Myrtle Beach
Transportation: -
Breakfast: First Eat, Myrtle Beach
Attraction: WonderWorks Myrtle Beach, Myrtle Beach
Lunch: Nagai, Myrtle Beach
Dinner: Twigly, Myrtle Beach
Accommodation: Hotel A, Myrtle Beach
"""

    parsed = parse_plan(plan)

    assert parsed.parse_success is True
    assert [day.day_number for day in parsed.days] == [1, 2]
    assert parsed.days[0].flight_number == "F3927581"
    assert parsed.days[1].current_city == "Myrtle Beach"


def test_travelplanner_parser_accepts_route_suffix_after_parenthesized_date():
    plan = """Day 1 (March 15, 2022): St. Louis to Orlando
Current City: from St. Louis to Orlando
Transportation: Flight Number: F4003646, from St. Louis to Orlando
Breakfast: First Eat, Orlando
Attraction: SeaWorld Orlando, Orlando
Lunch: Second Eat, Orlando
Dinner: Third Eat, Orlando
Accommodation: Hotel A, Orlando

Day 2 (March 16, 2022): Orlando
Current City: Orlando
Transportation: -
Breakfast: Fourth Eat, Orlando
Attraction: Universal Orlando Resort, Orlando
Lunch: Fifth Eat, Orlando
Dinner: Sixth Eat, Orlando
Accommodation: -
"""

    parsed = parse_plan(plan)

    assert parsed.parse_success is True
    assert [day.day_number for day in parsed.days] == [1, 2]
    assert parsed.days[0].flight_number == "F4003646"


def test_travelplanner_parser_accepts_date_and_route_after_day_colon():
    plan = """***** Travel Plan *****

**Day 1: March 2, 2022 (Orlando to Belleville)**
* **Current City:** from Orlando to Belleville
* **Transportation:** Self-driving from Orlando to Belleville
* **Breakfast:** First Eat, Belleville
* **Attraction:** First Museum, Belleville
* **Lunch:** Second Eat, Belleville
* **Dinner:** Third Eat, Belleville
* **Accommodation:** Hotel A, Belleville

### Day 2: March 3, 2022 - Belleville
Current City: Belleville
Transportation: -
Breakfast: Fourth Eat, Belleville
Attraction: Second Museum, Belleville
Lunch: Fifth Eat, Belleville
Dinner: Sixth Eat, Belleville
Accommodation: -
"""

    parsed = parse_plan(plan)

    assert parsed.parse_success is True
    assert [day.day_number for day in parsed.days] == [1, 2]
    assert parsed.days[0].to_city == "Belleville"
