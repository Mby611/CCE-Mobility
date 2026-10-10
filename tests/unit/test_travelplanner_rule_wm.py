from cce_mobility.benchmarks.travelplanner_eval import extract_unavailable_ids
from cce_mobility.benchmarks.travelplanner_rule_wm import TravelPlannerRuleWorldModel


def test_rule_wm_prefers_target_flight_for_cancellation():
    ref_info = {
        "Flight from A to B on 2022-03-01": [
            {"Flight Number": "F100001", "Price": 50},
            {"Flight Number": "F100002", "Price": 10},
        ]
    }

    wm = TravelPlannerRuleWorldModel(
        "flight_cancelled",
        params={"target_ids": ["F100001"]},
    )
    modified = wm.intervene_ref_info(ref_info)

    assert extract_unavailable_ids(modified) == ["F100001"]
    assert wm.records[0].metadata["target_ids"] == ["F100001"]


def test_rule_wm_skips_key_when_target_flight_absent():
    ref_info = {
        "Flight from A to B on 2022-03-01": [
            {"Flight Number": "F100001", "Price": 50},
            {"Flight Number": "F100002", "Price": 10},
        ]
    }

    wm = TravelPlannerRuleWorldModel(
        "flight_cancelled",
        params={"target_ids": ["F999999"]},
    )
    modified = wm.intervene_ref_info(ref_info)

    assert extract_unavailable_ids(modified) == []
    assert wm.records[0].triggered is False
