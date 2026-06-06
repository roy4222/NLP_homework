from pathlib import Path

import yaml


def test_issues_yaml_has_valid_minimum_shape():
    issues = yaml.safe_load(Path("data/issues.yaml").read_text(encoding="utf-8"))
    assert 15 <= len(issues) <= 25
    ids = [issue["id"] for issue in issues]
    assert len(ids) == len(set(ids))
    for issue in issues:
        assert issue["id"]
        assert issue["name_zh"]
        assert issue["name_en"]
        assert issue["category"] in {"criminal", "civil"}
        assert issue["laws"]
        assert issue["aliases"]
        assert issue["keywords"]
        for law in issue["laws"]:
            assert set(law) == {"law", "article"}
