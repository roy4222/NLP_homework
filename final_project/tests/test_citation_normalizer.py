from scripts.citation_normalizer import extract_citations


def test_extracts_arabic_article_with_subarticle():
    text = "依刑法第185條之3，酒後駕車可能構成公共危險罪。"
    assert extract_citations(text) == {("刑法", "185-3")}


def test_extracts_chinese_numeral_article():
    text = "民法第一百八十四條規定侵權行為損害賠償責任。"
    assert extract_citations(text) == {("民法", "184")}


def test_normalizes_law_alias_and_spaces():
    text = "依照 中華民國刑法 第 320 條，竊取他人動產者成立竊盜罪。"
    assert extract_citations(text) == {("刑法", "320")}


def test_deduplicates_repeated_articles():
    text = "刑法第339條是詐欺罪。刑法第339條也常見於詐騙案件。"
    assert extract_citations(text) == {("刑法", "339")}
