from literature_kb.terminology import rewrite_query


def test_long_phrase_match_is_explainable_and_avoids_nested_double_expansion():
    rewritten, matches = rewrite_query("图罗夫的基里尔与古代哲学的关系")
    assert "Кирилл" in rewritten and "античная" in rewritten
    assert len(matches) == 2
    assert {m["source"] for m in matches} == {"图罗夫的基里尔", "古代哲学"}


def test_no_match_keeps_original_and_does_not_invent_translation():
    query = "unrelated unknown user question"
    assert rewrite_query(query) == (query, [])


def test_expansion_is_bounded_and_title_variants_share_the_same_target():
    assert rewrite_query("伊戈尔远征记")[0] == rewrite_query("伊戈里远征记")[0]
    _, matches = rewrite_query(
        "图罗夫基里尔 莫诺马赫 拜占庭 基辅 索菲亚 修道院 编年史 体裁 训诫 音乐 建筑"
    )
    assert len(matches) == 8
