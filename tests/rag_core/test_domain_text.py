"""rag_core.domain.text: clean_text and extract_year."""

from rag_core.domain.text import clean_text, extract_year


def test_extract_year():
    assert extract_year("https://x/haushalt-2024/jahresbericht-der-universitaet-beispiel-2024/") == 2024
    assert extract_year(".../richtlinie-muster-verfahren-und-pruefung-der-posten2031-07/") == 2031
    assert extract_year("mitteilungsblatt-52-09", "Mitteilungsblatt 2021") == 2021
    assert extract_year("xy-zq-v3-11-27") is None
    assert extract_year("Tel 0000554433") is None
    assert extract_year("x 1949", "y 2050 und 2024") == 2024
    assert extract_year("", None) is None


def test_clean_text_strips_tags_only_when_asked():
    assert clean_text("<p>Öffnungszeiten<br />Mo&ndash;Fr</p>") == "Öffnungszeiten Mo–Fr"
    assert clean_text("< Naturschutz > &amp; <leer>", tags=False) == "< Naturschutz > & <leer>"
    assert clean_text("<!-- x -->a<b>b</b>", tags=False) == "a<b>b</b>"


def test_comments_are_removed_and_blank_lines_tidied():
    assert clean_text("a<!-- 🖼️ Image not available -->  \n\n\n\nb") == "a\n\nb"
