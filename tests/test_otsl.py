from src.data.otsl import html_to_otsl, otsl_to_html


def test_otsl_to_html_simple_2x2():
    otsl = "<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>"
    result = otsl_to_html(otsl)
    assert "<table>" in result
    assert "<td>A</td>" in result
    assert "<td>B</td>" in result
    assert "<td>C</td>" in result
    assert "<td>D</td>" in result
    assert result.count("<tr>") == 2


def test_otsl_to_html_with_empty_cells():
    otsl = "<fcel>A<ecel><nl><ecel><fcel>B<nl>"
    result = otsl_to_html(otsl)
    assert "<td>A</td>" in result
    assert "<td></td>" in result
    assert "<td>B</td>" in result


def test_otsl_to_html_with_horizontal_merge():
    otsl = "<fcel>A<lcel><nl><fcel>B<fcel>C<nl>"
    result = otsl_to_html(otsl)
    assert 'colspan="2"' in result
    assert ">A</td>" in result


def test_otsl_to_html_with_vertical_merge():
    otsl = "<fcel>A<fcel>B<nl><ucel><fcel>C<nl>"
    result = otsl_to_html(otsl)
    assert 'rowspan="2"' in result


def test_otsl_to_html_with_cross_merge():
    otsl = "<fcel>A<fcel>B<nl><xcel><fcel>C<nl><xcel><fcel>D<nl>"
    result = otsl_to_html(otsl)
    assert 'rowspan="3"' in result


def test_otsl_to_html_already_html_passthrough():
    html = "<table><tr><td>A</td></tr></table>"
    assert otsl_to_html(html) == html


def test_html_to_otsl_simple_2x2():
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    otsl = html_to_otsl(html)
    assert "<fcel>A" in otsl
    assert "<fcel>B" in otsl
    assert "<fcel>C" in otsl
    assert "<fcel>D" in otsl
    assert otsl.count("<nl>") == 2


def test_html_to_otsl_with_colspan():
    html = '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>'
    otsl = html_to_otsl(html)
    assert "<lcel>" in otsl


def test_html_to_otsl_with_rowspan():
    html = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    otsl = html_to_otsl(html)
    assert "<ucel>" in otsl


def test_html_to_otsl_empty_cells():
    html = "<table><tr><td></td><td>A</td></tr></table>"
    otsl = html_to_otsl(html)
    assert "<ecel>" in otsl


def test_roundtrip_simple_table():
    html_in = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    otsl = html_to_otsl(html_in)
    html_out = otsl_to_html(otsl)
    assert html_out == html_in


def test_roundtrip_with_colspan():
    html_in = '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>'
    otsl = html_to_otsl(html_in)
    html_out = otsl_to_html(otsl)
    assert html_out == html_in
