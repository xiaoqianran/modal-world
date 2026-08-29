import math


def test_subdivided_icosahedron_contains_poles_requiring_alternate_up():
    phi = (1 + math.sqrt(5)) / 2
    midpoint = [(a + b) / 2 for a, b in zip((0, 1, phi), (0, -1, phi))]
    norm = math.sqrt(sum(value * value for value in midpoint))
    normalized = [value / norm for value in midpoint]
    assert normalized == [0.0, 0.0, 1.0]
