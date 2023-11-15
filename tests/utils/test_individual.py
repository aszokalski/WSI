import pytest
from algorithms.evo.individual import (
    BinaryIndividualType,
    UnitRangeIndividualType,
    IndividualType,
)


@pytest.mark.parametrize("n_genes", [1, 2, 3, 4, 5])
def test_binary_individual_type(monkeypatch, n_genes):
    monkeypatch.setattr("random.randint", lambda a, b: 1)
    individual = BinaryIndividualType(n_genes)
    assert individual.n_genes == n_genes
    assert individual.discrete is True
    assert individual.bounds == (0, 1)
    assert individual.generate_random() == [1] * n_genes


@pytest.mark.parametrize("n_genes", [1, 2, 3, 4, 5])
def test_unit_range_individual_type(monkeypatch, n_genes):
    monkeypatch.setattr("random.uniform", lambda a, b: 0.5)
    individual = UnitRangeIndividualType(n_genes)
    assert individual.n_genes == n_genes
    assert individual.discrete is False
    assert individual.bounds == (0, 1)
    assert individual.generate_random() == [0.5] * n_genes


def test_individual_type_abstract_methods():
    with pytest.raises(TypeError):
        IndividualType(discrete=True, bounds=(0, 1), n_genes=1)

    class DummyIndividualType(IndividualType):
        def generate_random(self):
            pass

    with pytest.raises(TypeError):
        DummyIndividualType(discrete=True, bounds=(0, 1), n_genes=1)
