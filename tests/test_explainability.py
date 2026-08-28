import numpy as np
import pytest

from olives_biomarkers.evaluation.explainability import AttentionSanityChecker


@pytest.mark.parametrize('layout', ['gray', 'chw', 'hwc'])
def test_background_mass_resizes_image_to_heatmap(layout):
    left = np.zeros((8, 4), dtype=np.float32)
    right = np.ones((8, 4), dtype=np.float32)
    image = np.concatenate([left, right], axis=1)
    if layout == 'chw':
        image = np.stack([image] * 3, axis=0)
    elif layout == 'hwc':
        image = np.stack([image] * 3, axis=-1)

    heatmap = np.ones((4, 4), dtype=np.float32)
    checker = AttentionSanityChecker(intensity_threshold=0.05)

    assert checker.background_mass(heatmap, image) == pytest.approx(0.5)


def test_background_mass_rejects_unsupported_image_rank():
    checker = AttentionSanityChecker()

    with pytest.raises(ValueError, match='2D or 3D'):
        checker.background_mass(np.ones((4, 4)), np.ones(8))
