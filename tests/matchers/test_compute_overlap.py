import pytest

from tests.test_utils import get_annotated_image
from traccuracy.matchers._compute_overlap import (
    get_labels_with_overlap,
)


@pytest.mark.parametrize("overlap", ["iou", "iogt"])
def test_get_labels_with_overlap(overlap):
    n_labels = 3
    image1 = get_annotated_image(
        img_size=256, num_labels=n_labels, sequential=True, seed=1
    )
    image2 = get_annotated_image(
        img_size=256, num_labels=n_labels + 1, sequential=True, seed=2
    )
    empty_image = get_annotated_image(
        img_size=256, num_labels=0, sequential=True, seed=1
    )

    ious = get_labels_with_overlap(image1, image1, overlap)
    perfect_gt = [gt for gt, _, _ in ious]
    assert list(perfect_gt) == list(range(1, n_labels + 1))
    perfect_res = [res for _, res, _ in ious]
    assert list(perfect_res) == list(range(1, n_labels + 1))
    perfect_ious = [iou for _, _, iou in ious]
    assert list(perfect_ious) == [1.0] * n_labels

    get_labels_with_overlap(image1, image2, overlap)

    # Test empty labels array
    ious = get_labels_with_overlap(image1, empty_image, overlap)
    assert ious == []
