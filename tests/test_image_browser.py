"""Unit tests for ImageBrowser widget and thumbnail rendering."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import pytest
from PySide6.QtWidgets import QApplication

from app.ui.views.image_browser import ImageBrowser, _get_default_icon, _BASE_IMAGE_CACHE, _ICON_CACHE


@pytest.fixture
def sample_dataset(tmp_path: Path) -> list[Path]:
    """Create a dataset of 60 temporary test images."""
    paths = []
    for i in range(60):
        p = tmp_path / f"test_{i:03d}.jpg"
        # Alternate colors
        color = (i * 4 % 255, (i * 7) % 255, (i * 11) % 255)
        Image.new("RGB", (100, 100), color=color).save(p)
        paths.append(p)
    return paths


def test_image_browser_init(qapp: QApplication) -> None:
    """Test ImageBrowser initialization properties."""
    browser = ImageBrowser()
    assert browser.objectName() == "imageBrowser"
    assert browser.iconSize().width() == 62
    assert browser.gridSize().width() == 74
    assert browser.count() == 0


def test_image_browser_set_paths_and_badges(sample_dataset: list[Path], qapp: QApplication) -> None:
    """Test set_paths with annotation counts and badge updates."""
    browser = ImageBrowser()
    counts = {sample_dataset[0]: 5, sample_dataset[1]: 120}
    browser.set_paths(sample_dataset[:10], annotation_counts=counts)

    assert browser.count() == 10
    item0 = browser.item(0)
    assert item0 is not None
    assert "5 annotations" in item0.toolTip()

    item1 = browser.item(1)
    assert item1 is not None
    assert "120 annotations" in item1.toolTip()

    item2 = browser.item(2)
    assert item2 is not None
    assert "unannotated" in item2.toolTip()

    # Update annotation count
    browser.update_annotation_count(sample_dataset[2], 3)
    assert "3 annotations" in item2.toolTip()


def test_image_browser_progressive_and_lazy_loading(sample_dataset: list[Path], qapp: QApplication) -> None:
    """Test that all 60 items get rendered via background loader or scrolling."""
    browser = ImageBrowser()
    browser.resize(300, 400)
    browser.show()
    browser.set_paths(sample_dataset)

    assert browser.count() == 60

    # Ensure items at the end of the list are initially tracked in unrendered queue or rendered
    # Process background batches until complete
    while browser._unrendered_queue:
        browser._process_background_batch()

    # Verify all items now have thumbnails rendered and are in _rendered_paths
    for idx, path in enumerate(sample_dataset):
        assert path in browser._rendered_paths
        item = browser.item(idx)
        assert item is not None
        assert not item.icon().isNull()


def test_image_browser_selection_and_signal(sample_dataset: list[Path], qapp: QApplication) -> None:
    """Test selection emits signal and renders thumbnail."""
    browser = ImageBrowser()
    browser.set_paths(sample_dataset[:5])

    selected_paths = []
    browser.image_selected.connect(lambda p: selected_paths.append(p))

    browser.setCurrentRow(2)
    assert len(selected_paths) == 1
    assert selected_paths[0] == sample_dataset[2]


def test_image_browser_scroll_rendering(sample_dataset: list[Path], qapp: QApplication) -> None:
    """Test that scrolling to items immediately forces visible thumbnail rendering."""
    browser = ImageBrowser()
    browser.resize(200, 200)
    browser.show()
    browser.set_paths(sample_dataset)

    # Scroll to bottom
    browser.scrollToBottom()
    browser._load_visible_thumbnails()

    last_item_path = sample_dataset[-1]
    assert last_item_path in browser._rendered_paths
