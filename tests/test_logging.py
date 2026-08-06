import logging

from app.core.logging import configure_logging


def test_logging_creates_log_file(tmp_path) -> None:
    configure_logging(tmp_path, "INFO")
    logging.getLogger("test").info("startup")
    assert (tmp_path / "traffic-annotator.log").exists()
