from io import StringIO

from tools.data_management.progress import ProgressReporter


def test_progress_reporter_logs_periodic_non_tty_updates():
    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=2, stream=stream, force_tty=False)

    progress.update("scan", 1, total=5, phase="read")
    progress.update("scan", 2, total=5, phase="read")
    progress.update("scan", 3, total=5, phase="read")

    logs = stream.getvalue()
    assert "current=1" in logs
    assert "current=2" in logs
    assert "current=3" not in logs


def test_progress_reporter_draws_tty_bar():
    stream = StringIO()
    progress = ProgressReporter(enabled=True, stream=stream, force_tty=True)

    progress.update("scan", 2, total=4)
    progress.finish("scan", total=4)

    output = stream.getvalue()
    assert "\r[docds] scan" in output
    assert "2/4" in output
    assert "phase=done" in output
