import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_academic_report_is_generated_from_project_results(tmp_path):
    output = tmp_path / "report.md"
    subprocess.run(
        [sys.executable, "scripts/render_final_report.py", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = output.read_text(encoding="utf-8")
    assert "## 2. Kiến trúc và phương pháp" in report
    assert "## 4. Kết quả" in report
    assert "### 5.1 Sai lầm đầu tiên trong các ván CNN thua" in report
    assert report.count("### 5.1 Sai lầm đầu tiên trong các ván CNN thua") == 1
    assert "80.5%" in report
    assert "181 ván" in report
