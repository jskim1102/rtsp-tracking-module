from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_env_loader_reads_unterminated_last_line_and_strips_matching_quotes(tmp_path):
    env_file = tmp_path / "fixture.env"
    env_file.write_text(
        "PLAIN=hello world\n"
        "DOUBLE=\"two words\"\n"
        "SINGLE='three words'\n"
        "UNMATCHED=\"kept\n"
        "LAST=present",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source ./dev.sh; load_env_file "$1"; '
            'printf "%s\\n" "$PLAIN|$DOUBLE|$SINGLE|$UNMATCHED|$LAST"',
            "bash",
            str(env_file),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == (
        'hello world|two words|three words|"kept|present'
    )
