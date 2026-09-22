"""AeroPattern Free — entry point. Loads the full application body."""
from pathlib import Path
_p = Path(__file__).resolve().parent
_code = (_p / "_app_p1.py").read_text(encoding="utf-8") + (_p / "_app_p2.py").read_text(encoding="utf-8")
exec(compile(_code, "app.py", "exec"), globals())
