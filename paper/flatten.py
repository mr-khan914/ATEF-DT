#!/usr/bin/env python3
"""Write a single-file version of the manuscript for submission.

    python paper/flatten.py [--clean] [--out FILE]

Inlines the section files and the table bodies into one .tex file, and writes
every run-dependent value of values.tex directly into the text (no value macros
remain).  With --clean the revision marks are switched off.
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def expand(text: str) -> str:
    def sub_input(m):
        f = HERE / (m.group(1) + ".tex")
        return expand(f.read_text().rstrip("\n")) if f.exists() else m.group(0)

    text = re.sub(r"\\input\{([A-Za-z0-9_/]+)\}", sub_input, text)
    text = re.sub(r"\\tabin\{([A-Za-z0-9_]+)\}",
                  lambda m: (HERE / "tables" / (m.group(1) + ".tex")).read_text().rstrip("\n"), text)
    return text


def load_values() -> dict:
    vals = {}
    for k, v in re.findall(r"\\defval\{([^}]*)\}\{([^}]*)\}", (HERE / "values.tex").read_text()):
        vals[k] = v
    return vals


def inline_values(text: str) -> str:
    vals = load_values()
    # drop the macro block and the \input{values} line
    start = text.index("%% -- Run-dependent values")
    last = list(re.finditer(r"\\defval\{[^}]*\}\{[^}]*\}[^\n]*\n", text))[-1]
    text = text[:start] + text[last.end():].lstrip("\n")
    # the \defval definition line itself is part of the removed block

    def val(k):
        if k not in vals:
            raise SystemExit(f"value '{k}' is not defined in values.tex")
        return vals[k]

    text = re.sub(r"\\Vpm\{([^}]*)\}\{([^}]*)\}", lambda m: f"{val(m.group(1))}\\,$\\pm$\\,{val(m.group(2))}", text)
    text = re.sub(r"\\V\{([^}]*)\}", lambda m: val(m.group(1)), text)
    left = re.findall(r"\\(?:V|Vpm|defval|tabin)\b", text)
    if left:
        raise SystemExit(f"unreplaced value macros: {set(left)}")
    return text


COLOUR = {"E": "revE", "R1": "revRone", "R2": "revRtwo"}


def _src(tag):
    return "E" if tag.startswith("E") else ("R1" if tag.startswith("R1") else "R2")


def _arg(text, i):
    """Return (content, end) of the balanced {...} group that starts at text[i]."""
    assert text[i] == "{", text[i:i + 30]
    depth, k = 0, i
    while True:
        c = text[k]
        if c == "\\":
            k += 2
            continue
        depth += (c == "{") - (c == "}")
        k += 1
        if depth == 0:
            return text[i + 1:k - 1], k


def _replace_cmd(text, name, nargs, fn):
    out, i, pat = [], 0, "\\" + name + "{"
    while True:
        j = text.find(pat, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, args = j + len(name) + 1, []
        for _ in range(nargs):
            a, k = _arg(text, k)
            args.append(a)
        out.append(fn(*args))
        i = k


def resolve_revision_marks(text: str, clean: bool) -> str:
    """Replace the revision macros by plain LaTeX: explicit colours, or nothing."""
    # the macro definitions (expl3 block and helpers) are no longer needed
    a = text.index("\\ExplSyntaxOn")
    b = text.index("\\newenvironment{revblock}")
    b = text.index("\n", b) + 1
    text = text[:a] + text[b:]
    text = text.replace("\\newif\\ifshowrev\n", "").replace("\\showrevtrue\n", "").replace("\\colorlet{revcur}{revE}\n", "")

    def tag(t):
        ids = [x.strip() for x in t.split(",")]
        if clean:
            return ""
        return f"\\colorlet{{revcur}}{{{COLOUR[_src(ids[0])]}}}"

    text = _replace_cmd(text, "revcap", 2, lambda t, x: ("\\caption{" + x + "}") if clean else ("\\caption[" + x + "]{" + tag(t) + "{\\color{revcur}" + x + "}}"))
    text = _replace_cmd(text, "rtag", 1, tag)
    text = _replace_cmd(text, "rev", 1, lambda x: x if clean else "{\\color{revcur}" + x + "}")
    text = text.replace("\\begin{revblock}", "" if clean else "\\begingroup\\color{revcur}")
    text = text.replace("\\end{revblock}", "" if clean else "\\endgroup{}")
    if clean:
        a = text.index("%% -- Revision marks")
        b = text.index("\\definecolor{revRtwo}")
        b = text.index("\n", b) + 1
        text = text[:a] + text[b:]
    left = [m for m in ("\\rtag", "\\rev{", "revblock", "\\revcap", "\\ifshowrev", "ExplSyntax") if m in text]
    if left:
        raise SystemExit(f"unresolved revision macros: {left}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="switch the revision marks off")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = inline_values(expand((HERE / "main.tex").read_text()))
    text = resolve_revision_marks(text, a.clean)
    out = Path(a.out) if a.out else HERE / ("manuscript_clean.tex" if a.clean else "manuscript_highlighted.tex")
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
