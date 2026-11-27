#!/usr/bin/env python3
"""
Fill in the placeholders of the COUNT Forest report repo.  Works on Windows, macOS and Linux
(standard library only, Python 3.8+).

Run it from inside the repo folder (the one that contains mkdocs.yml):

    python fill_placeholders.py --user YOUR-GITHUB-NAME --repo count-forest-crowd-counting
    python fill_placeholders.py --youtube https://www.youtube.com/watch?v=XXXXXXXXXXX     # later, once the recording is up

It replaces <github-username> and <repo-name> (also their HTML-escaped forms &lt;...&gt;, which the README
and the last PPT slide use) in the text files AND inside the .pptx, and puts the YouTube id into
section 9 of docs/index.md.  Running it again is harmless.

Optional: --also OLD NEW  (repeatable) makes any other plain text replacement everywhere,
e.g.  --also Choudhary Chaudhary
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile

TEXT_EXT = {".md", ".yml", ".yaml", ".txt", ".html", ".css", ".js", ".toml", ".cfg"}
SKIP_DIRS = {".git", "site", "cache", "node_modules", "__pycache__", ".ipynb_checkpoints", "mall_dataset", ".venv", "venv"}

USER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def youtube_id(value):
    value = value.strip()
    if YT_ID_RE.match(value):
        return value
    for pat in (r"[?&]v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})", r"/embed/([A-Za-z0-9_-]{11})",
                r"/shorts/([A-Za-z0-9_-]{11})"):
        m = re.search(pat, value)
        if m:
            return m.group(1)
    return None


def build_rules(user, repo, yt, also):
    rules = []
    if user:
        rules += [("<github-username>", user), ("&lt;github-username&gt;", user)]
    if repo:
        rules += [("<repo-name>", repo), ("&lt;repo-name&gt;", repo)]
    if yt:
        rules += [("embed/PPT_VIDEO_ID", "embed/" + yt)]
    rules += list(also)
    return rules


def apply(text, rules):
    n = 0
    for old, new in rules:
        c = text.count(old)
        if c:
            text = text.replace(old, new)
            n += c
    return text, n


def fix_text_file(path, rules):
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
    except UnicodeDecodeError:
        return 0
    new, n = apply(text, rules)
    if n:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
    return n


def fix_pptx(path, rules):
    """Rewrite the .pptx (a zip of XML files) with the replacements applied; keeps entry order."""
    total = 0
    fd, tmp = tempfile.mkstemp(suffix=".pptx", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.lower().endswith((".xml", ".rels")):
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                    if text is not None:
                        text, n = apply(text, rules)
                        if n:
                            total += n
                            data = text.encode("utf-8")
                zout.writestr(item, data)
        if total:
            shutil.move(tmp, path)
        else:
            os.remove(tmp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return total


def remaining(root):
    """Placeholders still present anywhere (text files and inside .pptx)."""
    pats = ["github-username", "repo-name", "PPT_VIDEO_ID"]
    left = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for name in fn:
            p = os.path.join(dp, name)
            ext = os.path.splitext(name)[1].lower()
            if name == os.path.basename(__file__):
                continue
            if ext in TEXT_EXT:
                try:
                    t = open(p, encoding="utf-8").read()
                except UnicodeDecodeError:
                    continue
                for k in pats:
                    if k in t:
                        # the explanatory HTML comment in index.md mentions the id; only flag the real slot
                        if k == "PPT_VIDEO_ID" and "embed/PPT_VIDEO_ID" not in t:
                            continue
                        left.setdefault(os.path.relpath(p, root), []).append(k)
            elif ext == ".pptx":
                with zipfile.ZipFile(p) as z:
                    for item in z.infolist():
                        if item.filename.endswith((".xml", ".rels")):
                            t = z.read(item.filename).decode("utf-8", "ignore")
                            for k in pats[:2]:
                                if k in t:
                                    left.setdefault(os.path.relpath(p, root) + "::" + item.filename, []).append(k)
    return left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="your GitHub username")
    ap.add_argument("--repo", help="the repository name, e.g. count-forest-crowd-counting")
    ap.add_argument("--youtube", help="YouTube link or 11-character video id of the PPT screen recording")
    ap.add_argument("--also", nargs=2, action="append", metavar=("OLD", "NEW"), default=[],
                    help="extra plain-text replacement, repeatable")
    ap.add_argument("--dir", default=".", help="repo folder (default: current folder)")
    a = ap.parse_args()

    root = os.path.abspath(a.dir)
    if not os.path.exists(os.path.join(root, "mkdocs.yml")):
        sys.exit(f"mkdocs.yml not found in {root}. Run this from inside the repo folder, or pass --dir.")

    if a.user and not USER_RE.match(a.user):
        sys.exit(f"'{a.user}' does not look like a GitHub username (letters, digits and hyphens only).")
    if a.repo and not REPO_RE.match(a.repo):
        sys.exit(f"'{a.repo}' is not a valid repository name (letters, digits, . _ - only; no spaces).")
    yt = None
    if a.youtube:
        yt = youtube_id(a.youtube)
        if not yt:
            sys.exit("Could not find an 11-character YouTube video id in --youtube.")

    pending = remaining(root)
    needs_user = any("github-username" in v for v in pending.values())
    needs_repo = any("repo-name" in v for v in pending.values())
    if (needs_user and not a.user) or (needs_repo and not a.repo):
        if not (yt or a.also):
            sys.exit("Placeholders for the GitHub user / repo are still present: pass --user and --repo.")

    rules = build_rules(a.user, a.repo, yt, a.also)
    if not rules:
        sys.exit("Nothing to do: pass --user/--repo, --youtube or --also.")

    changed = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for name in fn:
            p = os.path.join(dp, name)
            if os.path.abspath(p) == os.path.abspath(__file__):
                continue
            ext = os.path.splitext(name)[1].lower()
            n = 0
            if ext in TEXT_EXT:
                n = fix_text_file(p, rules)
            elif ext == ".pptx":
                n = fix_pptx(p, rules)
            if n:
                changed[os.path.relpath(p, root)] = n

    print("Replaced in:")
    for k, v in sorted(changed.items()):
        print(f"  {k}: {v} place(s)")
    if not changed:
        print("  (nothing: the placeholders were already filled in)")

    left = remaining(root)
    if left:
        print("\nStill to fill in:")
        for k, v in sorted(left.items()):
            print(f"  {k}: {', '.join(sorted(set(v)))}")
        if any("PPT_VIDEO_ID" in v for v in left.values()):
            print("\n  PPT_VIDEO_ID is the slot for the PPT screen recording. Re-run with --youtube <link> once it is on YouTube.")
    else:
        print("\nAll placeholders are filled in.")


if __name__ == "__main__":
    main()
