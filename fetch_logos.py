import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE = "https://brandlogos.net/series/nfl-team-logos-vector"
LOGO_DIR = Path("web/static/images/logos")
LOGO_DIR.mkdir(parents=True, exist_ok=True)

res = requests.get(BASE)
res.raise_for_status()
html = res.text

soup = BeautifulSoup(html, "html.parser")
# find all <a> or <img> elements linking to SVG
for link in soup.select("a[href$='.svg']"):
    svg_url = link["href"]
    fname = svg_url.split("/")[-1]
    team = fname.split(".")[0].upper()  # depends on naming in site
    out = LOGO_DIR / f"{team}.svg"
    print("Downloading", svg_url, "=>", out)
    data = requests.get(svg_url).content
    out.write_bytes(data)
