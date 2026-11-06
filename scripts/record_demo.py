"""Enregistre une démo animée (GIF) de l'appli Streamlit, pilotée par playwright.

Usage, depuis la racine du projet (l'appli est lancée et arrêtée par le script) :

    uv run python scripts/record_demo.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "demo_streamlit.gif"
PORT = 8599
URL = f"http://localhost:{PORT}/"
WIDTH = 1000  # largeur du GIF (px)


def wait_ready(page: Page) -> None:
    """Attend que l'appli ait fini de calculer et d'afficher le graphique."""
    page.wait_for_selector("text=Intervalle à 90 %", timeout=60_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000
    )
    time.sleep(1.2)


def grab(page: Page) -> Image.Image:
    """Capture la page et la réduit à la largeur du GIF."""
    img = Image.open(io.BytesIO(page.screenshot(full_page=False))).convert("RGB")
    return img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)


def main() -> None:
    """Lance l'appli, enchaîne les états de la démo et écrit le GIF."""
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app" / "streamlit_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(URL, timeout=1)
                break
            except OSError:
                time.sleep(1)
        frames: list[tuple[Image.Image, int]] = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 860}, device_scale_factor=1)
            # 1. Un motif réel de trous, greffé sur une courbe du jeu dev
            page.goto(URL + "?courbe=12&motif=18")
            wait_ready(page)
            frames.append((grab(page), 2600))
            # 2. Clic sur « Trou personnalisé » dans la barre latérale
            page.get_by_text("Trou personnalisé").click()
            wait_ready(page)
            frames.append((grab(page), 2200))
            # 3. Un trou qui s'allonge : 12 h, 1 jour, 2 jours, 3 jours
            for duration in (24, 48, 96, 144):
                page.goto(URL + f"?courbe=12&mode=perso&debut=470&duree={duration}")
                wait_ready(page)
                frames.append((grab(page), 1800 if duration < 144 else 2600))
            # 4. Une autre courbe, même trou de 2 jours
            page.goto(URL + "?courbe=57&mode=perso&debut=470&duree=96")
            wait_ready(page)
            frames.append((grab(page), 2800))
            browser.close()
        images = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f, _ in frames]
        images[0].save(
            OUT,
            save_all=True,
            append_images=images[1:],
            duration=[d for _, d in frames],
            loop=0,
            optimize=True,
        )
        print(f"{OUT} : {len(frames)} images, {OUT.stat().st_size / 1e6:.2f} Mo")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
