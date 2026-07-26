"""
Web Scraping Skript für B2B/B2C Sprachstil-Analyse
Autor: [Dein Name]
Datum: 11.12.2025

Ziel: Extrahiert Text von Unternehmenswebseiten aus einer Excel-Liste.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import os
import logging
from urllib.parse import urljoin
from datetime import datetime

# ================= KONFIGURATION =================
INPUT_EXCEL = "/Users/nana-aicha/Desktop/Projekt Data Analytics/B2B_B2C_Sprachstil_Analyse/Daten/Unternehmensliste/150_Unternehmen_Webseiten_B2B_B2C.xlsx"  # Pfad zu deiner Excel
EXCEL_SHEET = "Unternehmen"  # Name des Blatts

# Ordner für Outputs
RAW_HTML_DIR = "/Users/nana-aicha/Desktop/Spezialisierungsmodul /klassifizierung_projekt/daten/raw_html"
CLEAN_TEXT_DIR = "/Users/nana-aicha/Desktop/Spezialisierungsmodul /klassifizierung_projekt/daten/cleaned_text"
LOG_DIR = "/Users/nana-aicha/Desktop/Spezialisierungsmodul /klassifizierung_projekt/daten/logs"

# Scraping-Einstellungen
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_DELAY = 2  # Sekunden zwischen Requests (höfliches Scraping)
MAX_RETRIES = 3  # Wiederholungsversuche bei Fehlern


# ================= LOGGING EINRICHTEN =================
def setup_logging():
    """Einrichtung des Logging-Systems"""
    os.makedirs(LOG_DIR, exist_ok=True)

    log_filename = f"{LOG_DIR}/scraping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Zeigt auch in der Konsole
        ]
    )
    return logging.getLogger(__name__)


# ================= HILFSFUNKTIONEN =================
def create_directories():
    """Erstellt alle benötigten Ordner"""
    for directory in [RAW_HTML_DIR, CLEAN_TEXT_DIR, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)
    logger.info("Alle Ordner wurden erstellt/geprüft.")


def clean_filename(text):
    """
    Macht aus einem String einen sicheren Dateinamen.
    Speziell angepasst für B2C/B2B-Mix Probleme.
    """
    if not isinstance(text, str):
        text = str(text)

    # ZUERST: Ersetze Schrägstriche (das Hauptproblem!)
    text = text.replace('/', '_slash_')
    text = text.replace('\\', '_backslash_')

    # Dann andere problematische Zeichen
    invalid_chars = '<>:"|?*' + chr(0)
    for char in invalid_chars:
        text = text.replace(char, '_')

    # Optionale Bereinigungen
    text = text.replace(' ', '_')
    text = text.replace('__', '_').replace('__', '_')  # Doppelt für Sicherheit

    # Kürzen und bereinigen
    text = text.strip('._')
    if not text:
        text = 'unknown'
    if len(text) > 100:
        text = text[:100]

    return text


def extract_main_text(soup):
    """
    Extrahiert den Haupttext von einer Webseite.
    Entfernt Navigation, Footer, Scripts, etc.
    """

    # Cookie-Banner KLASSEN hinzufügen (deutsche Seiten)
    cookie_selectors = [
        '.cookie-banner', '.cookie-notice', '#cookie-consent',
        '.js-cookie-banner', '.cookieinfo', '.cookieDialog',
        '[class*="cookie"]', '[id*="cookie"]', '[class*="consent"]'
    ]

    for selector in cookie_selectors:
        for element in soup.select(selector):
            element.decompose()  # Entferne Banner

    # Entferne unerwünschte Tags
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        element.decompose()

    # Finde den Hauptinhalt - verschiedene Strategien
    main_content = None

    # Strategie 1: Suche nach main, article oder bestimmten Klassen
    selectors = ["main", "article", ".main-content", ".content", "#content", ".post-content"]
    for selector in selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    # Strategie 2: Nehme den body, wenn nichts anderes gefunden
    if not main_content:
        main_content = soup.body

    if main_content:
        # Extrahiere Text
        text = main_content.get_text(separator=' ', strip=True)

        # Entferne überflüssige Leerzeilen und kürze
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)

        return text[:100000]  # Begrenze auf 100.000 Zeichen
    return ""


def scrape_page(url, session, company_name, page_type):
    """
    Scrapet eine einzelne Seite und gibt HTML und sauberen Text zurück
    """
    logger.info(f"Scrape: {company_name} - {page_type}")

    for attempt in range(MAX_RETRIES):
        try:
            # HTTP-Request mit Headers (sieht aus wie ein Browser)
            headers = {
                'User-Agent': USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7',
            }

            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Wirft Exception bei HTTP-Fehlern

            # Prüfe, ob wir HTML bekommen haben
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.warning(f"Kein HTML von {url} - Content-Type: {response.headers.get('Content-Type')}")
                return None, None

            html_content = response.text

            # Parse mit BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extrahiere sauberen Text
            clean_text = extract_main_text(soup)

            # Kurze Pause zwischen den Requests
            time.sleep(REQUEST_DELAY + random.uniform(0, 1))

            return html_content, clean_text

        except requests.exceptions.RequestException as e:
            logger.warning(f"Versuch {attempt + 1}/{MAX_RETRIES} fehlgeschlagen für {url}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)  # Längere Pause vor Wiederholung
            else:
                logger.error(f"Alle Versuche fehlgeschlagen für {url}")
                return None, None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei {url}: {str(e)}")
            return None, None

    return None, None


def save_data(company_name, page_type, html_content, clean_text):
    """
    Speichert HTML und bereinigten Text in Dateien
    """
    if not html_content or not clean_text:
        return False

    # Erstelle sichere Dateinamen
    safe_company = clean_filename(company_name)
    safe_page_type = clean_filename(page_type)

    # HTML speichern
    html_filename = f"{safe_company}_{safe_page_type}.html"
    html_path = os.path.join(RAW_HTML_DIR, html_filename)

    # WICHTIG: Stelle sicher, dass der Ordner existiert
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        logger.error(f"Fehler beim Speichern von HTML {html_path}: {e}")
        return False

    # Text speichern
    text_filename = f"{safe_company}_{safe_page_type}.txt"
    text_path = os.path.join(CLEAN_TEXT_DIR, text_filename)

    os.makedirs(os.path.dirname(text_path), exist_ok=True)

    try:
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        logger.info(f"Dateien gespeichert: {html_filename}, {text_filename}")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Speichern von Text {text_path}: {e}")
        return False

# ================= HAUPTFUNKTION =================
def main():
    """Hauptfunktion des Skripts"""
    logger.info("=" * 60)
    logger.info("START B2B/B2C WEB SCRAPING")
    logger.info("=" * 60)

    # 1. Ordner erstellen
    create_directories()

    # 2. Excel-Datei einlesen
    try:
        df = pd.read_excel(INPUT_EXCEL, sheet_name=EXCEL_SHEET)
        logger.info(f"Excel-Datei geladen: {len(df)} Unternehmen gefunden")
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Excel-Datei: {e}")
        return

    # 3. HTTP-Session erstellen (effizienter für mehrere Requests)
    session = requests.Session()

    # 4. Statistik-Variablen
    total_pages = 0
    successful_pages = 0
    failed_pages = []

    # 5. Durch alle Unternehmen iterieren
    for index, row in df.iterrows():
        company = row['Unternehmen']

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Verarbeite: {company} ")
        logger.info(f"{'=' * 40}")

        # Liste der zu scrapenden Seiten
        pages_to_scrape = [
            ('Homepage', row['Homepage-URL']),
            ('Ueber_uns', row['Über-uns-URL']),
            ('Produktseite', row['Produktseite-URL'])
        ]

        # Scrape jede Seite
        for page_type, url in pages_to_scrape:
            total_pages += 1

            # Prüfe, ob URL vorhanden ist
            if pd.isna(url) or str(url).strip() == '':
                logger.warning(f"URL fehlt für {company} - {page_type}")
                failed_pages.append((company, page_type, "URL fehlt"))
                continue

            # Scrape die Seite
            html, text = scrape_page(str(url).strip(), session, company, page_type)

            if html and text:
                # Speichere die Daten
                if save_data(company, page_type, html, text):
                    successful_pages += 1
                    logger.info(f"✓ Erfolg: {page_type} gespeichert")
                else:
                    failed_pages.append((company, page_type, "Speichern fehlgeschlagen"))
            else:
                failed_pages.append((company, page_type, "Scraping fehlgeschlagen"))

    # 6. Session schließen
    session.close()

    # 7. Zusammenfassung ausgeben
    logger.info("\n" + "=" * 60)
    logger.info("SCRAPING ZUSAMMENFASSUNG")
    logger.info("=" * 60)
    logger.info(f"Gesamtseiten: {total_pages}")
    logger.info(f"Erfolgreich: {successful_pages}")
    logger.info(f"Fehlgeschlagen: {len(failed_pages)}")

    if failed_pages:
        logger.info("\nFehlgeschlagene Seiten:")
        for company, page_type, reason in failed_pages:
            logger.info(f"  - {company}: {page_type} ({reason})")

    # 8. Metadaten speichern
    metadata = {
        'scraping_datum': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gesamte_seiten': total_pages,
        'erfolgreich': successful_pages,
        'fehlgeschlagen': len(failed_pages)
    }

    metadata_path = os.path.join(LOG_DIR, 'scraping_metadata.json')
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDaten gespeichert in:")
    logger.info(f"  - Roh-HTML: {RAW_HTML_DIR}")
    logger.info(f"  - Saubere Texte: {CLEAN_TEXT_DIR}")
    logger.info(f"  - Logs: {LOG_DIR}")
    logger.info("\nScraping abgeschlossen!")


# ================= SKRIPT START =================
if __name__ == "__main__":
    logger = setup_logging()
    main()