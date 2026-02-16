import pandas as pd
import os
import glob
import re
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog


class SimpleGrouper:
    def __init__(self):
        self.companies = {}  # code -> name
        self.company_prefixes = {}  # unternehmen -> präfix (mit _)
        self.groups = defaultdict(list)
        self.analyses = defaultdict(list)

        # Minimale Keywords
        self.b2b_words = ['unternehmen', 'lösung', 'software', 'system', 'integration', 'prozess', 'effizienz',
                          'kosten']
        self.b2c_words = ['du', 'dein', 'jetzt', 'neu', 'sparen', 'rabatt', 'angebot', 'kostenlos', 'schnell']

    def select_excel(self):
        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename(title="Excel mit Unternehmen", filetypes=[("Excel", "*.xlsx *.xls")])

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        return filedialog.askdirectory(title="Ordner mit .txt-Dateien")

    def normalize_name(self, name):
        """
        Konvertiert Unternehmensnamen in Präfix-Format:
        - "SAP [SAP]" -> "SAP"
        - "MediaMarkt" -> "MediaMarkt"
        - "GNT Group" -> "GNT_Group"
        - "Körber Supply Chain [Körb]" -> "Körber_Supply_Chain"
        """
        # Entferne [CODE] am Ende
        name = re.sub(r'\s*\[.*?\]$', '', name).strip()

        # Ersetze Leerzeichen durch Unterstriche
        name_with_underscores = name.replace(' ', '_')

        return name_with_underscores

    def load_companies(self, excel_path):
        """Lädt Unternehmen aus Excel und erstellt Präfixe"""
        df = pd.read_excel(excel_path)
        companies_raw = df.iloc[:, 0].dropna().tolist()

        print("\n📋 Unternehmen und ihre Präfixe:")
        for entry in companies_raw:
            entry = str(entry).strip()

            # Extrahiere Code für Anzeige
            match = re.search(r'\[(.*?)\]$', entry)
            code = match.group(1) if match else entry

            # Normalisiere Namen für Präfix
            prefix = self.normalize_name(entry)

            self.companies[code] = entry
            self.company_prefixes[prefix] = code

            print(f"  {entry} -> '{prefix}'")

        return self.companies

    def analyze_text(self, text):
        """Einfache B2B/B2C Analyse"""
        text = text.lower()
        words = text.split()

        b2b_count = sum(1 for w in words if w in self.b2b_words)
        b2c_count = sum(1 for w in words if w in self.b2c_words)

        if b2b_count + b2c_count == 0:
            return "neutral", 0

        b2b_percent = (b2b_count / (b2b_count + b2c_count)) * 100

        if b2b_percent > 66:
            return "B2B", b2b_percent
        elif b2b_percent < 33:
            return "B2C", 100 - b2b_percent
        else:
            return "mixed", 50

    def find_matching_prefix(self, filename):
        """
        Findet passendes Unternehmens-Präfix für Dateiname
        Sucht: Datei beginnt mit Präfix + '_'
        """
        for prefix, code in self.company_prefixes.items():
            # Prüfe ob Datei mit "PREFIX_" beginnt
            if filename.startswith(prefix + '_'):
                return code
        return None

    def process_files(self, folder_path):
        """Verarbeitet alle Dateien"""
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        print(f"\n📁 Gefunden: {len(txt_files)} .txt-Dateien")

        # Dateien verarbeiten
        print("\n" + "=" * 50)
        print("DATEIEN ANALYSIEREN")
        print("=" * 50)

        for file_path in txt_files:
            filename = os.path.basename(file_path)

            # Finde passendes Unternehmen
            code = self.find_matching_prefix(filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()

            style, confidence = self.analyze_text(text)

            if code:
                self.groups[code].append(file_path)
                self.analyses[code].append({
                    'file': filename,
                    'style': style,
                    'confidence': confidence
                })
                print(f"  ✓ {filename} -> {self.companies[code]} ({style})")
            else:
                print(f"  ❌ {filename} -> KEIN MATCH ({style})")

        # Ergebnisse anzeigen
        print("\n" + "=" * 50)
        print("ERGEBNISSE NACH UNTERNEHMEN")
        print("=" * 50)

        for code, name in self.companies.items():
            files = self.groups.get(code, [])
            analyses = self.analyses.get(code, [])

            if files:
                # Stil-Verteilung
                styles = [a['style'] for a in analyses]
                b2b = styles.count('B2B')
                b2c = styles.count('B2C')
                mixed = styles.count('mixed')

                print(f"\n{name} [{code}]: {len(files)} Dateien")
                print(f"  📊 B2B: {b2b} | B2C: {b2c} | Mixed: {mixed}")

                # Zeige Dateien
                for a in analyses[:5]:
                    icon = "🔵" if a['style'] == "B2B" else "🔴" if a['style'] == "B2C" else "🟣"
                    print(f"    {icon} {a['file']}")
            else:
                print(f"\n{name} [{code}]: ❌ Keine Dateien")

    def export(self, output="ergebnis.xlsx"):
        """Excel-Export"""
        data = []

        for code, name in self.companies.items():
            for a in self.analyses.get(code, []):
                data.append({
                    'Unternehmen': name,
                    'Code': code,
                    'Datei': a['file'],
                    'Stil': a['style'],
                    'Konfidenz': f"{a['confidence']:.0f}%"
                })

        df = pd.DataFrame(data)
        df.to_excel(output, index=False)
        print(f"\n💾 Exportiert: {output}")


# ==================== MAIN ====================

def main():
    print("\n" + "=" * 50)
    print("GRUPPIERUNG NACH UNTERNEHMENS-PRÄFIX")
    print("(Leerzeichen -> _, z.B. 'GNT Group' -> 'GNT_Group')")
    print("=" * 50)

    grouper = SimpleGrouper()

    # 1. Excel laden
    excel = grouper.select_excel()
    if not excel:
        print("❌ Keine Excel-Datei")
        return

    grouper.load_companies(excel)

    # 2. Ordner wählen
    folder = grouper.select_folder()
    if not folder:
        print("❌ Kein Ordner")
        return

    # 3. Verarbeiten
    grouper.process_files(folder)

    # 4. Export
    if input("\n💾 Excel exportieren? (j/n): ").lower() == 'j':
        grouper.export()

    print("\n✅ Fertig!")


if __name__ == "__main__":
    main()