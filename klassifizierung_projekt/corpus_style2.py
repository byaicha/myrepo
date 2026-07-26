import pandas as pd
import os
import glob
import re
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ab welcher regelbasierten Konfidenz ein Text als Pseudo-Label
# fürs Training des ML-Modells verwendet wird
PSEUDO_LABEL_MIN_CONFIDENCE = 60

# Mindestanzahl an Gesamt-Signalen (b2b_score + b2c_score), unterhalb derer
# ein Text NICHT eindeutig B2B/B2C zugeordnet wird, sondern als "mixed"
# (unzureichende Evidenz) gilt - verhindert, dass ein einzelnes Wort
# ohne Gegensignal schon zu 100% Konfidenz führt
MIN_EVIDENCE = 3


class SimpleGrouper:
    def __init__(self):
        self.companies = {}  # code -> name
        self.company_prefixes = {}  # unternehmen -> präfix (mit _)
        self.groups = defaultdict(list)
        self.analyses = defaultdict(list)

        # ML-Komponenten (werden erst nach train_ml_classifier() gefüllt)
        self.vectorizer = None
        self.ml_model = None

        # Erweiterte Keyword-Listen: fachlich/nominal (B2B) vs. werblich/direkt (B2C)
        self.b2b_words = [
            'unternehmen', 'lösung', 'lösungen', 'software', 'system', 'systeme',
            'integration', 'prozess', 'prozesse', 'effizienz', 'kosten', 'infrastruktur',
            'implementierung', 'optimierung', 'schnittstelle', 'branche', 'kompetenz',
            'expertise', 'partner', 'geschäftsprozesse', 'wertschöpfung', 'skalierbar',
            'individuell', 'beratung', 'projekt', 'anforderungen', 'compliance',
            'ressourcen', 'plattform', 'konzern', 'b2b', 'produktivität', 'nachhaltigkeit',
            'zertifiziert', 'qualitätsmanagement', 'kunde', 'kunden', 'anwendungsfall'
        ]
        self.b2c_words = [
            'du', 'dein', 'deine', 'deinen', 'dir', 'dich', 'jetzt', 'neu', 'sparen',
            'rabatt', 'angebot', 'angebote', 'kostenlos', 'schnell', 'einfach', 'entdecke',
            'hol dir', 'jetzt shoppen', 'lieblings', 'exklusiv', 'gratis', 'gutschein',
            'liefer', 'bestellen', 'kaufen', 'trend', 'style', 'lifestyle', 'genieße',
            'überrasch', 'sichere dir', 'nur heute', 'limitiert', 'wow', 'perfekt für dich'
        ]

        # Formelle Anrede zählt als B2B-Signal (typisch für Geschäftskommunikation)
        self.formal_words = ['sie', 'ihnen', 'ihr unternehmen', 'ihre anforderungen']

        # Explizite Zielgruppen-Marker: anders als die Wortlisten oben (die eher
        # Formalität/Register messen) zielen diese direkt auf die genannte Zielgruppe -
        # Vertrags-/Geschäftskunden-Sprache (B2B) vs. Privatkunden-/Alltagssprache (B2C).
        # Das soll Fälle wie "formeller Konzern-Text, der trotzdem an Privatkunden
        # gerichtet ist" (z.B. Versicherungen, Banken) besser auffangen.
        self.b2b_audience_phrases = [
            'geschäftskunden', 'firmenkunden', 'gewerbekunden', 'unternehmenslösung',
            'unternehmenslösungen', 'rahmenvertrag', 'ansprechpartner für ihr unternehmen',
            'vertriebspartner', 'geschäftspartner', 'ihre firma', 'ihr unternehmen',
            'großkunden', 'für unternehmen', 'für ihr business', 'firmenkundengeschäft',
            'geschäftskonto', 'gewerbeversicherung'
        ]
        self.b2c_audience_phrases = [
            'privatkunden', 'endkunden', 'verbraucher', 'für dich und deine familie',
            'für deine familie', 'zuhause', 'in den warenkorb', 'warenkorb',
            'als privatperson', 'deine versicherung', 'dein zuhause', 'für dich und deine liebsten',
            'privatkundengeschäft', 'privatversicherung', 'für privatpersonen'
        ]

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
        """
        B2B/B2C-Analyse auf Basis mehrerer linguistischer Merkmale:
        1. Fach-/Werbevokabular (Keyword-Listen)
        2. Anrede (Sie = B2B-Signal, du = B2C-Signal)
        3. Durchschnittliche Satzlänge (B2B-Texte tendieren zu längeren,
           komplexeren Sätzen; B2C-Texte zu kurzen, direkten Sätzen)
        4. Nominalstil (Nominalisierungsdichte: Wörter auf -ung/-heit/-keit/
           -tion/-ismus) - B2B-Texte sind tendenziell nominaler/"verdinglichter"
        5. Ausrufezeichen-/Imperativ-Dichte (Call-to-Action-Signal, typisch B2C)
        6. Selbstreferenz ("wir/unser") vs. Kundenreferenz (du/Sie) - Blickrichtung
           des Textes: spricht er über das Unternehmen oder zum Kunden?
        7. Durchschnittliche Wortlänge (lange Komposita = Fachsprache = B2B-Signal)

        Gibt (style, confidence, features) zurück, wobei 'features' ein Dict
        mit den Rohwerten der einzelnen Merkmale ist (für Transparenz/Export).
        """
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return "neutral", 0, {}

        b2b_count = sum(1 for w in words if w in self.b2b_words)
        b2c_count = sum(1 for w in words if w in self.b2c_words)

        # Anrede-Signal: "sie/ihnen" formal vs. "du/dein/dich" informell
        formal_count = sum(1 for w in words if w in ['sie', 'ihnen'])
        informal_count = sum(1 for w in words if w in ['du', 'dein', 'deine', 'deinen', 'dir', 'dich'])

        b2b_score = b2b_count + formal_count
        b2c_score = b2c_count + informal_count

        # Satzlängen-Feature: kurze Durchschnittssätze deuten eher auf B2C hin
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_sentence_len = 0
        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_len >= 14:
                b2b_score += 1
            elif avg_sentence_len <= 8:
                b2c_score += 1

        # Nominalstil: Dichte von Nominalisierungen (-ung/-heit/-keit/-tion/-ismus)
        nominal_count = len(re.findall(r'\b\w+(?:ung|heit|keit|tion|ismus)\b', text_lower))
        nominal_density = nominal_count / len(words)
        if nominal_density >= 0.04:
            b2b_score += 1
        elif nominal_density <= 0.015:
            b2c_score += 1

        # Ausrufezeichen-/Imperativ-Dichte (Call-to-Action, typisch B2C)
        exclamation_count = text.count('!')
        imperative_matches = re.findall(
            r'\b(entdecke|hol dir|sichere dir|spare|genieße|bestelle|kaufe|erlebe|'
            r'verpasse nicht|jetzt informieren)\b', text_lower
        )
        cta_count = exclamation_count + len(imperative_matches)
        cta_density = cta_count / len(sentences) if sentences else 0
        if cta_density >= 0.15:
            b2c_score += 1

        # Selbstreferenz (wir/unser) vs. Kundenreferenz (du/Sie) - Blickrichtung
        self_ref_count = sum(1 for w in words if w in ['wir', 'unser', 'unsere', 'unseren', 'unserem', 'uns'])
        customer_ref_count = formal_count + informal_count
        self_ref_percent = None
        if self_ref_count + customer_ref_count > 0:
            self_ref_percent = self_ref_count / (self_ref_count + customer_ref_count)
            if self_ref_percent >= 0.65:
                b2b_score += 1
            elif self_ref_percent <= 0.35:
                b2c_score += 1

        # Durchschnittliche Wortlänge (lange Komposita = Fachsprache = B2B-Signal)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len >= 6.0:
            b2b_score += 1
        elif avg_word_len <= 4.5:
            b2c_score += 1

        # Explizite Zielgruppen-Marker (Geschäfts- vs. Privatkunden-Sprache).
        # Höher gewichtet (x2) als die generischen Wortlisten, weil sie direkter
        # auf die tatsächliche Zielgruppe abzielen statt nur auf Formalität/Register -
        # soll formelle, aber an Privatkunden gerichtete Texte (z.B. Versicherungen,
        # Banken) nicht mehr fälschlich als B2B einstufen.
        audience_b2b_count = sum(text_lower.count(p) for p in self.b2b_audience_phrases)
        audience_b2c_count = sum(text_lower.count(p) for p in self.b2c_audience_phrases)
        b2b_score += audience_b2b_count * 2
        b2c_score += audience_b2c_count * 2

        features = {
            'avg_sentence_len': round(avg_sentence_len, 1),
            'nominal_density': round(nominal_density, 3),
            'cta_density': round(cta_density, 3),
            'self_ref_percent': round(self_ref_percent, 2) if self_ref_percent is not None else None,
            'avg_word_len': round(avg_word_len, 1),
            'audience_b2b_count': audience_b2b_count,
            'audience_b2c_count': audience_b2c_count,
            'total_evidence': b2b_score + b2c_score,
        }

        if b2b_score + b2c_score == 0:
            return "neutral", 0, features

        # Mindest-Evidenz-Kriterium: zu wenige Gesamt-Signale (z.B. nur 1 B2B-Wort
        # ohne jedes Gegensignal) reichen nicht für eine eindeutige Klassifikation,
        # auch wenn die Prozentformel rechnerisch 100% ergeben würde
        if b2b_score + b2c_score < MIN_EVIDENCE:
            return "mixed", 50, features

        b2b_percent = (b2b_score / (b2b_score + b2c_score)) * 100

        if b2b_percent > 66:
            return "B2B", b2b_percent, features
        elif b2b_percent < 33:
            return "B2C", 100 - b2b_percent, features
        else:
            return "mixed", 50, features

    def train_ml_classifier(self):
        """
        Trainiert einen TF-IDF + Logistic-Regression-Klassifikator.
        Als Trainingsdaten (Pseudo-Labels) dienen nur die Dateien, die der
        regelbasierte Klassifikator eindeutig (Konfidenz >= PSEUDO_LABEL_MIN_CONFIDENCE)
        als B2B oder B2C eingeordnet hat - "mixed"/"neutral" fließen nicht ein.
        Anschließend wird JEDE Datei (auch die unsicheren) vom trainierten
        Modell neu bewertet; das Ergebnis landet zusätzlich als 'ml_style'/
        'ml_confidence' in den Analysen.
        """
        all_entries = [a for entries in self.analyses.values() for a in entries]

        train_texts, train_labels = [], []
        for a in all_entries:
            if a['style'] in ('B2B', 'B2C') and a['confidence'] >= PSEUDO_LABEL_MIN_CONFIDENCE:
                train_texts.append(a['text'])
                train_labels.append(a['style'])

        n_b2b = train_labels.count('B2B')
        n_b2c = train_labels.count('B2C')
        print(f"\n🤖 ML-Training: {len(train_texts)} Pseudo-Labels (B2B: {n_b2b} | B2C: {n_b2c})")

        if n_b2b < 3 or n_b2c < 3:
            print("  ⚠️  Zu wenige eindeutige Pseudo-Labels je Klasse (mind. 3 nötig) "
                  "- ML-Modell wird übersprungen. Regelbasierte Werte bleiben maßgeblich.")
            for a in all_entries:
                a['ml_style'] = None
                a['ml_confidence'] = None
            return

        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train = self.vectorizer.fit_transform(train_texts)

        self.ml_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.ml_model.fit(X_train, train_labels)

        # Alle Dateien (auch mixed/neutral) mit dem trainierten Modell neu einordnen
        all_texts = [a['text'] for a in all_entries]
        X_all = self.vectorizer.transform(all_texts)
        predictions = self.ml_model.predict(X_all)
        probabilities = self.ml_model.predict_proba(X_all)
        classes = list(self.ml_model.classes_)

        for a, pred, proba in zip(all_entries, predictions, probabilities):
            a['ml_style'] = pred
            a['ml_confidence'] = proba[classes.index(pred)] * 100

        print("  ✓ ML-Modell trainiert und auf alle Dateien angewendet.")

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

            style, confidence, features = self.analyze_text(text)

            if code:
                self.groups[code].append(file_path)
                self.analyses[code].append({
                    'file': filename,
                    'style': style,
                    'confidence': confidence,
                    'text': text,
                    'features': features
                })
                print(f"  ✓ {filename} -> {self.companies[code]} ({style})")
            else:
                print(f"  ❌ {filename} -> KEIN MATCH ({style})")

        # Trainiere ML-Modell auf Basis der regelbasierten Pseudo-Labels
        # und ordne alle Dateien zusätzlich per ML-Modell ein
        self.train_ml_classifier()

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
                ml_style = a.get('ml_style')
                ml_confidence = a.get('ml_confidence')
                f = a.get('features', {})
                data.append({
                    'Unternehmen': name,
                    'Code': code,
                    'Datei': a['file'],
                    'Stil_regelbasiert': a['style'],
                    'Konfidenz_regelbasiert': f"{a['confidence']:.0f}%",
                    'Stil_ML': ml_style if ml_style else 'n/a',
                    'Konfidenz_ML': f"{ml_confidence:.0f}%" if ml_confidence is not None else 'n/a',
                    'Übereinstimmung': (a['style'] == ml_style) if ml_style else 'n/a',
                    'Feature_Satzlaenge_avg': f.get('avg_sentence_len'),
                    'Feature_Nominaldichte': f.get('nominal_density'),
                    'Feature_CTA_Dichte': f.get('cta_density'),
                    'Feature_Selbstreferenz_Anteil': f.get('self_ref_percent'),
                    'Feature_Wortlaenge_avg': f.get('avg_word_len'),
                    'Feature_Gesamt_Evidenz': f.get('total_evidence'),
                    'Feature_Zielgruppe_B2B_Marker': f.get('audience_b2b_count'),
                    'Feature_Zielgruppe_B2C_Marker': f.get('audience_b2c_count'),
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