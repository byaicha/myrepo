# Analyse et classification de textes web B2B et B2C

Projet universitaire réalisé dans le cadre du master **Data Analytics** à l'Université Justus-Liebig de Gießen.

Ce projet construit un corpus à partir de sites d'entreprises et étudie les différences de style entre les communications B2B et B2C. Il associe collecte automatisée, nettoyage de textes, création de variables linguistiques et classification supervisée.

## Compétences démontrées

- constitution et contrôle d'une base de 130 entreprises et de leurs URL ;
- web scraping responsable avec `requests` et BeautifulSoup ;
- extraction et nettoyage du contenu principal des pages ;
- rapprochement des textes avec la base des entreprises ;
- création de variables linguistiques : longueur des phrases, densité nominale, appels à l'action, références au lecteur et marqueurs de cible ;
- classification explicable fondée sur des règles ;
- vectorisation TF-IDF et régression logistique avec pondération des classes ;
- export structuré des résultats pour l'analyse.

## Données et résultats

Le fichier source contient **130 entreprises** et six champs : entreprise, site, secteur et URL de trois types de pages. Le résultat actuellement disponible comprend **119 textes issus de 45 entreprises**.

| Résultat | Nombre |
|---|---:|
| Textes classés B2B par les règles | 64 |
| Textes classés B2C par les règles | 26 |
| Textes classés mixtes | 29 |
| Classements concordants entre règles et modèle | 75,6 % |

Le taux de concordance ne constitue pas une mesure de vérité terrain : le modèle est entraîné à partir de pseudo-labels produits par les règles. Il sert ici à comparer deux méthodes de catégorisation et à identifier les cas ambigus.

## Pipeline

```text
Liste des entreprises et URL
        ↓
Collecte des pages web
        ↓
Nettoyage et extraction du texte
        ↓
Variables linguistiques et règles B2B/B2C
        ↓
TF-IDF et régression logistique
        ↓
Comparaison et export Excel
```

## Structure

```text
├── data/
│   └── company_urls.xlsx
├── results/
│   └── classification_results.xlsx
├── src/
│   ├── scrape_corpus.py
│   └── classify_texts.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution

Collecte des pages :

```bash
python src/scrape_corpus.py
```

Classification des textes et création du fichier de résultats :

```bash
python src/classify_texts.py
```

Les chemins peuvent également être précisés :

```bash
python src/classify_texts.py \
  --companies data/company_urls.xlsx \
  --texts data/cleaned_text \
  --output results/classification_results.xlsx
```

## Limites et éthique de collecte

- Les pages peuvent changer de structure et rendre l'extraction incomplète.
- Les règles linguistiques simplifient des styles de communication complexes.
- Les pseudo-labels ne remplacent pas une annotation humaine indépendante.
- La collecte utilise un délai entre les requêtes. Toute réutilisation doit aussi respecter les conditions d'utilisation et les fichiers `robots.txt` des sites concernés.

## Auteure

**Aicha Agrien**  
Master Data Analytics · Université Justus-Liebig de Gießen
