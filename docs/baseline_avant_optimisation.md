# Baseline avant optimisation

Cette mesure est le point de départ des futures expériences, pas une version optimisée.
Le tag Git `baseline-avant-optimisation` identifiera le code instrumenté et le modèle utilisés.
Le modèle, son prétraitement, son seuil et ses paramètres n'ont pas été modifiés.

## Protocole réalisé

- Le 21 novembre 2026, de **10:10:28 à 10:16:42, heure de Paris** (début du dernier appel).
- **300 clients distincts**, tirage `--seed 42`, depuis `application_test_final_encoded.csv`.
- Appels successifs, pauses aléatoires de 0,2 à 1 seconde, sans concurrence ni test de charge.
- API locale sur `127.0.0.1:17861`, pour éviter les ports des serveurs déjà ouverts.
- Modèle chargé avant le lot ; le contrôle T08 avait été effectué. Le démarrage n'est pas mesuré.
- Aucun profiler ni suite de tests en parallèle pendant le lot.
- **300 réussites, 0 rejet, 0 panne**, 300 identifiants retrouvés dans SQLite.
- Le score reçu par le client correspond au score enregistré pour les 300 appels.
- Les **494 lignes historiques** sont conservées ; leurs nouvelles mesures restent nulles.

Commande utilisée, environnement virtuel activé :

```powershell
# Terminal serveur ; choisir un port libre.
$env:GRADIO_SERVER_PORT = '17861'
python -m app.main
# Dans un autre terminal :
python scripts/simuler_appels_api.py --url http://127.0.0.1:17861 --count 300 --seed 42 --metrics-output tests/reports/performance/baseline.csv
```

Le serveur temporaire a été arrêté après la mesure. Pour un lancement habituel, le port par défaut
reste 7860. Le modèle doit rester identique, ainsi que le CSV, pour reproduire le tirage.
Ne pas écraser `baseline.csv` lors d'une expérience : choisir un autre nom de sortie.

## Configuration de départ

| Élément | Configuration mesurée |
|---|---|
| Système | Windows 11, build 26300 ; exécution locale hors Docker |
| CPU | AMD Ryzen 9 5900HX ; 8 cœurs, 16 processeurs logiques |
| GPU disponible | NVIDIA GeForce RTX 3060 Laptop GPU ; **non utilisé par le pipeline** |
| Python | 3.13.3 |
| Modèle | `credit_scoring_lightgbm`, version 4 ; seuil 0,5 |
| LightGBM | 4.7.0 ; 300 estimateurs ; `n_jobs=-1` ; `num_threads=16` dans le booster |
| scikit-learn / pandas | 1.9.0 / 3.0.5 |
| Gradio / gradio-client | 6.31.0 / 2.7.2 |
| MLflow / FastAPI / Uvicorn | 3.16.0 / 0.162.0 / 0.54.0 |

Empreinte SHA-256 du modèle, identique à celle de la référence T08 :
`f5553d02bb1a2487952570cd1b5d8703724e2996c8e25712e72322da1e74df74`.
La configuration GPU a été vérifiée dans les paramètres du modèle chargé : backend CPU
(valeur par défaut de LightGBM). Il n'y a donc pas de pourcentage GPU artificiellement fixé à zéro.

## Résultats observés

| Mesure | Moyenne | Médiane | P95 |
|---|---:|---:|---:|
| Inférence du pipeline (ms) | 9,29 | 10,50 | 12,00 |
| Traitement applicatif (ms) | 59,23 | 61,06 | 72,21 |
| Réponse complète côté client (ms) | 637,19 | 626,45 | 737,58 |
| Temps CPU du processus pendant le traitement (ms) | 60,83 | 62,50 | 78,13 |
| CPU du processus, 100 % = un cœur | 102,50 % | 99,72 % | 122,49 % |

Les trois temps ne mesurent pas la même chose :

- `inference_ms` : uniquement `pipeline.predict_proba`, donc imputation et LightGBM.
- `duration_ms` : traitement de l'appel, incluant validation et préparation du DataFrame,
  hors file Gradio, réseau et écriture SQLite.
- `latency_ms` : de la soumission par gradio-client à l'obtention du résultat ; inclut les
  échanges, la file, le traitement, le stockage et la récupération du résultat par le client.
  Les pauses entre clients et le chargement du CSV sont exclus.
- `cpu_time_ms` : temps CPU du processus serveur, tous ses threads, pendant le traitement.
  `cpu_percent = 100 × cpu_time_ms / temps écoulé` ; il peut dépasser 100 %. Ce n'est ni le
  CPU global de la machine, ni une mesure limitée au seul calcul LightGBM. Les autres activités
  du même processus et la granularité du compteur Windows peuvent influencer les mesures courtes.

L'écart entre latence et traitement ne permet pas à lui seul d'attribuer un goulot
d'étranglement au réseau ou à Gradio. Le profilage et les optimisations seront réalisés ensuite,
une modification à la fois. Les chiffres ci-dessus ne constituent pas une amélioration démontrée.

## Conservation et comparaison future

- SQLite local : `data/api_calls.sqlite3`, colonnes ajoutées sans effacer l'historique.
- Sauvegarde préalable : `data/api_calls.sqlite3.before-baseline`.
- CSV client : `tests/reports/performance/baseline.csv`.
- Mesures rapprochées par `request_id` : `tests/reports/performance/baseline_joined.csv`.
- Valeurs exactes et versions : `tests/reports/performance/summary.json`.
- Les données détaillées restent locales, ignorées par Git. Ce rapport agrégé est versionné.

Empreinte du CSV client :
`3a9bbb747ba9fa09e74af440003d79b2d90ce4d08fc314d6fdc51bfc2e513d5b`.
Chaque ligne conserve l'empreinte `input_sha256`, sans recopier les entrées dans le CSV.
Pour une future comparaison, rapprocher les lots par cette empreinte, pas par les identifiants
de requête qui changent à chaque appel. Comparer latences, inférences, CPU, écarts de score
et changements de classe, dans des conditions matérielles et logicielles comparables.

T08 conserve le score **0.9206103043784433**, classe **1**, seuil **0,5**, pour son client synthétique.
Ce contrôle et l'empreinte du modèle attestent de l'absence de changement de ce modèle ; sans
vérités terrain des crédits, aucune conclusion nouvelle sur la précision métier ou les biais
ne peut être tirée. Les tests supplémentaires couvrent les mesures et la migration SQLite.

Références : [horloges Python](https://docs.python.org/3.13/library/time.html)
et [paramètres LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
