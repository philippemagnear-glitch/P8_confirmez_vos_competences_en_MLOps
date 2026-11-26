# Baseline avant optimisation

Cette mesure est le point de départ des futures expériences, pas une version optimisée.
Le tag Git `baseline-avant-optimisation` identifie le code instrumenté et le modèle utilisés.
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
- Les appels antérieurs au premier enregistrement de latence ont ensuite été supprimés de la base active.

Pour réaliser un futur lot comparable, environnement virtuel activé :

Les commandes ci-dessous sont celles du protocole historique. Avec le code actuel,
l'API utilise ONNX ; ces commandes seules ne reproduisent donc pas le moteur de départ.
Voir le [rapport d'optimisation](rapport_optimisations.md) pour les changements et leurs limites.

```powershell
# Terminal serveur ; choisir un port libre.
$env:GRADIO_SERVER_PORT = '17861'
python -m app.main
# Dans un autre terminal :
python scripts/simuler_appels_api.py --url http://127.0.0.1:17861 --count 300 --seed 42
```

Le serveur temporaire a été arrêté après la mesure. Pour un lancement habituel, le port par défaut
reste 7860. Le modèle doit rester identique, ainsi que le CSV, pour reproduire le tirage.
Les dates des appels permettent de retrouver la période mesurée. Les 300 appels de départ ont déjà été
mesurés : il n'est pas nécessaire de les rejouer.

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

Les trois temps ne mesurent pas la même chose :

- `inference_ms` : uniquement `pipeline.predict_proba`, donc imputation et LightGBM.
- `duration_ms` : traitement de l'appel, incluant validation et préparation du DataFrame,
  hors file Gradio, réseau et écriture SQLite.
- `latency_ms` : de la soumission par gradio-client à l'obtention du résultat ; inclut les
  échanges, la file, le traitement, le stockage et la récupération du résultat par le client.
  Les pauses entre clients et le chargement du CSV sont exclus.
- `cpu_time_ms` : temps CPU du processus serveur, tous ses threads, pendant le traitement.
  Ce n'est ni le CPU global de la machine, ni une mesure limitée au seul calcul LightGBM. Les autres activités
  du même processus et la granularité du compteur Windows peuvent influencer les mesures courtes.

L'écart entre latence et traitement ne permet pas à lui seul d'attribuer un goulot
d'étranglement au réseau ou à Gradio. Le profilage et les optimisations seront réalisés ensuite,
une modification à la fois. Les chiffres ci-dessus ne constituent pas une amélioration démontrée.

## Conservation et comparaison future

- SQLite local : `data/api_calls.sqlite3`, colonnes ajoutées sans effacer l'historique.
- La sauvegarde temporaire `data/api_calls.sqlite3.before-baseline` a depuis été supprimée
  lors du nettoyage ; elle ne constitue plus une source disponible.
- `api_calls` : entrées, résultats, temps serveur, CPU et latence client sur la même ligne.
- Les 300 latences initiales ont été importées par `request_id`, sans nouvelle prédiction.
- Les conditions et résultats de la baseline sont documentés dans ce rapport.
- Les données détaillées restent locales, ignorées par Git. Ce rapport agrégé est versionné.

Le tag publié reste inchangé : il contient la première instrumentation. Le commit suivant
centralise ses mesures dans SQLite et supprime les exports devenus redondants.
L'API reçoit désormais la latence via `POST /measurements`, après réception du score.
Un envoi identique est accepté sans doublon ; une tentative de remplacement est refusée.
En cas d'échec du stockage de cette mesure, le script s'arrête ; la prédiction reste conservée.

Pour une future comparaison, sélectionner les périodes des essais avec `timestamp_utc`.
Utiliser le même CSV et la même graine pour reproduire les clients.
Comparer latences, inférences, CPU, écarts de score
et changements de classe, dans des conditions matérielles et logicielles comparables.

Consultation des mesures, depuis la racine :

```powershell
python -m sqlite3 data/api_calls.sqlite3 "SELECT request_id,inference_ms,latency_ms,cpu_time_ms FROM api_calls WHERE operation='predict' AND timestamp_utc >= '2026-11-21T09:10:28' AND timestamp_utc < '2026-11-21T09:16:44' ORDER BY timestamp_utc LIMIT 5;"
```

T08 conserve le score **0.9206103043784433**, classe **1**, seuil **0,5**, pour son client synthétique.
Ce contrôle et l'empreinte du modèle attestent de l'absence de changement de ce modèle ; sans
vérités terrain des crédits, aucune conclusion nouvelle sur la précision métier ou les biais
ne peut être tirée. Les tests supplémentaires couvrent les mesures et la migration SQLite.

Références : [horloges Python](https://docs.python.org/3.13/library/time.html)
et [paramètres LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
