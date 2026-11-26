# Scoring crédit et monitoring — Projet 8 MLOps

Application Gradio/FastAPI de scoring crédit, avec historique SQLite et dashboard Streamlit/Evidently.
Le modèle entraîné est **LightGBM `credit_scoring_lightgbm`, version 4** ; les prédictions utilisent **ONNX Runtime sur CPU**.

## Fonctionnement

```text
JSON client → validation → tableau NumPy float32 → ONNX (imputation + arbres) → score
                                                                          ↓
                                                                       SQLite
                                                                          ↓
                                                               Streamlit / Evidently
```

Le pipeline MLflow est converti une fois par processus ; chaque client reçoit son tableau selon un schéma précalculé.
Les contrôles des variables, types, bornes métier et valeurs manquantes sont conservés.
Le seuil est de 0,5 : `prediction=1` lorsque `score >= threshold`. `null` numérique devient
NaN pour l'imputation ; les zéros sont conservés et les booléens manquants sont refusés.
Jointures et encodage restent réalisés en amont ; l'API attend les 310 variables préparées.
La conversion n'entraîne pas le modèle et ne modifie pas l'artefact sauvegardé.

## Installation et lancement local

Prérequis : Python 3.13 et `uv`. Depuis la racine, dans PowerShell :

```powershell
# À la première installation seulement, si .venv n'existe pas :
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\Activate.ps1
python -m app.main
```

Dans un deuxième terminal, avec `.venv` activé :

```powershell
python -m streamlit run app/monitoring/dashboard.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Prédictions : **http://localhost:7860** ; dashboard : **http://localhost:8501**. Les boutons ouvrent un nouvel onglet.
Arrêt : `Ctrl+C`. Aucune authentification n'est configurée.
`/health` vérifie la disponibilité de l'API après chargement du modèle, sans tester de
prédiction ni SQLite ; `/_stcore/health` est le contrôle de santé propre à Streamlit.

| Variable | Valeur par défaut | Usage |
|---|---|---|
| `MODEL_NUM_THREADS` | `1` | Threads CPU ONNX ; entier strictement positif. |
| `GRADIO_SERVER_NAME` | `127.0.0.1` en local | Adresse d'écoute ; `0.0.0.0` dans l'image. |
| `GRADIO_SERVER_PORT` | `7860` | Port de l'API. |
| `API_CALLS_DB_PATH` | `data/api_calls.sqlite3` à la racine | Historique SQLite. |
| `MONITORING_URL` | `http://localhost:8501` | Bouton vers le dashboard. |
| `SCORING_URL` | `http://localhost:7860` | Bouton vers les prédictions. |

ONNX est le seul moteur ; une erreur de conversion bloque le démarrage.
Les liens configurables doivent être accessibles au navigateur ; ils ne changent pas les ports d'écoute.

## Utilisation et mesures

L'interface accepte un JSON et peut proposer 50 clients du CSV de démonstration, si présent.
Les endpoints Gradio `/schema` et `/predict` sont décrits dans **Use via API**.
Il ne s'agit pas d'un endpoint REST `POST /predict` : Gradio utilise un POST puis un flux SSE.
Une entrée invalide produit une `AppError` explicite ; une panne interne reste générique.
La réponse contient `score`, `prediction`, `threshold`, `model_name`, `model_version`, `request_id`.
Contrat détaillé : [plan des tests API](docs/plan_tests_api_scoring.md).

API démarrée, depuis un autre terminal :

```powershell
python scripts/simuler_appels_api.py --count 100 --seed 42
```

Le script tire des lignes de `data/processed/application_test_final_encoded.csv` ; `--csv` et `--url` sont configurables.
Une session HTTP et un contexte TLS sont réutilisés, certificats vérifiés, pour les versions Gradio épinglées.
Les appels sont espacés ; ce n'est pas un test de charge. Une erreur réseau arrête le lot sans nouvel essai.

SQLite conserve entrées/résultats des réussites, motifs des erreurs, version,
`model_backend` (`onnx` pour les nouveaux appels) et `execution_device` (`cpu`).
Les anciens moteurs inconnus restent à NULL ; les entrées des erreurs sont exclues. Base et datasets sont ignorés par Git.
Une erreur de stockage est journalisée sans bloquer la réponse de scoring.

Les mesures sont `cpu_time_ms`, `preparation_ms`, `inference_ms`, `other_elapsed_ms`, `latency_ms`.
L'inférence comprend l'imputation et la classification ONNX. Le script transmet la latence
via `POST /measurements`, corrélée par `request_id` ; cet envoi, le démarrage et les pauses
sont exclus. Les appels manuels n'ont pas cette mesure client. Les autres délais sont le reste
calculé, stockage inclus ; le CPU mesure le processus serveur et reste indépendant.
Les médianes des étapes ne s'additionnent pas. Aucun export CSV n'est créé.

## Dashboard

- **Dérive et prédictions** : les entrées des 100 dernières prédictions compatibles
  (même version/seuil) sont comparées au dataset de référence
  `data/processed/application_train_final_encoded.csv`, avant imputation.
- **Méthodes** : Wasserstein sur les numériques et Jensen–Shannon sur les catégories,
  seuil commun initial 0,1 ; KS sur les numériques avec correction de Bonferroni.
- **Scores et importances** : référence calculée avec le pipeline LightGBM original ;
  top 20 des variables par gain normalisé d'entraînement, sans interprétation causale individuelle.
- **Performances** : les N derniers appels sont comparés aux N précédents, sans chevauchement,
  avec N réglable entre 50 et 500. Les cartes affichent leurs effectifs utilisables.

Les calculs démarrent au clic. Le cache de référence dépend du contenu du CSV et de l'artefact.
Une absence de drift ne prouve ni l'équivalence des moteurs ni la précision ; les fenêtres peuvent mélanger les anciens moteurs.

## Profilage et validation

Les appels API ordinaires n'activent jamais cProfile. Diagnostic séparé :

```powershell
python -m scripts.profiler_prediction --count 1 --seed 42
python -m snakeviz tests/reports/profiling
```

Le diagnostic utilise le traitement réel sans Gradio, supprime ses lignes SQLite et conserve le fichier `.prof`.
Un arrêt forcé peut empêcher le nettoyage. Les temps profilés ne servent pas à prouver les gains.

```powershell
python -m pytest
python tests/verifier_api_deployee.py --url http://localhost:7860
```

Ces commandes sont à lancer manuellement. Les tests unitaires simulent ONNX ; T08 utilise
le vrai modèle et sa référence numérique figée, sans modifier le score ni la tolérance
pour masquer un écart. Le test réel et le déploiement doivent confirmer la compatibilité ONNX.
Les essais locaux ont montré une préparation médiane de 0,84 ms et une inférence de 0,21 ms ;
100 clients communs LightGBM/ONNX ont conservé leur décision avec des écarts de score < 1e-6.
Protocole, résultats et limites : [rapport d'optimisation](docs/rapport_optimisations.md).

## Docker et CI/CD

```powershell
docker compose up --build -d --remove-orphans
docker compose down
```

Compose lance Gradio et Streamlit avec la même image, ports publiés sur `127.0.0.1`.
Le dossier `data/` est monté en écriture pour l'API, en lecture seule pour le dashboard ;
il persiste après `down`. Aucun Nginx ni préfixe `/monitoring`. Les CSV locaux ne sont pas dans l'image.
Le workflow `test → build → deploy` installe `requirements.txt`, contrôle la même image et teste
santé, prédiction et persistance SQLite (T13). Le déploiement CI est temporaire, sans hébergement.
Les mises à jour ONNX ne sont pas encore déclarées validées par une exécution de ce pipeline.

## Architecture

`app/main.py` expose l'application ; `prediction.py` charge et exécute le modèle ;
`scoring_service.py` coordonne le scoring et sa traçabilité ; `storage.py` écrit SQLite.
`app/monitoring/` contient l'analyse et le dashboard. `scripts/` conserve le simulateur et
le diagnostic cProfile. `tests/` contient les contrôles et références synthétiques.
Le PoC reste dans `notebooks/01_premiers_pas_data_drift.ipynb`.
LightGBM, scikit-learn et MLflow restent nécessaires à la conversion et au monitoring.
Une nouvelle version doit être exportée avec sa signature/provenance dans `app/artifacts/`, puis validée.
