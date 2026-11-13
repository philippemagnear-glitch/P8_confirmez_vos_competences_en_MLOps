# API de scoring crédit — Projet 8 MLOps

Projet OpenClassrooms exposant un modèle LightGBM entraîné lors d'un précédent TP.
L'application reçoit les données préparées d'un client et retourne un score de défaut.
Elle ne réalise ni entraînement, ni jointures, ni encodage des données brutes.

## Fonctionnalités

- Interface Gradio avec sélection d'un client de démonstration ou saisie JSON.
- API Gradio `predict` pour le score et `schema` pour le contrat d'entrée.
- Chargement du modèle une fois au démarrage, puis réutilisation entre les requêtes.
- Validation des champs, types et bornes établies pour les données préparées.
- Tests unitaires et d'intégration, couverture des lignes et des branches.
- Workflow GitHub Actions exécutant les tests sur les `push` et `pull_request`.

## Modèle et contrat d'entrée

Le modèle `credit_scoring_lightgbm`, version `4`, est conservé dans
`app/artifacts/credit_scoring_lightgbm/4/`. Le pipeline contient l'imputation et LightGBM.
Les noms, types et ordre des variables proviennent de la signature du fichier `MLmodel`.
La provenance et le seuil de décision sont décrits dans `provenance.json`.
L'application n'a besoin ni d'une base MLflow ni d'un serveur MLflow.

Chaque requête représente un seul client :

- les **310 clés** du contrat API sont obligatoires, même si la signature MLflow d'origine indique des champs facultatifs ;
- les clés supplémentaires, dont `SK_ID_CURR` et `TARGET`, sont refusées ;
- `null` est accepté pour les valeurs numériques manquantes et transmis à l'imputation ;
- `0` conserve sa valeur ; un champ booléen attend `true` ou `false` et refuse `null` ;
- le score correspond à la probabilité de la classe de défaut `1` ;
- la prédiction vaut `1` pour un score supérieur ou égal au seuil de `0.5`.

Le revenu nul reste accepté. Les soldes signés et les rapports de montants ne sont pas
systématiquement bornés à `[0, 1]`. Les proportions établies, compteurs et dates relatives
font l'objet de validations spécifiques dans `app/prediction.py`.

## Structure

```text
.github/workflows/tests-api.yml        # intégration continue
app/
  main.py                             # interface et endpoints Gradio
  prediction.py                       # chargement, validation et prédiction
  artifacts/credit_scoring_lightgbm/4/ # pipeline et métadonnées sélectionnés
data/
  raw/                                # sources locales conservées, immuables
  processed/                          # données locales préparées
  
docs/plan_tests_api_scoring.md         # plan T01 à T10
tests/
  fixtures/                           # JSON complet, schéma et référence T08
  conftest.py                         # fixtures et serveur temporaire
  test_api.py                         # appels HTTP et callbacks Gradio
  test_prediction.py                  # tests de validation et de prédiction
  pytest.ini                          # configuration partagée local/CI
  coverage.ini                        # stockage de la couverture
  reports/                            # rapports générés, ignorés par Git
  .pytest_cache/                      # cache généré, ignoré par Git
  .tmp/                               # fichiers temporaires des tests
requirements.txt
```

Les notebooks, documents d'entraînement et anciens runs MLflow ne font plus partie
du projet courant. Leur historique Git reste disponible. Les données sont conservées
pour la démonstration et les futurs travaux de monitoring, sans être versionnées.

## Installation

Prérequis : Python 3.13 et `uv`. Depuis la racine du dépôt, sous PowerShell :

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

La première commande est destinée à une installation initiale ; conserver l'environnement
existant s'il a déjà été créé. Les dépendances directes sont pandas, MLflow, scikit-learn,
LightGBM, Gradio, pytest et pytest-cov.

## Utilisation

```powershell
.\.venv\Scripts\python.exe -m app.main
```

Ouvrir `http://127.0.0.1:7860`. Le serveur est local et le partage public est désactivé.
Pour appliquer une modification Python, arrêter le serveur avec `Ctrl+C`, puis le relancer.

Pour la sélection de clients, placer le fichier `application_test_final_encoded.csv`
dans `data/processed/`. L'interface lit au maximum ses 50 premières lignes et sélectionne
les variables attendues. Ce CSV n'est pas nécessaire aux appels directs à l'API.

Les fonctions publiques sont accessibles par le protocole Gradio, notamment avec
`gradio_client`. Le lien API de l'interface décrit les appels `predict` et `schema`.
Il ne s'agit pas d'une route REST classique `POST /predict`.

## Tests et couverture

Commande à exécuter depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m pytest -c tests/pytest.ini tests -q
```

La configuration lance les tests et la couverture des lignes et branches de
`app/main.py` et `app/prediction.py`. Elle place les rapports dans `tests/reports/`
(`coverage.json` et `.coverage`), le cache dans `tests/.pytest_cache/` et les fichiers
temporaires dans `tests/.tmp/`. Ces dossiers sont générés automatiquement.

Les tests utilisent un JSON synthétique complet, indépendant du CSV de démonstration,
et un pipeline simulé. Les tests d'intégration démarrent puis arrêtent un serveur local.
T08 requiert le JSON exact `tests/fixtures/client_v4_exact.json`, son empreinte liée à
la référence dans `t08_reference.json` et l'artefact exact v4. Il est explicitement
ignoré si ces éléments ne sont pas disponibles ; le JSON synthétique ne le remplace pas.

La CI utilise la même configuration et échoue si un test échoue. Elle n'effectue pas
encore de déploiement. Docker, le stockage persistant des appels, l'analyse de dérive
et le tableau de bord de monitoring restent à réaliser.
