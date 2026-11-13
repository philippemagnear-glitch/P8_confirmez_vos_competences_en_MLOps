# API de scoring crédit — Projet 8 MLOps

API et interface Gradio exposant le pipeline LightGBM `credit_scoring_lightgbm`, version **4**.
Pour un client déjà préparé, elles retournent une probabilité de défaut et une classe.
Stack : Python 3.13, pandas, scikit-learn, LightGBM, MLflow, Gradio, pytest et Docker.
Les versions des dépendances directes sont fixées dans `requirements.txt`.

## Fonctionnement et contrat d'entrée

Le modèle est chargé au démarrage **une seule fois par processus**, puis réutilisé.
Les appels de prédiction de l'interface et de l'API partagent une file avec un calcul à la fois.
Plusieurs processus auraient chacun leur propre modèle en mémoire.
Les artefacts locaux dans `app/artifacts/credit_scoring_lightgbm/4/` rendent l'application
indépendante d'un serveur ou d'une base MLflow. Le pipeline contient l'imputation et LightGBM ;
l'application ne réalise ni entraînement, ni jointures, ni encodage des données brutes.

- `/predict` attend un objet JSON représentant **un seul client avec les 310 clés obligatoires**.
- `/schema` expose les noms, types, ordre et possibilité de `null`, depuis `MLmodel`.
- L'API impose toutes les clés, même celles déclarées facultatives dans la signature MLflow v4.
- Une clé absente est nommée dans l'erreur avant l'appel au modèle ; les clés inconnues sont refusées.
- `TARGET` et `SK_ID_CURR` ne sont pas des entrées du modèle.
- Un `null` numérique explicite passe par l'imputation ; **zéro conserve sa valeur**.
- Les booléens exigent `true` ou `false` et refusent `null` ; les chaînes numériques sont refusées.
- L'ordre reçu peut varier : les colonnes sont réordonnées selon la signature avant le calcul.
- Le score doit être fini et compris entre 0 et 1. La classe de défaut est `1` ;
  `prediction = 1` lorsque `score >= threshold`, avec un seuil actuel de **0.5**.

Les règles de valeur figurent dans `app/prediction.py` et le [plan T01–T10](docs/plan_tests_api_scoring.md).
Le revenu nul reste accepté faute de règle contraire établie. Les soldes signés sont conservés ;
un rapport de montants n'est pas nécessairement une proportion limitée à 1.

## Rôle des fichiers

| Fichier ou dossier | Rôle |
|---|---|
| `app/main.py` | Construit l'interface, expose les endpoints et traduit les erreurs pour l'appelant. |
| `app/prediction.py` | Charge et contrôle l'artefact, valide les entrées et calcule le score. |
| `app/artifacts/credit_scoring_lightgbm/4/` | Pipeline `model.pkl`, signature `MLmodel`, provenance et fichiers MLflow associés. |
| `tests/conftest.py` | Préparations partagées par pytest : client JSON, modèle simulé et serveur Gradio temporaire. |
| `tests/test_prediction.py` | Tests unitaires de validation, ordre, seuil, score et réutilisation du modèle. |
| `tests/test_api.py` | Tests des callbacks et appels HTTP Gradio, dont T08 avec le vrai modèle. |
| `tests/verifier_api_deployee.py` | Attend une API démarrée, appelle `/schema` puis `/predict` et vérifie la référence. |
| `tests/test_verifier_api_deployee.py` | Vérifie que ce contrôleur refuse les références manquantes et les mauvais résultats. |
| `tests/fixtures/client_valid.json` | Entrée complète d'un **client fictif**, indépendante du CSV de démonstration. |
| `tests/fixtures/t08_reference.json` | Résultat attendu pour ce client avec le vrai modèle, identifiants et empreintes. |
| `pytest.ini` | Configuration unique des tests et de la couverture des lignes et branches. |
| `tests/reports/` | Rapports générés, ignorés par Git ; cache dans `tests/.pytest_cache/`, temporaires dans `tests/.tmp/`. |
| `Dockerfile`, `.dockerignore` | Construction de l'image avec l'application et ses artefacts, sans datasets ni secrets. |
| `.github/workflows/tests-api.yml` | Automatisation `test → build → deploy` sur GitHub Actions. |

Les données `data/raw/` et `data/processed/` et l'environnement local sont ignorés par Git.
Les anciennes expérimentations sont consultables dans l'historique Git.

## Installation et lancement local

Depuis la racine du dépôt, sous PowerShell, avec Python 3.13 et `uv` :

```powershell
# Créer l'environnement uniquement s'il n'existe pas déjà.
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m app.main
```

Ouvrir `http://127.0.0.1:7860`. Arrêter avec `Ctrl+C` ; redémarrer après une modification.
`GRADIO_SERVER_NAME` et `GRADIO_SERVER_PORT` permettent de modifier l'adresse d'écoute.
La liste de démonstration charge au maximum 50 lignes de
`data/processed/application_test_final_encoded.csv`. Ce CSV est facultatif : s'il manque
ou est incompatible, la saisie JSON et l'API restent disponibles. Les tests fabriquent leurs
propres CSV quand nécessaire ; ils ne dépendent pas de ce fichier local.

## Use via API et erreurs

Le lien **Use via API** décrit `/predict` et `/schema`. Il s'agit du protocole Gradio,
pas d'une route REST classique `POST /predict`. Exemple depuis un programme tiers :

```python
import json
from pathlib import Path
from gradio_client import Client
from gradio_client.exceptions import AppError

client = Client("http://127.0.0.1:7860")
print(client.predict(api_name="/schema"))
payload = json.loads(Path("tests/fixtures/client_valid.json").read_text(encoding="utf-8"))
try:
    print(client.predict(client=payload, api_name="/predict"))
except AppError as error:
    print(error)
```

Le résultat contient `score`, `prediction`, `threshold`, `model_name` et `model_version`.
Une donnée invalide déclenche `AppError` avec une explication, nommant les colonnes concernées
lorsque possible. Une panne du calcul déclenche un message générique :
« La prédiction a échoué. Consulter les logs du serveur. ». Aucun nouveau score n'est retourné ;
le journal d'erreur du calcul indique le type d'exception sans les valeurs du client.
En HTTP brut, le POST `/gradio_api/call/predict` renvoie un `event_id` ; le GET
`/gradio_api/call/predict/{event_id}` transmet un événement SSE `complete` ou `error`.
Le succès du POST seul ne garantit donc pas celui du calcul.

## Tests et référence T08

Une commande lance les tests unitaires, les tests d'intégration et la couverture, en local comme en CI :

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Pytest affiche les résultats par ID T01–T10. Pytest-cov affiche la couverture et produit
`tests/reports/coverage.json` pour `app/main.py` et `app/prediction.py`, avec lignes et branches.
La couverture indique le code exécuté ; elle ne garantit pas à elle seule la qualité des tests.
Les tests unitaires simulent le calcul, tout en utilisant la vraie validation des données.
Le schéma est lu directement dans `MLmodel`, sans copie JSON supplémentaire.

**T08 envoie le client fictif complet à une vraie API avec le vrai pipeline v4.**
Son résultat attendu est **0.9206103043784433**, classe **1**, seuil **0.5**, avec une tolérance
absolue de **0.000001**. Cette référence synthétique a remplacé la ligne historique absente
qui produisait `0.17086662346702958` : ces deux scores correspondent à des entrées différentes.
Le résultat attendu n'est pas simulé et n'est jamais recalculé automatiquement pendant les tests.
L'absence du modèle ou du client, une empreinte incompatible ou un résultat différent fait échouer le contrôle.
T08 et Docker utilisent le même JSON et les mêmes fonctions de `verifier_api_deployee.py`.
Ce script utilise seulement la bibliothèque standard Python et retourne un code d'échec en cas de problème.
Il ne démarre pas le serveur, ne l'arrête pas et ne modifie pas la référence.

## Docker et pipeline CI/CD

Avec un moteur Docker Linux démarré, depuis la racine (port hôte 17860 pour éviter le serveur local) :

```powershell
docker build --tag credit-scoring:local .
if ($LASTEXITCODE -ne 0) { throw 'Construction Docker echouee' }
docker run --detach --name scoring --publish 127.0.0.1:17860:7860 credit-scoring:local
if ($LASTEXITCODE -ne 0) { throw 'Demarrage Docker echoue' }
try {
    .\.venv\Scripts\python.exe tests/verifier_api_deployee.py --url http://127.0.0.1:17860 --timeout 120
    if ($LASTEXITCODE -ne 0) { docker logs scoring; throw 'Verification API echouee' }
} finally {
    docker rm --force scoring
}
```

L'image démarre avec `python -m app.main`, un utilisateur non root et une écoute interne
sur `0.0.0.0:7860`. Le port hôte est publié uniquement sur `127.0.0.1`.
Sur chaque `push` et `pull_request`, GitHub Actions exécute successivement :

1. **`test`** : installe les dépendances, lance pytest et affiche la couverture des lignes et branches par fichier.
2. **`build`** : après réussite des tests, construit l'image et l'exporte comme artefact du workflow.
3. **`deploy`** : récupère cette même image, vérifie son identifiant, démarre le conteneur
   sur `127.0.0.1:7860`, attend l'API et contrôle `/schema` puis `/predict` avec le vrai modèle.
   En cas d'échec, affiche les journaux ; une étape `always()` supprime le conteneur même si le contrôle échoue.

Une image construite peut contenir une application inutilisable : le contrôle HTTP vérifie son fonctionnement.
**Ce déploiement est temporaire sur le runner GitHub Actions, sans hébergement permanent.**
L'image n'est pas publiée dans un registre ; aucun secret n'est nécessaire pour l'artefact actuellement versionné.
Après un push, consulter l'onglet **Actions** du dépôt pour constater le résultat des trois jobs.

## Changer de modèle : mises à jour manuelles

Enregistrer une nouvelle version dans MLflow ne change pas automatiquement le modèle de l'API.

1. Exporter le pipeline approuvé et ses fichiers MLflow dans `app/artifacts/<nom>/<version>/`.
   Mettre à jour `provenance.json` : nom, version, identifiants réels du modèle et du run, seuil et provenance.
   Le code actuel sélectionne explicitement la classe de défaut **1** : modifier `positive_class` seul ne la change pas.
2. Modifier `ARTIFACTS_PATH` dans `app/prediction.py` et `ARTIFACTS` dans `tests/verifier_api_deployee.py`.
   Vérifier le prétraitement embarqué et la compatibilité des dépendances avec le nouvel artefact.
   Le schéma sera relu depuis son `MLmodel` ; toutes ses clés resteront obligatoires.
3. Adapter `client_valid.json` si les variables ou types changent, ainsi que les règles métier si nécessaire.
   Mettre à jour les attentes de version et de nombre de variables dans les tests, dont `conftest.py`,
   le contrôleur de déploiement et la documentation. Les assertions actuelles ciblent la version 4 et 310 variables.
4. Calculer directement avec le vrai pipeline le résultat du JSON, en dehors de l'API, puis l'examiner et l'approuver.
   Mettre à jour manuellement `t08_reference.json` : score, classe, seuil, nom, version, identifiants MLflow,
   empreinte SHA-256 des octets de `model.pkl` et empreinte SHA-256 du JSON canonique du client.
   Ce JSON canonique est `json.dumps(client, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
   allow_nan=False).encode('utf-8')`. Conserver une tolérance justifiée et expliquer la référence dans `note`.
5. Relancer pytest, reconstruire l'image et exécuter le contrôle Docker ci-dessus, puis redémarrer l'API.
   Examiner et versionner ensemble ces changements ; le push déclenchera la vérification CI/CD.

T08 mesure une non-régression technique, pas la qualité prédictive : évaluer le nouveau modèle sur des données
appropriées avant son adoption. Ne jamais changer la référence simplement pour faire passer un test.
Aucun notebook ni générateur de référence n'est fourni ; cette préparation reste manuelle.
Le stockage des appels, le monitoring, l'analyse de dérive et son tableau de bord restent à réaliser.
