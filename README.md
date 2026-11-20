# API de scoring crédit — Projet 8 MLOps

Application Gradio exposant le pipeline LightGBM **`credit_scoring_lightgbm`, version 4**,
entraîné lors d'un précédent projet OpenClassrooms. À partir des données préparées d'un client,
elle retourne une probabilité de défaut et une classe, puis enregistre l'appel dans SQLite.

**Disponible :** prédiction, schéma d'entrée, historique SQLite, simulation d'appels,
tests automatisés, image Docker et pipeline `test → build → deploy`.
L'analyse de dérive, le tableau de bord et l'hébergement permanent restent à réaliser.
Stack : Python 3.13, Gradio, pandas, scikit-learn, LightGBM, MLflow, pytest et Docker.
Versions des dépendances directes : [requirements.txt](requirements.txt).

## Fonctionnement et organisation

```text
Programme appelant ou interface → Gradio → validation → pipeline entraîné → score et classe
                                            └─ réussite, rejet ou panne → historique SQLite
```

Le pipeline contient l'imputation et LightGBM ; il est chargé **une fois par processus**, puis partagé.
Les prédictions de l'interface et de l'API passent une à une dans la même file. Les artefacts sont locaux,
sans serveur MLflow. Jointures, encodage des données brutes et entraînement sont réalisés en amont.

| Fichier ou dossier | Rôle |
|---|---|
| `app/main.py` | Interface Gradio, endpoints, traduction des erreurs et déclenchement du stockage. |
| `app/prediction.py` | Chargement du modèle, contrôle des entrées et calcul de la prédiction. |
| `app/storage.py` | Écriture de l'historique dans SQLite. |
| `app/artifacts/credit_scoring_lightgbm/4/` | Pipeline `model.pkl`, signature `MLmodel`, provenance et fichiers MLflow. |
| `scripts/simuler_appels_api.py` | Envoi de clients tirés au hasard dans le CSV processed. |
| `tests/conftest.py` | Fixtures pytest : données, modèles simulés, bases isolées et serveurs temporaires. |
| `tests/test_prediction.py`, `tests/test_api.py` | Validation, seuil, réutilisation du modèle, interface et appels Gradio. |
| `tests/test_storage.py` | Enregistrement SQLite, erreurs de stockage et persistance Docker. |
| `tests/verifier_api_deployee.py` | Contrôle HTTP d'une API démarrée, partagé par T08 et le job `deploy`. |
| `tests/test_verifier_api_deployee.py` | Vérifie que ce contrôleur refuse les références et résultats incorrects. |
| `tests/verifier_persistance_docker.py` | Contrôle T13 avec deux conteneurs successifs et un même volume. |
| `tests/fixtures/` | JSON complet du client fictif et résultat attendu pour T08. |
| `pytest.ini` | Configuration commune de pytest et de la couverture. |
| `Dockerfile`, `.dockerignore` | Construction de l'image ; seuls l'application et les fichiers nécessaires y entrent. |
| `.github/workflows/tests-api.yml` | Tests, construction et validation du conteneur sur GitHub Actions. |

Datasets locaux et environnement Python sont ignorés par Git ; les anciennes expériences restent dans l'historique.
Le [plan T01–T13](docs/plan_tests_api_scoring.md) détaille les scénarios.

## Installation et démarrage local

Prérequis : Python 3.13 et `uv`. Commandes PowerShell depuis la racine du dépôt :

```powershell
# Créer l'environnement uniquement s'il n'existe pas déjà.
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\Activate.ps1
python -m app.main
```

Ouvrir **http://127.0.0.1:7860** ; arrêter avec `Ctrl+C`. Les commandes suivantes supposent `.venv` activé à la racine.

| Variable d'environnement | Valeur par défaut hors Docker | Usage |
|---|---|---|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Adresse d'écoute. |
| `GRADIO_SERVER_PORT` | `7860` | Port d'écoute. |
| `API_CALLS_DB_PATH` | `<racine>/data/api_calls.sqlite3` | Fichier SQLite ; à définir avant le démarrage. |

L'interface accepte un JSON ou propose jusqu'à 50 clients du CSV facultatif
`data/processed/application_test_final_encoded.csv`. S'il manque ou est incompatible,
la saisie et les appels API restent disponibles. Les données RAW ne sont pas modifiées.

## Contrat d'entrée, appels et erreurs

- `/predict` reçoit un objet JSON représentant **un client avec les 310 clés obligatoires**.
- `/schema` expose leurs noms, types, ordre et possibilité de `null`, d'après `MLmodel`.
- L'API exige toutes les clés, même celles déclarées facultatives dans la signature MLflow v4.
- Une clé absente est nommée dans l'erreur avant le calcul ; les clés inconnues sont refusées,
  notamment `TARGET` et `SK_ID_CURR`. Les colonnes sont réordonnées selon le modèle.
- Un `null` numérique explicite est imputé ; **zéro reste zéro**. Les booléens exigent
  `true` ou `false` et refusent `null`. Les chaînes numériques et nombres non finis sont refusés.
- Le score appartient à `[0, 1]`. La classe de défaut est `1` ; elle est prédite lorsque
  `score >= threshold`, avec un seuil actuel de **0.5**.

Les règles de valeur figurent dans `app/prediction.py` et le plan de tests. Le revenu nul reste accepté ;
les soldes signés sont conservés et les rapports de montants ne sont pas systématiquement limités à 1.

Le lien **Use via API** de Gradio décrit les deux fonctions. Exemple depuis un programme tiers :

```python
import json
from pathlib import Path
from gradio_client import Client
from gradio_client.exceptions import AppError

client = Client("http://127.0.0.1:7860")
try:
    schema = client.predict(api_name="/schema")
    payload = json.loads(Path("tests/fixtures/client_valid.json").read_text(encoding="utf-8"))
    print(client.predict(client=payload, api_name="/predict"))
except AppError as error:
    print(error)
finally:
    client.close()
```

La réponse contient `score`, `prediction`, `threshold`, `model_name` et `model_version`.
Une entrée invalide déclenche `AppError` avec son motif. Une panne du calcul renvoie
« La prédiction a échoué. Consulter les logs du serveur. », sans nouveau score ;
le programme appelant doit aussi gérer les erreurs réseau.
En HTTP brut, utiliser le protocole Gradio : POST `/gradio_api/call/predict`, puis GET
`/gradio_api/call/predict/{event_id}`. Le flux SSE contient `complete` ou `error` ; le succès
du POST seul ne garantit pas celui du calcul. Ce n'est pas une route REST `POST /predict`.

## Historique SQLite

La base et la table `api_calls` sont créées à la première écriture réussie. Chaque appel traité conserve
`request_id` (UUID), `timestamp_utc` (début UTC), `operation`, `status` et `duration_ms`.
La durée exclut l'attente dans la file et l'écriture SQLite.

| Opération et statut | Données supplémentaires enregistrées |
|---|---|
| `predict` / `success` | `input_json` original avec ses `null`, `score`, `prediction`, `threshold`, `model_version`. |
| `schema` / `success` | Aucun contenu client ni résultat de prédiction. |
| `rejected` | `reason` indique le motif ; aucune copie de l'entrée ni score. |
| `error` | `reason` contient un message générique et le type d'exception, sans son texte potentiellement sensible. |

Une connexion est ouverte et fermée par écriture. L'enregistrement est synchrone ; une erreur
SQLite est journalisée sans changer la réponse initiale. **Une prédiction réussie ne garantit
donc pas son enregistrement** : consulter les logs du serveur en cas de ligne manquante.
Le stockage complet s'applique à toute prédiction acceptée ; son usage est prévu pour le TP,
sans filtre distinguant un client fictif d'un client réel. Bases et entrées ne doivent pas être versionnées.
Aucune purge automatique n'est prévue.

**Cas non enregistrés :** le texte JSON mal formé dans l'interface est rejeté avant `predict` ;
il en va de même des requêtes refusées par Gradio avant l'entrée dans les fonctions applicatives.
Un JSON valide auquel manque une clé entre dans `predict` et produit une ligne `rejected`.
Après un appel, consulter la base locale avec :

```powershell
python -m sqlite3 data/api_calls.sqlite3 "SELECT request_id, timestamp_utc, operation, status, score, duration_ms FROM api_calls ORDER BY rowid DESC LIMIT 5;"
```

## Docker et persistance

Prérequis : moteur Docker Linux démarré. Construire l'image puis monter un volume nommé :

```powershell
docker build --tag credit-scoring:local .
# Exécuter après une construction réussie ; le port 17860 évite le serveur local.
docker run --detach --name scoring --publish 127.0.0.1:17860:7860 --mount type=volume,source=scoring-history,target=/app/data credit-scoring:local
python tests/verifier_api_deployee.py --url http://127.0.0.1:17860 --timeout 120
docker exec scoring python -m sqlite3 /app/data/api_calls.sqlite3 "SELECT operation, status, score, duration_ms FROM api_calls ORDER BY rowid DESC LIMIT 5;"
```

Ouvrir **http://127.0.0.1:17860**. L'image embarque le modèle, écoute sur `0.0.0.0:7860`
et démarre avec `python -m app.main` sous l'UID non root `10001`. Le port hôte reste local.
Dans Docker, `API_CALLS_DB_PATH` vaut `/app/data/api_calls.sqlite3` ; le volume doit être
accessible à cet utilisateur. La base locale de l'ordinateur et celle du volume sont distinctes.
`docker logs scoring` affiche les journaux. `docker rm --force scoring` supprime le conteneur ;
le volume `scoring-history` demeure. Réutiliser le même volume au prochain lancement conserve
l'historique. **Sans volume, supprimer le conteneur supprime aussi sa base.**
Les datasets et tests ne sont pas copiés dans l'image ; le contrôle est lancé depuis l'hôte.

## Alimenter l'historique avec des appels simulés

Avec l'API démarrée, ouvrir un second terminal avec `.venv` activé :

```powershell
python scripts/simuler_appels_api.py
python scripts/simuler_appels_api.py --url http://127.0.0.1:17860 --count 100
python scripts/simuler_appels_api.py --count 5 --seed 42
```

Le script lit le CSV processed (`--csv` permet de changer le chemin), demande `/schema`, puis
sélectionne 100 lignes distinctes par défaut et envoie leurs variables à `/predict` une à une.
Types, `null` et zéros sont conservés ; une pause aléatoire de 0,2 à 1 seconde sépare les appels.
`--seed` reproduit le tirage sur un même CSV ; les clients peuvent revenir entre deux lots.
L'API ciblée écrit dans sa propre base. Le script affiche les réussites et erreurs ; une panne
réseau ou un délai dépassé arrête le lot sans nouvel essai automatique. `Ctrl+C` interrompt le lot.
Ce tirage conserve le CSV et alimente l'historique ; il ne constitue ni un test de charge ni une dérive volontaire.

## Tests et couverture

```powershell
python -m pytest
```

Les tests unitaires simulent le calcul et vérifient la vraie validation ; les tests d'intégration
appellent un serveur Gradio temporaire avec des bases isolées. Le CSV local n'est pas nécessaire :
les cas CSV utilisent des fichiers temporaires. `conftest.py` partage ces préparations et affiche T01–T13.
La couverture des lignes et branches de `app/main.py` et `app/prediction.py` est affichée et
écrite dans `tests/reports/coverage.json` ; elle ne couvre pas tout le projet et ne garantit
pas la qualité des assertions. Rapports, cache et temporaires restent sous `tests/`, ignorés par Git.

**T08** utilise `client_valid.json`, un client fictif complet, et le vrai modèle v4. Le résultat
figé dans `t08_reference.json` est **0.9206103043784433**, classe **1**, seuil **0.5**, tolérance
absolue **0.000001**. La ligne historique donnant `0.17086662346702958` était absente : la
référence actuelle est synthétique. Un fichier manquant, une empreinte incompatible ou un
résultat différent fait échouer le contrôle. La référence n'est pas régénérée pendant les tests.
T08 et le contrôle Docker réutilisent les mêmes fonctions HTTP de `verifier_api_deployee.py`.

**T11–T12** vérifient les traces de réussite, de rejet et de panne, ainsi que la conservation de
la réponse si SQLite échoue. **T13** supprime un premier conteneur et compare les lignes dans
un second monté sur le même volume. Pour inclure T13 après construction de l'image :

```powershell
$env:SCORING_DOCKER_IMAGE = 'credit-scoring:local'
python -m pytest
Remove-Item Env:SCORING_DOCKER_IMAGE
```

Sans cette variable, pytest indique T13 comme ignoré. Le script supprime uniquement ses conteneurs et volume de test.

## Pipeline GitHub Actions

Sur les `push` et `pull_request`, les trois jobs s'enchaînent sur Ubuntu 24.04 :

1. **`test`** installe les dépendances dans un environnement neuf, lance pytest et affiche la couverture.
2. **`build`**, après réussite, construit l'image et la transmet par un artefact conservé un jour.
3. **`deploy`** charge cette même image, contrôle son identifiant, démarre le conteneur sur
   `127.0.0.1:7860`, puis appelle `/schema` et `/predict` avec le vrai modèle et la référence T08.
   Il exécute aussi obligatoirement T13 avec cette image. Un contrôle en échec fait échouer le job ;
   les journaux sont affichés et une étape `always()` supprime le conteneur principal.

Les contrôles utilisent la bibliothèque standard Python ; `deploy` n'installe pas pytest.
Conteneurs et volumes de test sont temporaires : **aucun hébergement permanent ni historique entre deux runners**.
L'image n'est pas publiée dans un registre ; aucun secret utilisateur n'est requis pour l'artefact versionné.
Dans **Actions → exécution → test**, ouvrir « Couverture des lignes et des branches par fichier ».
L'étape des tests liste les lignes manquantes ; `coverage.json` n'est pas publié comme artefact téléchargeable.

## Remplacer le modèle

La version 4 est sélectionnée explicitement ; une nouvelle version MLflow n'est pas adoptée automatiquement.

1. Exporter le pipeline approuvé et ses fichiers dans `app/artifacts/<nom>/<version>/`, puis mettre
   à jour `provenance.json` (identifiants MLflow, version, source et seuil). Vérifier les dépendances.
2. Modifier `ARTIFACTS_PATH` dans `app/prediction.py` et `ARTIFACTS` dans `tests/verifier_api_deployee.py`.
   Contrôler le prétraitement et la signature ; l'API continuera d'exiger toutes ses clés.
   La classe de défaut est codée à **1** : modifier `positive_class` seul ne change pas ce comportement.
3. Adapter `client_valid.json`, les règles métier si nécessaire, les attentes dans les tests et
   `conftest.py`, le contrôleur et la documentation. Les assertions actuelles ciblent 310 variables et la v4.
4. Calculer directement avec le vrai pipeline, hors API, le résultat du JSON et l'approuver avant de
   mettre à jour `t08_reference.json` : score, classe, seuil, nom, version, identifiants, SHA-256 des
   octets de `model.pkl` et SHA-256 du JSON client canonique ci-dessous. Justifier la tolérance et `note`.
5. Relancer les tests, reconstruire l'image, vérifier l'API et T13, puis redémarrer le service.
   Versionner ensemble artefact, référence et adaptations ; le push déclenche le pipeline.

```python
json.dumps(client, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
           allow_nan=False).encode('utf-8')
```

Aucun générateur de référence n'est fourni. T08 vérifie une non-régression technique ; évaluer séparément
la qualité prédictive du modèle. Ne pas changer la référence pour masquer un échec.
