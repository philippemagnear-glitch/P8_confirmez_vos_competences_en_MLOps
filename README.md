# API de scoring crédit — Projet 8 MLOps

API Gradio exposant le modèle LightGBM `credit_scoring_lightgbm`, version `4`,
entraîné lors d'un précédent TP OpenClassrooms. Elle reçoit les données préparées
d'un client et retourne sa probabilité de défaut et sa classe prédite.

## Fonctionnement

Le pipeline est chargé une seule fois au démarrage et partagé entre les requêtes.
Il contient l'imputation et LightGBM. Ses artefacts sont conservés dans
`app/artifacts/credit_scoring_lightgbm/4/`, sans dépendance à une base ou un serveur MLflow.
L'application ne réalise ni entraînement, ni jointures, ni encodage des données brutes.

- `predict` reçoit un objet JSON contenant les **310 clés obligatoires** du modèle.
- `schema` fournit les noms, types et ordre des variables, lus depuis `MLmodel`.
- Les clés supplémentaires, dont `TARGET` et `SK_ID_CURR`, sont refusées.
- Un `null` numérique explicite est imputé ; `0` conserve sa valeur.
- Les booléens attendent `true` ou `false` et refusent `null`.
- La classe `1` correspond au défaut ; elle est prédite pour un score `>= 0.5`.

Le revenu nul reste accepté. Les soldes signés sont conservés et les rapports de
montants ne sont pas systématiquement plafonnés à 1. Les règles établies sur les
compteurs, proportions et dates relatives sont appliquées dans `app/prediction.py`.

## Structure

```text
app/main.py                           # interface et fonctions publiques Gradio
app/prediction.py                     # chargement, validation et prédiction
app/artifacts/credit_scoring_lightgbm/4/ # modèle sélectionné et provenance
tests/fixtures/                       # JSON client et résultat de référence
tests/verifier_api_deployee.py        # contrôle une API démarrée, notamment dans Docker
tests/test_*.py                       # tests unitaires et d'intégration
pytest.ini                           # configuration unique des tests et de la couverture
tests/reports/                        # résultats générés, ignorés par Git
docs/plan_tests_api_scoring.md         # plan T01 à T10
.github/workflows/tests-api.yml        # test → build → deploy
Dockerfile, .dockerignore             # image autonome sans datasets ni secrets
requirements.txt                     # dépendances Python
```

Les données locales dans `data/raw/` et `data/processed/` sont conservées pour la
démonstration et le futur monitoring. Les anciennes expérimentations restent dans
l'historique Git. Les données et l'environnement local ne sont pas versionnés.

## Installation et lancement local

Prérequis : Python 3.13 et `uv`. Depuis la racine, sous PowerShell :

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m app.main
```

Conserver l'environnement existant s'il est déjà créé. Ouvrir `http://127.0.0.1:7860`.
Arrêter avec `Ctrl+C` et relancer pour appliquer une modification Python.

La sélection de clients lit au maximum 50 lignes de
`data/processed/application_test_final_encoded.csv`. Ce fichier est facultatif :
il n'est utilisé ni par les tests d'intégration ni par l'image Docker.

## Use via API et erreurs

Le lien **Use via API** de l'interface décrit `predict` et `schema`.
Il s'agit du protocole Gradio, pas d'une route REST classique `POST /predict`.
Exemple avec le JSON complet de référence :

```python
import json
from pathlib import Path
from gradio_client import Client
from gradio_client.exceptions import AppError

client = Client("http://127.0.0.1:7860")
payload = json.loads(Path("tests/fixtures/client_valid.json").read_text())
try:
    result = client.predict(client=payload, api_name="/predict")
    print(result)
except AppError as error:
    print(error)
```

Une réponse réussie contient `score`, `prediction`, `threshold`, `model_name` et
`model_version`. Une requête invalide déclenche `AppError` avec le champ à corriger.
Une panne du calcul déclenche `AppError` avec le message générique
« La prédiction a échoué. Consulter les logs du serveur. ». Aucun nouveau score
n'est retourné ; les logs applicatifs n'exposent pas les valeurs client.

En HTTP brut, le POST `/gradio_api/call/predict` retourne un `event_id`. Le GET
`/gradio_api/call/predict/{event_id}` fournit un événement SSE `complete` ou `error`.
Le succès du POST seul ne garantit donc pas celui de la prédiction.

## Tests et référence numérique

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Depuis la racine, cette commande lance les tests unitaires et d'intégration ainsi que
la couverture. Le job `test` utilise la même commande avec son interpréteur Python.
Le fichier unique `pytest.ini` regroupe les réglages de pytest et de coverage.

Les couvertures des lignes et des branches de `main.py` et `prediction.py` sont
affichées et écrites dans `tests/reports/coverage.json`. Les autres résultats restent
dans `tests/reports/`, le cache dans `tests/.pytest_cache/`, les temporaires dans `tests/.tmp/`.

Les tests lisent les noms, types et ordre des variables directement dans `MLmodel`,
sans copie du schéma dans les fixtures. Le calcul reste simulé pour les tests unitaires.

T08 et le contrôle Docker partagent **le même JSON et la même vérification HTTP**.
La ligne historique du score `0.17086662346702958` est absente. Une référence synthétique
a été établie avec le vrai pipeline v4 : score **`0.9206103043784433`**, classe **1**,
seuil **0.5**, tolérance absolue **`0.000001`**. Elle est figée dans
`tests/fixtures/t08_reference.json`, avec les empreintes du modèle et du JSON canonique.
Une absence ou une modification incompatible fait échouer le contrôle ; T08 n'est pas ignoré.

`tests/test_api.py` contient les tests de l'API exécutés avec pytest.
`tests/verifier_api_deployee.py` contrôle un service déjà démarré, comme le ferait
un programme appelant : il attend sa disponibilité, appelle `/schema`, puis envoie
le client fictif complet à `/predict` et vérifie le résultat de référence.
T08 réutilise ses fonctions ; le job `deploy` exécute ce même contrôle contre
le conteneur issu de `build`. Une construction Docker réussie ne garantit pas
que l'application démarre et prédit correctement : ce contrôle vérifie ces points
et fait échouer le job si nécessaire. GitHub Actions arrête ensuite le conteneur,
même en cas d'échec. Il s'agit d'une validation temporaire du déploiement,
sans hébergement permanent. Le script ne génère ni ne met à jour la référence.
## Changer de modèle : procédure manuelle

L'application ne sélectionne pas automatiquement la dernière version MLflow.
Ajouter une version dans MLflow ne change donc pas le modèle utilisé par l'API.
Pour remplacer le modèle actuellement déployé :

1. Exporter le pipeline entraîné et ses fichiers MLflow dans un dossier dédié
   `app/artifacts/<nom>/<version>/`. Renseigner `provenance.json` avec les identifiants
   réels du modèle et du run, la version, la classe positive et le seuil approuvé.
2. Modifier `ARTIFACTS_PATH` dans `app/prediction.py` et `ARTIFACTS` dans
   `tests/verifier_api_deployee.py`. Le schéma sera lu depuis le nouveau `MLmodel` ; toutes ses
   clés restent obligatoires. Vérifier que le pipeline contient le prétraitement attendu.
3. Vérifier `tests/fixtures/client_valid.json` : conserver ce client fictif si son
   schéma reste compatible ; sinon adapter les clés et les valeurs au nouveau schéma.
   Adapter aussi les règles métier dans `app/prediction.py` si nécessaire, les attentes
   de version et de nombre de variables dans les tests et `tests/verifier_api_deployee.py`, ainsi
   que les mentions de version et de schéma dans la documentation.
4. Avec le vrai pipeline sélectionné, calculer directement le résultat de ce JSON
   en dehors de l'API, puis examiner et approuver ce résultat. Mettre à jour manuellement
   `tests/fixtures/t08_reference.json` : score, classe, seuil, nom, version, identifiants
   MLflow et empreintes SHA-256 du fichier `model.pkl` et du client. L'empreinte client
   porte sur `json.dumps(client, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
   allow_nan=False).encode('utf-8')`, pas sur les octets du fichier JSON. Conserver une
   tolérance justifiée (actuellement `0.000001`) et expliquer la nouvelle référence dans `note`.
5. Relancer les tests et le contrôle Docker avec les commandes ci-dessous, reconstruire
   l'image et redémarrer l'API. Examiner les changements avant de les versionner ; le
   prochain push déclenchera `test → build → deploy` avec le nouvel artefact embarqué.

T08 vérifie une prédiction connue ; il ne mesure pas la qualité du nouveau modèle.
Évaluer cette qualité sur des données de validation avant d'approuver son remplacement.
Ne jamais modifier la référence simplement pour faire passer un test en échec.
Aucun notebook ni générateur de référence n'est fourni : cette mise à jour reste manuelle.
## Docker

Prérequis : Docker avec un moteur Linux démarré. Depuis la racine :

```powershell
docker build --tag credit-scoring:local .
docker run --detach --name scoring --publish 127.0.0.1:7860:7860 credit-scoring:local
try {
    .\.venv\Scripts\python.exe tests/verifier_api_deployee.py --url http://127.0.0.1:7860 --timeout 120
    if ($LASTEXITCODE -ne 0) { docker logs scoring; throw 'Verification API echouee' }
} finally {
    docker rm --force scoring
}
```

L'image embarque l'application et l'artefact, écoute sur `0.0.0.0:7860` à l'intérieur,
et utilise un utilisateur non root. Le port hôte est publié uniquement sur `127.0.0.1`.

## GitHub Actions

Sur les `push` et `pull_request`, `test` lance la suite et affiche la couverture.
Après son succès, `build` construit l'image et la transmet comme artefact du workflow.
`deploy` charge cette image, vérifie son identifiant, démarre le conteneur et appelle
`schema`, puis `predict`, avec la référence commune. Il affiche les journaux en cas
d'échec et supprime le conteneur dans une étape `always()`. Aucun secret n'est nécessaire
pour le modèle actuellement versionné. L'image n'est pas publiée dans un registre.

Ce démarrage est temporaire sur le runner GitHub Actions, sans hébergement permanent.
Le stockage des appels, l'analyse de dérive et le tableau de bord restent à réaliser.
