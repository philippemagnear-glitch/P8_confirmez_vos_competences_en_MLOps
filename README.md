# Scoring crédit et suivi de dérive — Projet 8 MLOps

Application Gradio utilisant le pipeline **LightGBM `credit_scoring_lightgbm`, version 4**.
Elle reçoit les données préparées d'un client, retourne un score et une classe, puis
trace l'appel dans SQLite. Un dashboard Streamlit analyse cet historique avec Evidently.
Le notebook `notebooks/01_premiers_pas_data_drift.ipynb` conserve le PoC exploratoire.

Stack : Python 3.13, Gradio, FastAPI, MLflow, LightGBM, pandas, Evidently, Streamlit,
pytest et Docker. Les dépendances directes sont centralisées dans `requirements.txt`.

## Organisation

```text
Client → Gradio → Validation → Pipeline entraîné → Score et classe
            └──────────────────────────────────→ SQLite
                                                   ↓ lecture seule
                                              Monitoring → Streamlit
```

| Fichier ou dossier | Rôle |
|---|---|
| `app/main.py` | Interface Gradio, endpoints et déclenchement du stockage. |
| `app/prediction.py` | Chargement du modèle, validation et prédiction. |
| `app/storage.py` | Historique SQLite, sans bloquer la réponse si le stockage échoue. |
| `app/artifacts/credit_scoring_lightgbm/4/` | Pipeline, signature MLflow et provenance. |
| `monitoring.py` | Lecture SQLite, analyses Evidently et indicateurs opérationnels. |
| `dashboard.py` | Affichage Streamlit, cache et déclenchement manuel de l'analyse. |
| `scripts/simuler_appels_api.py` | Simulation d'appels depuis le CSV de clients préparés. |
| `tests/` | Tests, JSON synthétique de référence et contrôles du déploiement. |
| `compose.yaml`, `Dockerfile` | Deux services locaux utilisant la même image. |
| `.github/workflows/tests-api.yml` | Pipeline `test → build → deploy`. |

Le modèle est chargé une fois par processus, sans serveur MLflow ; les prédictions partagent une file.
Le pipeline contient l'imputation et LightGBM ; jointures et encodage sont réalisés en amont.

## Installation et lancement local

Prérequis : Python 3.13 et `uv`. Depuis la racine, dans PowerShell :

```powershell
# Uniquement si .venv n'existe pas encore :
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\Activate.ps1
python -m app.main
```

Dans un deuxième terminal, avec `.venv` activé :

```powershell
python -m streamlit run dashboard.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

- Prédictions : **http://localhost:7860**.
- Dashboard : **http://localhost:8501**.
- Santé de l'API : **http://localhost:7860/health**.
- Santé de Streamlit : **http://localhost:8501/_stcore/health**.

Les boutons ouvrent l'autre application dans un nouvel onglet, sans démarrer son serveur.
Arrêter chaque serveur avec `Ctrl+C`.
`/health` confirme que le serveur répond après chargement du modèle ; il ne teste ni une
prédiction, ni SQLite, ni Evidently. Aucune authentification n'est configurée.

| Variable d'environnement | Valeur par défaut hors Docker | Usage |
|---|---|---|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Adresse d'écoute de Gradio. |
| `GRADIO_SERVER_PORT` | `7860` | Port Gradio. |
| `API_CALLS_DB_PATH` | `<racine>/data/api_calls.sqlite3` | Chemin de SQLite. |
| `MONITORING_URL` | `http://localhost:8501` | Adresse du bouton vers le dashboard. |
| `SCORING_URL` | `http://localhost:7860` | Adresse du bouton vers les prédictions. |

Pour un futur hébergement, définir par exemple `$env:MONITORING_URL="https://dashboard.exemple.fr"`
et `$env:SCORING_URL="https://scoring.exemple.fr"` avant le lancement. Compose transmet ces variables.
Ces adresses accessibles au navigateur modifient les liens, pas les ports d'écoute.

## Docker Compose et persistance

Avec Docker Desktop démarré et les ports 7860/8501 libres :

```powershell
docker compose up --build -d --remove-orphans
docker compose logs -f
# Arrêter et supprimer les conteneurs, en conservant les données locales :
docker compose down
```

Gradio et Streamlit gardent les mêmes adresses, sans Nginx ni préfixe `/monitoring`.
Ports publiés sur `127.0.0.1` ; `--remove-orphans` retire les anciens services du projet Compose.

Le dossier local `data/` est monté dans les deux conteneurs : l'API écrit dans SQLite,
Streamlit lit le dossier en lecture seule. Supprimer ou remplacer les conteneurs conserve
`data/api_calls.sqlite3`. Une base située dans un ancien volume nommé n'est pas migrée automatiquement.
Le modèle est embarqué dans l'image ; les datasets et les tests ne le sont pas.
Pour analyser la dérive, fournir **`data/processed/application_train_final_encoded.csv`**.

L'image démarre avec `python -m app.main`, écoute sur `0.0.0.0:7860` sous l'UID non root `10001`.
SQLite est à `/app/data/api_calls.sqlite3` : le montage doit être accessible à cet utilisateur.
Sans stockage monté, supprimer le conteneur supprime sa base.

## Utiliser l'API et simuler des appels

`/predict` reçoit un client préparé avec **310 clés obligatoires** ; `/schema` expose les noms,
l'ordre, les types et les valeurs manquantes acceptées. L'API exige aussi les clés déclarées
facultatives dans la signature MLflow v4. Toute clé absente est nommée avant le calcul ; les
clés inconnues, dont `TARGET` et `SK_ID_CURR`, sont refusées. Les colonnes sont réordonnées.

Les `null` numériques sont imputés ; **zéro reste zéro**. Les booléens attendent `true` ou `false`
et refusent `null`. Chaînes numériques et nombres non finis sont refusés. Le revenu nul reste
accepté ; les soldes signés sont conservés, et tous les rapports ne sont pas bornés à 1.
Les autres règles figurent dans `app/prediction.py` et le [plan de tests](docs/plan_tests_api_scoring.md).
Score dans `[0, 1]` ; classe `1` lorsque `score >= threshold` ; seuil actuel : `0.5`.

L'interface peut charger 50 clients du CSV facultatif `data/processed/application_test_final_encoded.csv`.
Son absence n'empêche ni la saisie JSON ni les appels depuis un programme tiers :

```python
import json
from pathlib import Path
from gradio_client import Client
from gradio_client.exceptions import AppError

client = Client("http://localhost:7860")
try:
    schema = client.predict(api_name="/schema")
    payload = json.loads(Path("tests/fixtures/client_valid.json").read_text(encoding="utf-8"))
    print(client.predict(client=payload, api_name="/predict"))
except AppError as error:
    print(error)
finally:
    client.close()
```

La réponse contient `score`, `prediction`, `threshold`, `model_name`, `model_version`
et `request_id`, qui relie la réponse à sa ligne SQLite.
**Use via API** décrit le protocole : une entrée invalide produit `AppError` avec le motif ;
une panne interne produit un message générique, sans nouveau score ni détails sensibles.
Le programme appelant doit aussi gérer les erreurs réseau. En HTTP brut, le POST
`/gradio_api/call/predict` fournit un `event_id`, puis le GET associé transmet un événement SSE
`complete` ou `error` : le statut du POST seul ne garantit pas le succès. Ce n'est pas `POST /predict` en REST.

Pour alimenter SQLite, avec l'API démarrée :

```powershell
python scripts/simuler_appels_api.py
python scripts/simuler_appels_api.py --count 100 --seed 42
python scripts/simuler_appels_api.py --count 300 --seed 42
```

Le script tire 100 lignes distinctes du CSV de test préparé, préserve types, `null` et zéros,
puis envoie les appels espacés de 0,2 à 1 seconde. `--csv` et `--url` changent les sources et la destination.
Les clients peuvent revenir entre lots ; une panne réseau arrête le lot. Ce n'est pas un test de charge.

Toutes les mesures sont conservées dans la table SQLite `api_calls`, sans export CSV.
Après chaque réussite, le script transmet la latence complète à `POST /measurements`,
avec le `request_id` de la prédiction. L'envoi de la mesure
est exclu de cette latence. Une erreur de transmission arrête le script sans rejouer la prédiction.
Les appels manuels restent enregistrés mais n'ont pas cette mesure côté client.
Les rejets et pannes restent tracés côté serveur, sans latence client corrélée.
SQLite conserve également `inference_ms` (imputation et LightGBM), `cpu_time_ms`
et `execution_device`. Le temps CPU concerne le processus serveur pendant le traitement,
pas l'utilisation CPU globale de la machine. Le backend CPU actuel n'utilise
pas le GPU. Mesures et limites de comparaison : [baseline avant optimisation](docs/baseline_avant_optimisation.md).

## Historique SQLite et monitoring

La table `api_calls` conserve UUID, date UTC, opération, statut et durée. Une prédiction
réussie conserve aussi les entrées JSON avec leurs `null`, le score, la classe, le seuil et
la version. Un rejet ou une panne conserve le motif sans recopier les entrées ni le score.
Les appels `/schema` sont tracés sans données client. Chaque écriture ouvre et ferme sa connexion.
Une erreur SQLite est journalisée sans modifier la réponse : un succès ne garantit pas son stockage.
Les JSON mal formés rejetés par l'interface et les requêtes refusées avant les fonctions
applicatives ne sont pas enregistrés. Il n'existe ni purge ni filtre automatique des clients réels :
le stockage intégral est destiné aux données synthétiques du TP. Bases et datasets sont ignorés par Git.

```powershell
python -m sqlite3 data/api_calls.sqlite3 "SELECT operation, status, score, duration_ms FROM api_calls ORDER BY rowid DESC LIMIT 5;"
python -m sqlite3 data/api_calls.sqlite3 "SELECT timestamp_utc, inference_ms, latency_ms, cpu_time_ms FROM api_calls WHERE operation='predict' ORDER BY rowid DESC LIMIT 5;"
```

Le bouton **Lancer l'analyse** compare les réussites de même version et seuil, parmi les
100 derniers appels `/predict`, au dataset de référence préparé, avant imputation.
Les 100 appels précédents servent au suivi opérationnel. La taille des fenêtres est configurable.

- **Distributions** : Wasserstein pour les numériques, Jensen–Shannon pour les catégories,
  seuil initial commun `0.1` ; KS pour les numériques, seuil global `0.05` corrigé par Bonferroni,
  score compris. Trois tableaux présentent les alertes des distances, de KS et leur intersection.
- **Valeurs manquantes** : suivi séparé des proportions, sans alerte automatique.
- **Prédictions** : score moyen et proportion de classes `1`, comparés aux scores du modèle
  sur le dataset de référence. Les scores sont aussi analysés par Evidently.
- **Variables importantes** : top 20 par gain LightGBM normalisé ; ce pourcentage représente
  une part du gain total d'entraînement, pas la part expliquée d'une prédiction individuelle.
- **Service** : rejets et pannes sur tous les appels ; médiane et P95 sur les réussites.
  Les durées excluent réseau, file Gradio et écriture SQLite. Les appels non tracés sont absents.

Sous 50 observations utilisables, l'indicateur concerné est indisponible. Les écarts du service
exigent deux fenêtres complètes. Les seuils sont exploratoires : une dérive ne prouve pas une
baisse de performance et des défauts prédits ne sont pas des défauts réels. Les clients répétés
et la comparaison dataset de référence/appels issus du test limitent l'interprétation temporelle.

La première préparation peut prendre plusieurs minutes. Le cache mémoire dépend du contenu du CSV
et du modèle ; un changement impose son recalcul. Les calculs ne repartent qu'au clic.
Les résultats restent dans la session et peuvent disparaître au rechargement ; une erreur conserve
le dernier succès. Aucun planificateur ni export automatique n'est configuré.

## Suivi des performances

L'onglet **Performances** lit SQLite au clic sur **Actualiser les performances**,
sans charger le dataset de référence ni relancer Evidently. Choisir entre **50 et 500
appels par fenêtre** (100 par défaut). Les derniers appels sont comparés aux précédents,
sans chevauchement. Les écarts exigent deux fenêtres complètes.
Les rejets et pannes utilisent tous les appels `/predict` ; les médianes et le P95
utilisent uniquement les réussites avec une mesure valide. Chaque carte affiche son effectif.
Sous 50 observations, l'indicateur est indisponible ; entre 50 et 99 latences, le P95
est provisoire. Les courbes suivent des groupes d'appels, pas des intervalles de temps égaux.
Le tableau détaille les périodes, versions et effectifs. Les anciennes mesures absentes
ne sont pas remplacées par zéro. Pour démontrer un gain, comparer les mêmes clients
sans profiler, dans les mêmes conditions ; ces courbes seules ne prouvent pas une optimisation.

## Tests et pipeline CI/CD

```powershell
python -m pytest
# Contrôle HTTP d'une API déjà démarrée avec le vrai modèle :
python tests/verifier_api_deployee.py --url http://localhost:7860
# Inclure T13 après construction de l'image :
$env:SCORING_DOCKER_IMAGE = 'credit-scoring:local'
python -m pytest
Remove-Item Env:SCORING_DOCKER_IMAGE
```

Les tests unitaires simulent le calcul, les tests d'intégration utilisent des serveurs et bases
isolés ; aucun CSV local n'est requis. Les tests de navigation vérifient les liens configurables
avec Gradio et Streamlit AppTest, et les tests de santé contrôlent `/health`. Le plan couvre T01–T13.
T11/T12 vérifient les traces et les erreurs SQLite ; T13 remplace un conteneur avec le même volume,
et reste ignoré sans `SCORING_DOCKER_IMAGE`. Ses ressources temporaires sont supprimées après contrôle.
La détection de dérive complète n'est pas couverte par ces tests automatisés.

T08 utilise le JSON synthétique complet `client_valid.json` et le modèle v4 : score attendu
**0.9206103043784433**, classe **1**, seuil **0.5**, tolérance absolue **0.000001**.
La ligne historique donnant `0.17086662346702958` était absente ; cette référence synthétique
la remplace. Une empreinte incompatible ou un fichier manquant fait échouer le test.
T08 et le contrôle du conteneur partagent `verifier_api_deployee.py` et `t08_reference.json`.

La couverture des **lignes et branches** de `app/main.py` et `app/prediction.py` est affichée
et écrite dans `tests/reports/coverage.json`. Rapports, cache et temporaires sont ignorés par Git.
Dans Actions, job `test`, ouvrir « Couverture des lignes et des branches par fichier ».
La couverture ne porte pas sur tout le projet ; le JSON n'est pas publié en artefact.

Sur les push et pull requests, Ubuntu 24.04 exécute successivement :

1. `test` : installation et pytest ; un échec bloque la suite.
2. `build` : construction et transmission de l'image comme artefact, conservé un jour.
3. `deploy` : contrôle de l'identité de cette image, démarrage sur la boucle locale, `/schema`
   puis `/predict` avec le vrai modèle, santé des deux services et lecture SQLite depuis Streamlit.
   T13 est obligatoire. Les logs sont affichés en cas d'échec ; les deux conteneurs sont supprimés même en échec.

Déploiement temporaire : ni hébergement permanent, ni registre, ni historique entre runners.
Le CSV de référence est absent de CI : la dérive complète n'y est pas calculée.
Les contrôles du runner utilisent la bibliothèque standard, sans réinstaller pytest.
Aucun secret n'est nécessaire pour l'artefact versionné.

## Remplacer le modèle

Une nouvelle version MLflow n'est jamais adoptée automatiquement :

1. Exporter le pipeline approuvé dans `app/artifacts/<nom>/<version>/`, avec sa signature et sa
   provenance : identifiants MLflow, source, version et seuil. Vérifier les dépendances.
2. Modifier `ARTIFACTS_PATH` dans `app/prediction.py` et `ARTIFACTS` dans
   `tests/verifier_api_deployee.py`. La classe positive est codée à **1** : modifier uniquement
   `positive_class` dans les métadonnées ne suffit pas.
3. Adapter le schéma, le JSON client, les règles métier et les tests actuellement liés aux
   310 variables et à la v4. Le monitoring exige une référence compatible et des appels de même version/seuil.
4. Calculer hors API le résultat du JSON avec le vrai pipeline et l'approuver avant de mettre à jour
   `t08_reference.json` : score, classe, seuil, nom/version, identifiants, empreintes SHA-256 du modèle
   et du JSON canonique. Justifier la tolérance et la note ; aucun générateur de référence n'est fourni.
5. Relancer les tests, reconstruire l'image, contrôler l'API et T13, puis redémarrer les services.
   Versionner ensemble le modèle, sa provenance et les adaptations. Le cache du monitoring sera invalidé.

Le JSON canonique est obtenu par `json.dumps(client, sort_keys=True, separators=(',', ':'),
ensure_ascii=True, allow_nan=False).encode('utf-8')`. Ne pas modifier la référence pour masquer
un échec. T08 vérifie une non-régression technique ; évaluer séparément la qualité du modèle.
