# Bilan des optimisations du scoring crédit

Objectif : réduire les temps de préparation, d'inférence et de réponse, en vérifiant que les prédictions restent cohérentes.

## Mesure et suivi des essais

| Outil ou méthode | Utilisation dans le projet |
|---|---|
| **Mesures Python et SQLite** | Des chronomètres Python mesurent la préparation des données, l'inférence et le temps complet entre l'envoi d'un appel et la réception du résultat. Ces durées sont enregistrées dans SQLite et affichées dans Streamlit pour comparer les lots d'appels. |
| **cProfile et SnakeViz** | cProfile indique combien de fois chaque fonction est appelée et combien de temps elle prend, seule ou avec les fonctions qu'elle appelle. SnakeViz affiche ces résultats sous forme de rectangles et de tableau pour repérer les traitements coûteux. |
| **Comparaison des prédictions** | Les mêmes clients sont rejoués avec le même CSV et la même graine `seed 42`. Dans SQLite, les appels sont associés par leurs données d'entrée, la version du modèle et le seuil. On compare ensuite leurs probabilités de défaut et leurs décisions. |
| **Traçabilité** | SQLite conserve les appels, les scores, les temps et le moteur utilisé. Git permet de retrouver le code : [baseline initiale](baseline_avant_optimisation.md), profilage `23cbdc5`, première optimisation `f194a93`, version ONNX nettoyée `9a469b1`. |

## Étapes, résultats et décisions

| Étape | Ce que nous avons changé et pourquoi | Résultat observé | Ce que nous avons gardé |
|---|---|---|---|
| **1. Préparer les données avec Pandas** | Avant une prédiction, le code convertissait séparément le type de nombreuses colonnes. Nous avons regroupé les colonnes de même type pour éviter de répéter ces opérations. | Au départ, la préparation prenait **88,7 ms**, dont **85,8 ms dans `astype`**. Après modification, le gros bloc `astype` disparaît et la fonction de prédiction passe de **177 à 94,5 ms** sur les profils présentés. | Cette première amélioration a ensuite été remplacée par la préparation NumPy de l'étape 6. |
| **2. Simplifier l'enregistrement SQLite** | Le code écrivait dans la base, puis effectuait une autre écriture uniquement pour enregistrer le temps pris par le stockage. Nous avons supprimé cette deuxième écriture de diagnostic. | Le stockage prenait **92,3 ms** dans le profil qui a motivé cette modification. Sur le passage présenté après simplification, il prend **17,8 ms** ; le traitement complet passe de **123 à 35,8 ms**. | Les prédictions et leurs mesures restent enregistrées. Seule la mesure dédiée au temps de stockage a été retirée. |
| **3. Éviter de recréer le client HTTP** | Le simulateur répétait la création des outils nécessaires pour envoyer les demandes et recevoir les réponses, notamment la configuration des certificats HTTPS. Il les prépare maintenant une fois et les réutilise pour tout le lot. | Temps de réponse médian observé : **624,84 → 107,66 ms**, sur un lot initial de 100 appels puis un essai de 10 appels. Les 10 clients comparés ont conservé exactement les mêmes scores. | Réutiliser la session HTTP et la configuration TLS pendant le lot, en gardant la vérification des certificats. |
| **4. Comparer 1 et 16 threads CPU** | Nous avons autorisé LightGBM à utiliser un seul thread, puis jusqu'à 16, pour voir si répartir le travail sur plusieurs unités de calcul du processeur accélérait une prédiction. | Aucun gain notable constaté. Le P95, qui décrit les appels les plus lents, était même légèrement plus élevé avec 16 threads. | Un thread par défaut dans la configuration finale ONNX. L'essai LightGBM n'a pas donné de raison d'en mobiliser davantage pour cet usage. |
| **5. Exécuter le modèle avec ONNX Runtime** | Sans réentraîner le modèle, nous avons traduit ses étapes et ses arbres dans le format ONNX. ONNX Runtime effectue alors le remplacement des valeurs manquantes et le calcul des probabilités à la place de scikit-learn et LightGBM. | Temps d'inférence médian : **2,442 → 0,111 ms** dans le test séparé, et **3,04 → 0,22 ms** pendant les appels API. | ONNX Runtime pour exécuter les prédictions sur CPU. Le modèle original reste nécessaire à la conversion et au monitoring. |
| **6. Fournir directement les données à ONNX avec NumPy** | ONNX attend un tableau numérique. Nous avons supprimé les tableaux Pandas intermédiaires et rempli directement ce tableau dans le bon ordre. La liste et le type des variables sont préparés une fois au démarrage. | Temps de préparation médian : **7,05 → 0,84 ms**. Temps d'inférence médian : **0,31 → 0,21 ms** sur les fenêtres affichées. | Un tableau NumPy créé pour chaque client, avec les mêmes contrôles des variables, des types et des valeurs autorisées. |
| **7. Examiner le GPU et la quantification** | Nous avons envisagé d'utiliser la carte graphique ou de réduire la précision numérique de certains calculs pour les accélérer. Ces pistes n'ont pas été mises en œuvre. | Aucun essai mesuré : nous ne leur attribuons aucun gain. | Le processeur CPU et ONNX sans quantification. |

Les durées des étapes 1 et 2 sont lues dans les captures cProfile avant/après :
elles montrent les temps des fonctions sur ces passages, avec le profiler actif.
Elles ne sont pas les médianes de 100 appels et peuvent varier avec les caches et les accès disque.

## Contrôle des prédictions

| Comparaison | Clients comparés | Différence maximale entre les probabilités | Décisions différentes au seuil de 50 % | Ce que cela montre |
|---|---:|---:|---:|---|
| **ONNX optimisé / LightGBM** | 100 | **0,000000219**, soit environ **0,000022 point de pourcentage** | **0** | Pour les mêmes clients, ONNX donne des probabilités presque identiques à celles de LightGBM. Les 100 décisions sont conservées. |
| **ONNX avant / après simplification de la préparation** | 80 | **0 : scores strictement identiques** | **0** | Remplacer la préparation Pandas par NumPy n'a changé aucun score ni aucune décision sur les 80 clients communs disponibles. |

Ces comparaisons vérifient que l'exécution optimisée conserve les résultats sur les clients testés.
Elles ne mesurent pas la justesse des prédictions face aux défauts réellement survenus.

## Configuration finale

| Sujet | Configuration retenue |
|---|---|
| Prédiction | **ONNX Runtime CPU, un thread**, avec imputation et classification dans la même session. |
| Préparation | Validation des 310 variables, schéma préparé au démarrage, tableau NumPy float32 construit directement. |
| Chargement | Pipeline MLflow chargé et converti **une seule fois au démarrage**, sans modifier le modèle sauvegardé. |
| Bibliothèques | ONNX Runtime 1.32.0 et NumPy 2.5.4 ; LightGBM, scikit-learn et MLflow servent à la conversion et au monitoring. Versions complètes dans [requirements.txt](../requirements.txt). |
| Monitoring | SQLite conserve le moteur, les scores, les décisions et les durées. Le pipeline LightGBM original fournit la référence et les importances des variables. |
| Résultat global observé | Latence médiane initiale **626,45 ms**, dernière valeur **105,62 ms**. |

## Déroulement du travail

Nous avons commencé par découper les principales étapes avec des chronomètres Python pour
comprendre où passait le temps. Nous avons ensuite utilisé cProfile et son interface visuelle
SnakeViz pour examiner les fonctions de notre code Python et améliorer la préparation Pandas.

Le profilage dans Gradio ne donnait pas une hiérarchie d'appels exploitable. Nous avons donc
poursuivi le découpage des échanges avec des chronomètres Python. Cela a révélé le coût des
créations répétées de clients HTTP et de contextes TLS. Leur réutilisation a réduit le temps de réponse.

Enfin, nous avons testé ONNX pour accélérer l'inférence, puis NumPy pour simplifier la préparation.
La latence médiane observée est passée d'environ **626 à 106 ms**. Sur les 100 derniers clients
comparés à LightGBM, les probabilités sont presque identiques et aucune décision n'a changé.
