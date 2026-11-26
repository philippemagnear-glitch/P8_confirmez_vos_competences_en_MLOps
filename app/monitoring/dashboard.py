"""Interface Streamlit du suivi de scoring.

Affiche les résultats de analysis.py. Le bouton déclenche l'analyse ; le cache
de référence, le verrou et les résultats de session sont gérés ici.
Lancement : python -m streamlit run app/monitoring/dashboard.py
"""

import logging
import os
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

from app.monitoring.analysis import (
    DB_PATH, TRAIN_PATH, analyser_appels, analyser_performances, charger_reference, empreinte_reference,
)

SESSION_KEY = "rapport_scoring_evidently"
LOGGER = logging.getLogger(__name__)
reference_en_cache = st.cache_resource(max_entries=1, show_spinner=False)(charger_reference)


@st.cache_resource
def verrou_analyse():
    """Une seule analyse simultanée dans le processus Streamlit."""
    return threading.Lock()


def main():
    """Seul le bouton relance l'analyse ; les résultats restent dans la session."""
    st.set_page_config(page_title="Suivi du scoring crédit", layout="wide")
    st.title("Suivi du scoring crédit")
    scoring_url = os.environ.get("SCORING_URL", "http://localhost:7860")
    st.link_button("Ouvrir les prédictions", scoring_url)
    with st.sidebar:
        st.header("Paramètres")
        database = Path(st.text_input("Chemin SQLite", str(DB_PATH)))
        nombre = int(st.number_input("Appels par fenêtre", 50, 10000, 100, 50))
        distance = float(st.number_input("Seuil Wasserstein / Jensen–Shannon", min_value=0.001, value=0.1, step=0.01))
        ks = float(st.number_input(
            "Seuil KS global (avant correction)", min_value=0.0001, max_value=0.9999,
            value=0.05, step=0.001, format="%.4f",
            help="Divisé par le nombre de tests numériques, score compris.",
        ))
        st.caption(f"Données de référence : {TRAIN_PATH.name}")
    drift, performances = st.tabs(["Dérive et prédictions", "Performances"])
    with drift:
        afficher_drift(database, nombre, distance, ks)
    with performances:
        afficher_performances(database)


def afficher_drift(database, nombre, distance, ks):
    """Conserve la vue Evidently et son déclenchement indépendant."""
    button = st.empty()
    if button.button("Lancer l'analyse", type="primary", key="lancer"):
        button.button("Analyse en cours…", disabled=True, key="patienter")
        lock = verrou_analyse()
        if not lock.acquire(blocking=False):
            st.session_state["erreur"] = "Une analyse est déjà en cours. Réessayez ensuite."
        else:
            try:
                with st.spinner("Analyse en cours ; la première référence peut prendre plusieurs minutes."):
                    reference = reference_en_cache(empreinte_reference())
                    result = analyser_appels(database, nombre, distance, ks, reference)
                st.session_state[SESSION_KEY] = result
                st.session_state.pop("erreur", None)
            except Exception as error:
                LOGGER.error("Échec du monitoring (%s). Vérifier les sources et l'artefact.", type(error).__name__)
                st.session_state["erreur"] = "Analyse échouée : vérifiez les sources et l'artefact. Le dernier résultat réussi est conservé."
            finally:
                lock.release()
        st.rerun()
    if st.session_state.get("erreur"):
        st.error(st.session_state["erreur"])
    result = st.session_state.get(SESSION_KEY)
    if result is None:
        st.info(f"Cliquez sur « Lancer l'analyse » pour analyser les {nombre} dernières prédictions enregistrées en base.")
        return
    st.caption(f"Dernière analyse : {result['date']} — modèle v{result['version']}, seuil {result['seuil']}.")
    if result["parametres"] != (str(database.resolve()), nombre, distance, ks):
        st.warning("Paramètres modifiés : relancez l'analyse pour actualiser les résultats.")
    if result["appels"] < result["parametres"][1]:
        st.warning("Fenêtre incomplète : tous les appels disponibles ont été utilisés.")
    summary = result["synthese"]
    entries = summary.drop(index="score")
    distances_detected = entries["Résultat"].eq("Dérive détectée")
    ks_detected = entries["Détection KS"]
    predictions, service, missing_values = st.columns(3, gap="large")
    with predictions:
        st.subheader("Prédictions")
        if result["decisions"] is None:
            st.warning("données insuffisantes pour comparer les classes")
        else:
            decisions = result["decisions"]
            scores, rates = decisions["Score moyen"], decisions["Défauts prédits (%)"]
            st.metric("Score moyen des appels API", f"{scores['API']:.3f}", f"{scores['API'] - scores['Train']:+.3f} vs dataset de référence", delta_color="off")
            st.metric("Défauts prédits des appels API", f"{rates['API']:.1f} %", f"{rates['API'] - rates['Train']:+.1f} points vs dataset de référence", delta_color="off")
    with service:
        st.subheader("Service")
        operations = result["operations"]
        for label in operations.index:
            value = operations.loc[label, "Actuel"]
            delta = operations.loc[label, "Écart"]
            text = "données insuffisantes" if pd.isna(value) else f"{value:.2f}"
            unit = "points" if "%" in label else "ms"
            evolution = "comparaison indisponible" if pd.isna(delta) else f"{delta:+.2f} {unit} vs précédent"
            st.write(f"**{label} : {text}**")
            st.caption(evolution)
        if result["precedents"] < result["parametres"][1] or result["appels"] < result["parametres"][1]:
            st.caption("Deux fenêtres complètes sont nécessaires pour comparer le service.")
    with missing_values:
        st.subheader("Valeurs manquantes")
        if result["clients"] == 0:
            st.info("Aucune prédiction réussie comparable.")
        else:
            missing_count = int((result["clients"] - entries["Renseignées API"]).sum())
            api_rate = missing_count / (result["clients"] * len(entries)) * 100
            reference_rate = entries["Null train (%)"].mean()
            st.metric(
                "Valeurs absentes dans les appels API",
                f"{missing_count:,}".replace(",", " "),
                f"{api_rate - reference_rate:+.2f} points vs dataset de référence", delta_color="off",
            )
            st.caption(f"Appels API : {api_rate:.2f} % · Dataset de référence : {reference_rate:.2f} %.")
    st.divider()
    st.subheader("Dérive des 20 variables les plus importantes")
    importance = result["importance"]
    if importance.empty:
        st.info("Importance par gain indisponible pour cet artefact.")
    else:
        top = importance.nlargest(20).to_frame("Importance par gain (%)").join(
            entries[["Résultat", "Méthode", "Détection KS", "P-value KS"]]
        )
        top["Détection KS"] = [
            "Non applicable" if method == "jensenshannon" else
            "Indisponible" if pd.isna(p_value) else "Oui" if detected else "Non"
            for method, detected, p_value in zip(
                top["Méthode"], top["Détection KS"], top["P-value KS"]
            )
        ]
        st.caption(f"Part cumulée du gain total : {top['Importance par gain (%)'].sum():.1f} %.")
        st.dataframe(top.drop(columns=["Méthode", "P-value KS"]), width="stretch", column_config={
            "Importance par gain (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Résultat": st.column_config.TextColumn(label="Dérive Wasserstein / Jensen–Shannon"),
        })
    st.header("Récapitulatif des dérives")
    recaps = [
        ("Wasserstein / Jensen–Shannon", distances_detected, ["Méthode", "Distance", "Seuil distance", "Renseignées API"]),
        ("KS — seuil global corrigé", ks_detected, ["P-value KS", "Seuil KS", "Renseignées API"]),
        ("Détectées par les deux méthodes", distances_detected & ks_detected,
         ["Distance", "Seuil distance", "P-value KS", "Seuil KS"]),
    ]
    for title, mask, columns in recaps:
        detected = entries.loc[mask, columns]
        st.subheader(f"{title} · {len(detected)} variable(s)")
        if detected.empty:
            st.info("Aucune variable signalée.")
        else:
            order = "P-value KS" if title.startswith("KS") else "Distance"
            # Les distances de méthodes différentes ne sont pas classées ensemble.
            sorted_rows = detected.sort_values(["Méthode", "Distance"], ascending=[True, False]) if "Méthode" in columns else detected.sort_values(order, ascending=order == "P-value KS")
            st.dataframe(
                sorted_rows, width="stretch",
                column_config={column: st.column_config.NumberColumn(
                    label=column.replace("API", "appels API"), format="%.4g"
                ) for column in columns if column != "Méthode"},
            )


def afficher_performances(database):
    """Affiche les agrégats SQLite ; seul le bouton actualise l'instantané."""
    st.subheader("Évolution des performances")
    nombre = int(st.number_input(
        "Nombre d'appels par fenêtre", min_value=50, max_value=500, value=100, step=50,
        help="Les temps utilisent seulement les prédictions réussies disposant d'une mesure.",
        key="taille_performances",
    ))
    if st.button("Actualiser les performances"):
        try:
            st.session_state["performances"] = analyser_performances(database, nombre)
            st.session_state.pop("erreur_performances", None)
        except Exception as error:
            LOGGER.error("Lecture des performances impossible (%s).", type(error).__name__)
            st.session_state["erreur_performances"] = "Lecture impossible : vérifiez le chemin et le schéma SQLite. Le dernier résultat est conservé."
    if st.session_state.get("erreur_performances"):
        st.error(st.session_state["erreur_performances"])
    report = st.session_state.get("performances")
    if report is None:
        st.info("Cliquez sur « Actualiser les performances » pour lire les mesures enregistrées.")
        return
    if report["parametres"] != (str(database.resolve()), nombre):
        st.warning("Paramètres modifiés : actualisez les performances.")
    current, previous = report["actuel"], report["precedent"]
    st.caption(f"Lecture du {report['date']} · Périodes en heure de Paris.")
    st.write(f"**Actuel :** {current['Appels']} appels · {current['Début']} → {current['Fin']}")
    st.write(f"**Précédent :** {previous['Appels']} appels · {previous['Début']} → {previous['Fin']}")
    if current["Appels"] == 0:
        st.info("Aucune mesure disponible.")
        return
    if not report["comparaison"]:
        st.info("Deux fenêtres complètes sont nécessaires pour afficher les écarts.")
    for title, metrics in [
        ("Temps de réponse", [("Latence médiane (ms)", "latency_ms"), ("Latence P95 (ms)", "latency_ms")]),
        ("Calcul et ressources", [("Inférence médiane (ms)", "inference_ms"), ("Temps CPU médian (ms)", "cpu_time_ms")]),
        ("Préparation", [("Préparation médiane (ms)", "preparation_ms")]),
    ]:
        st.subheader(title)
        for column, (label, measurement) in zip(st.columns(2), metrics):
            with column:
                value, old = current[label], previous[label]
                st.metric(label, "Données insuffisantes" if pd.isna(value) else f"{value:.2f}")
                if report["comparaison"] and pd.notna(value) and pd.notna(old):
                    difference = round(value - old, 2)
                    text = "0,00" if difference == 0 else f"{difference:+.2f}".replace(".", ",")
                    st.caption(f"Différence avec les {previous['Appels']} appels précédents : {text} ms.")
                count = current[f"Mesures {measurement}"]
                st.caption(f"{count} mesure(s) utilisée(s)")
                if label == "Latence P95 (ms)" and 50 <= count < 100:
                    st.caption("P95 provisoire : moins de 100 mesures.")
    if sum(current[f"Mesures {name}"] for name in ["latency_ms", "inference_ms", "cpu_time_ms"]) == 0:
        st.info("Aucune mesure de performance disponible sur les appels actuels.")


if __name__ == "__main__":
    main()
