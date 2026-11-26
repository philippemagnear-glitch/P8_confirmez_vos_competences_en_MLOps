"""Traitement commun aux appels Gradio et au diagnostic sans serveur."""

import logging
from datetime import datetime, timezone
from time import process_time
from uuid import uuid4

from app.prediction import InputValidationError


LOGGER = logging.getLogger(__name__)


def process_api_call(model, store, operation, client=None, *, request_id=None):
    """Exécute le calcul et trace son issue dans SQLite, sans dépendre de Gradio.

    Le diagnostic fournit son identifiant avant le calcul pour pouvoir supprimer
    uniquement sa ligne, y compris lorsque la prédiction échoue.
    """
    if operation not in {"predict", "schema"}:
        raise ValueError("Opération inconnue.")
    cpu_started = process_time()
    request_id = request_id or str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    measurements = {}
    status, result, reason = "error", None, None
    try:
        result = (
            model.predict(client, measurements=measurements)
            if operation == "predict" else model.schema()
        )
        if operation == "predict":
            result = {**result, "request_id": request_id}
        status = "success"
        return result
    except InputValidationError as error:
        status, reason = "rejected", str(error)
        raise
    except Exception as error:
        # Ne pas journaliser l'exception brute : elle peut contenir les entrées.
        message = (
            "La prédiction a échoué. Consulter les logs du serveur."
            if operation == "predict" else
            "La lecture du schéma a échoué. Consulter les logs du serveur."
        )
        reason = f"{message} ({type(error).__name__})"
        LOGGER.error("Échec de %s (%s).", operation, type(error).__name__)
        raise
    finally:
        measurements.update(
            cpu_time_ms=(process_time() - cpu_started) * 1000,
            execution_device=model.execution_device if operation == "predict" else None,
            model_backend=getattr(model, "backend", None),
        )
        store.record_api_call(
            operation=operation, status=status, started_at=started_at,
            inputs=client, result=result, reason=reason,
            request_id=request_id, measurements=measurements,
        )
