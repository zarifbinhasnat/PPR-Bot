"""Health/readiness endpoints.

A common production pattern: separate *liveness* ("is the process up?") from
*readiness* ("are the models loaded and is it safe to send traffic?").
Orchestrators (Docker/Kubernetes/load balancers) poll these to decide when to
route requests or restart a stuck instance.
"""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness: the process is running and can respond."""
    return {"status": "ok"}


@router.get("/readiness")
def readiness(request: Request) -> dict:
    """Readiness: the orchestrator (with models/indexes) is loaded."""
    ready = getattr(request.app.state, "orchestrator", None) is not None
    return {"ready": ready}
