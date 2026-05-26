"""API layer: FastAPI app exposing scored opportunities.

Productization note: this is the product API surface.  Phase 1 ships a single
``GET /opportunities`` endpoint returning filtered/scored events.  Future phases
add authentication, per-user profile-based scoring, webhooks, and SSE streams.
"""

from prediction_alpha.api.app import create_app

__all__ = ["create_app"]
