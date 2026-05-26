"""Feedback and Self-Improvement Loop for Prediction Alpha Engine.

This module closes the loop:
- Log real outcomes (resolutions) when markets settle.
- Track calibration (Brier, log-loss, etc.) over time.
- Provide signals for simple "retraining" (prompt tuning, weight adjustment, or model retrain).

Designed to be lightweight and sovereign. Can be extended with a real job queue later.
"""