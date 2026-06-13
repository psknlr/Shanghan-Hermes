"""Stdlib-only web console for Hermes-Shanghanlun.

service.py     framework-agnostic API surface (testable without HTTP)
http_server.py http.server handler + static SPA + JSON routes
static/        self-contained single-page app (vanilla JS, no build, no CDN)
"""
from .service import ServiceContext, get_service

__all__ = ["ServiceContext", "get_service"]
