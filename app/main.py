import json
import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router

app = FastAPI(title="Ragamuffin", version="0.1.0")
app.include_router(router)
logger = logging.getLogger("uvicorn.error")


@app.middleware("http")
async def log_request(request: Request, call_next):
    start = perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - keep unexpected backend details private
        request.state.failure = "unexpected_error"
        response = JSONResponse({"detail": "Internal server error"}, status_code=500)

    route = getattr(request.scope.get("route"), "path", None)
    if response.status_code >= 400 or route not in {"/readyz", "/healthz"}:
        event = {
            "event": "http_request",
            "route": route,
            "status": response.status_code,
            "duration_ms": round((perf_counter() - start) * 1000, 2),
        }
        failure = getattr(request.state, "failure", None)
        if failure:
            event["failure"] = failure
        logger.log(
            logging.WARNING if response.status_code >= 500 else logging.INFO,
            json.dumps(event),
        )
    return response


@app.get("/healthz")
def health():
    return {"status": "ok"}
