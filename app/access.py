"""Direct loopback-only policy, not authentication or reverse-proxy support."""
import ipaddress
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROXY_HEADERS = {b"forwarded", b"x-forwarded-for", b"x-forwarded-host", b"x-forwarded-proto",
                 b"x-forwarded-port", b"x-real-ip"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def authority(value, scheme):
    """Parse a strict origin authority; reject userinfo, paths and ambiguous ports."""
    if scheme not in {"http", "https"} or not value or any(c.isspace() for c in value):
        raise ValueError("Invalid authority")
    parsed = urlsplit(f"{scheme}://{value}")
    if parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Invalid authority")
    host = parsed.hostname
    port = parsed.port
    if host not in LOCAL_HOSTS or (port is not None and not 1 <= port <= 65535):
        raise ValueError("Invalid local host/port")
    canonical = f"[{host}]" if host == "::1" else host
    if port is not None:
        canonical += f":{port}"
    if value.lower() != canonical:
        raise ValueError("Noncanonical authority")
    return scheme, host, port if port is not None else (443 if scheme == "https" else 80)


def access_denial(scope):
    headers = scope.get("headers", [])
    if any(name.lower() in PROXY_HEADERS for name, value in headers):
        return 400, "Forwarded/proxy headers are unsupported in the local profile"
    peer = scope.get("client")
    try:
        local_peer = bool(peer) and ipaddress.ip_address(peer[0]).is_loopback
    except ValueError:
        local_peer = False
    if not local_peer:
        return 403, "Local profile requires a direct loopback peer"
    hosts = [value for name, value in headers if name.lower() == b"host"]
    try:
        if len(hosts) != 1:
            raise ValueError("Exactly one Host required")
        target = authority(hosts[0].decode("ascii"), scope.get("scheme", "http"))
    except (ValueError, UnicodeError):
        return 400, "Invalid local Host header"
    if scope["method"] not in SAFE_METHODS:
        origins = [value for name, value in headers if name.lower() == b"origin"]
        if origins:
            try:
                if len(origins) != 1:
                    raise ValueError("Multiple origins")
                origin = urlsplit(origins[0].decode("ascii"))
                if origin.path or origin.query or origin.fragment:
                    raise ValueError("Origin is not a bare origin")
                if authority(origin.netloc, origin.scheme) != target:
                    raise ValueError("Origin differs from request origin")
            except (ValueError, UnicodeError):
                return 403, "Origin must match the local request origin"
        # No Origin is permitted for direct local CLI clients. This is not auth.
    return None


class LocalAccessMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            denied = access_denial(scope)
            if denied:
                status, detail = denied
                return await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)
        return await self.app(scope, receive, send)
