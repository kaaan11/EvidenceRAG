"""Streaming HTTP and multipart budgets for the two input endpoints.

The pinned Starlette parser closes its partial spools on BaseException. Keep
body-limit failures visible here even when a downstream layer converts them to
400/500 responses. No complete request body is buffered by this middleware.
"""
from contextlib import asynccontextmanager

from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.responses import JSONResponse


class RequestLimitExceeded(MultiPartException):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in {"/documents", "/ask"}:
            return await self.app(scope, receive, send)
        limits = self.settings()
        limit = limits.max_http_body_bytes if scope["path"] == "/documents" else limits.max_json_body_bytes
        response = JSONResponse({"detail": f"HTTP body exceeds {limit} bytes"}, status_code=413)
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    size = int(value)
                    if size < 0:
                        raise ValueError()
                except ValueError:
                    return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
                if size > limit:
                    return await response(scope, receive, send)
        received = 0
        exceeded = False
        replaced = False

        async def bounded_receive():
            nonlocal received, exceeded
            if exceeded:
                raise RequestLimitExceeded("HTTP body limit exceeded")
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    # Never deliver the over-budget chunk to the parser.
                    raise RequestLimitExceeded("HTTP body limit exceeded")
            return message

        async def bounded_send(message):
            nonlocal replaced
            if exceeded:
                if not replaced:
                    replaced = True
                    await response(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, bounded_receive, bounded_send)
        except Exception:
            if not exceeded:
                raise
            if not replaced:
                await response(scope, receive, send)


class LimitedMultipartParser(MultiPartParser):
    def __init__(self, headers, stream, limits):
        super().__init__(headers, stream, max_files=1, max_fields=0)
        self.limits = limits
        self.header_bytes = 0
        self.file_bytes = 0
        self.complete = False

    def on_end(self):
        self.complete = True
        super().on_end()

    async def parse(self):
        try:
            form = await super().parse()
            # python-multipart finalize can accept a truncated last part without
            # raising; those unfinished spools are absent from FormData.close().
            if not self.complete:
                raise MultiPartException("Incomplete multipart body")
            return form
        except BaseException:
            for file in self._files_to_close_on_error:
                file.close()
            raise

    def on_part_begin(self):
        self.header_bytes = 0
        self.file_bytes = 0
        super().on_part_begin()

    def _header_budget(self, count):
        self.header_bytes += count
        if self.header_bytes > self.limits.max_multipart_header_bytes:
            raise RequestLimitExceeded("Multipart part headers exceed byte limit")

    def on_header_field(self, data, start, end):
        self._header_budget(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data, start, end):
        self._header_budget(end - start)
        super().on_header_value(data, start, end)

    def on_header_end(self):
        self._header_budget(4)  # colon/space and terminating CRLF
        super().on_header_end()

    def on_headers_finished(self):
        super().on_headers_finished()
        part = self._current_part
        if part.file is None or part.field_name != "file":
            raise MultiPartException("Expected one file part named 'file'; other fields are not accepted")
        if len(part.file.filename.encode("utf-8")) > self.limits.max_filename_bytes:
            raise RequestLimitExceeded("Filename exceeds UTF-8 byte limit")

    def on_part_data(self, data, start, end):
        self.file_bytes += end - start
        if self.file_bytes > self.limits.max_upload_bytes:
            raise RequestLimitExceeded(f"File exceeds {self.limits.max_upload_bytes} bytes")
        super().on_part_data(data, start, end)


@asynccontextmanager
async def bounded_upload(request, limits):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(status_code=400, detail="Expected multipart/form-data")
    try:
        form = await LimitedMultipartParser(request.headers, request.stream(), limits).parse()
    except RequestLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=exc.message) from exc
    except MultiPartException as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    try:
        if "file" not in form:
            raise HTTPException(status_code=422, detail="File part is required")
        yield form["file"]
    finally:
        await form.close()
