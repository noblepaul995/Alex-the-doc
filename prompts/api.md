You are documenting a codebase's HTTP API surface. Below are the endpoint handlers detected in the code, each with its route/decorator, signature, parameters, return type, and docstring where available.

Detected endpoints:
{endpoint_digest}

Rules:
- Produce a Markdown API reference: one section per endpoint, using whatever HTTP method and path the route/decorator implies.
- For each endpoint, document:
  - **Purpose**: derived from its docstring (or briefly from its name and path if no docstring is available).
  - **Parameters**: list the parameters and their type annotations exactly as shown in `Parameters` (omit if `(none)`).
  - **Returns**: state the return type from `Return type` when annotated (omit if `(not annotated)`).
- Do NOT invent request/response schemas, query parameters, status codes, or behavior that aren't shown in the digest above.
- If multiple endpoints share an obvious resource (e.g. several `/items/...` routes), you may group them under one heading.
- No preamble before the first heading.

API Reference:
