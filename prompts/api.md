You are documenting a codebase's HTTP API surface. Below are the endpoint handlers detected in the code, each with its decorator (which usually encodes the HTTP method and path) and docstring where available.

Detected endpoints:
{endpoint_digest}

Rules:
- Produce a Markdown API reference: one section per endpoint, using whatever HTTP method and path the decorator implies.
- For each endpoint, briefly describe its purpose based on its name and docstring. If no docstring is available, keep the description short and derive it only from the function name and path — do not invent request/response schemas, parameters, or behavior that aren't shown.
- If multiple endpoints share an obvious resource (e.g. several `/items/...` routes), you may group them under one heading.
- No preamble before the first heading.

API Reference:
