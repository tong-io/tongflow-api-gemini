# tongflow-api-gemini

Official TongFlow plugin. Text generation via [Google Gemini](https://ai.google.dev).

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate / rewrite text** (`gen-text`) — create or edit copy from a prompt.
- **Combine text** (`combine-text`) — merge multiple text nodes into one.
- **Split long text** (`split-text`) — break a long passage into chunks.
- **Arrange & batch groups** (`arrange-group`) — group and arrange text/clip batches for downstream processing.
- **Filter or drop clips** (`drop-video`) — drop unwanted clips by rule.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | ✅ | Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `GOOGLE_API_KEY` is accepted as a fallback. |
| `GEMINI_MODEL` | optional | Override the default model (e.g. `gemini-2.0-flash`). |

Values are stored locally and take effect without a restart.
