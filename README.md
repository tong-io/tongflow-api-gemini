# tongflow-api-gemini

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text and image generation via [Google Gemini](https://ai.google.dev).

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate / rewrite text** (`gen-text`) — create or edit copy from a prompt.
- **Combine text** (`combine-text`) — merge multiple text nodes into one.
- **Split long text** (`split-text`) — break a long passage into chunks.
- **Generate image** (`image-gen`) — text → image via Nano Banana.
- **Edit image** (`image-edit`) — image + text → image.
- **Fuse images** (`image-fusion`) — combine multiple reference images into one.
- **Arrange & batch groups** (`arrange-group`) — group and arrange text/clip batches for downstream processing.
- **Filter or drop clips** (`drop-video`) — drop unwanted clips by rule.

Image slots default to **Nano Banana Lite** (`gemini-3.1-flash-lite-image`) — Google's low-latency, 1K image model — via the same `generateContent` API and key as the text slots.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | ✅ | Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `GOOGLE_API_KEY` is accepted as a fallback. |
| `GEMINI_MODEL` | optional | Override the default text model (e.g. `gemini-2.0-flash`). |
| `GEMINI_IMAGE_MODEL` | optional | Override the default image model (e.g. `gemini-3.1-flash-image`, `gemini-3-pro-image`). |
| `GEMINI_IMAGE_SIZE` | optional | Output resolution (`1K` default; `2K`/`4K` require a Flash/Pro image model). |
| `GEMINI_IMAGE_ASPECT_RATIO` | optional | Force an aspect ratio (e.g. `16:9`). Defaults to the node's width/height, snapped to the nearest supported ratio. |

Values are stored locally and take effect without a restart.
