# FADA Web Chat

Next.js chat interface for the Hugging Face ZeroGPU FADA Space. The client now detects Hugging Face Spaces and talks to the queued Gradio API directly, while still keeping the older RunPod/OpenAI-style path available.

## Environment

Copy `.env.example` to `.env.local` and set:

```env
NEXT_PUBLIC_LLM_ENDPOINT=https://elyasamri-fada-gemma4-zerogpu.hf.space
NEXT_PUBLIC_LLM_MODEL=gemma3-4b
NEXT_PUBLIC_LLM_API_KEY=your-huggingface-token
```

The browser calls the configured endpoint directly. For the current private HF Space, that means the Hugging Face token is exposed to the client, so only use a token you are comfortable shipping to the browser or switch the Space to public / add a backend proxy first.

`NEXT_PUBLIC_LLM_ENDPOINT` can be either:

- the HF Space host, such as `https://elyasamri-fada-gemma4-zerogpu.hf.space`
- a Gradio API base URL under that Space
- or the older OpenAI-style endpoint URL

For HF Spaces, the client posts to `/gradio_api/call/analyze_image`, sends the selected Space model key, waits for the queued SSE completion event, and uses the most recent attached image plus recent chat history as request context.

## Run

```bash
npm run dev
```
