import { ChatApp } from "@/components/chat-app";

export default function Home() {
  const endpoint =
    process.env.NEXT_PUBLIC_LLM_ENDPOINT ??
    process.env.LLM_ENDPOINT ??
    "https://elyasamri-fada-gemma4-zerogpu.hf.space";
  const model =
    process.env.NEXT_PUBLIC_LLM_MODEL ??
    process.env.LLM_MODEL ??
    "gemma3-4b";
  const apiKey =
    process.env.NEXT_PUBLIC_LLM_API_KEY ??
    process.env.LLM_API_KEY ??
    process.env.HF_TOKEN ??
    null;

  return <ChatApp endpoint={endpoint} model={model} apiKey={apiKey} />;
}
