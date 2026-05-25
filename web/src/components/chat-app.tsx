"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ImagePlus,
  Loader2,
  MessageCircle,
  PanelLeft,
  Plus,
  Search,
  Send,
  Trash2,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  ChatContentPart,
  ChatMessage,
  createId,
  toEndpointMessages,
  useChatStore,
} from "@/lib/chat-store";
import { cn } from "@/lib/utils";

function contentText(content: ChatMessage["content"]) {
  if (typeof content === "string") {
    return content;
  }
  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join(" ");
}

function imageParts(content: ChatMessage["content"]) {
  if (typeof content === "string") {
    return [];
  }
  return content.filter(
    (part): part is { type: "image_url"; image_url: string } =>
      part.type === "image_url",
  );
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div
      className={cn(
        "break-words text-sm leading-6",
        "[&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_p]:my-3",
        "[&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-5",
        "[&_ol]:my-3 [&_ol]:list-decimal [&_ol]:pl-5",
        "[&_li]:my-1",
        "[&_a]:underline [&_a]:underline-offset-4",
        "[&_blockquote]:my-3 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-4 [&_blockquote]:italic",
        "[&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1",
        "[&_pre]:my-3 [&_pre]:overflow-x-auto [&_pre]:rounded-2xl [&_pre]:border [&_pre]:border-border/60 [&_pre]:bg-background/70 [&_pre]:p-3",
        "[&_code]:rounded [&_code]:bg-background/60 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.925em]",
        "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function formatRelative(timestamp: number) {
  const diff = Date.now() - timestamp;
  if (diff < 60_000) return "now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h`;
  return `${Math.floor(diff / 86_400_000)}d`;
}

async function fileToDataUri(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

const RUNPOD_POLL_INTERVAL_MS = 5000;
const RUNPOD_POLL_TIMEOUT_MS = 8 * 60_000;
const RUNPOD_JOB_TTL_MS = 300_000;
const GRADIO_HISTORY_WINDOW = 8;
const DEFAULT_MAX_TOKENS = 512;
const DEFAULT_TEMPERATURE = 0.2;
const DEFAULT_HF_MODEL_KEY = "gemma3-4b";
const MODEL_STORAGE_KEY = "fada-selected-model";

const HF_MODEL_OPTIONS = [
  { key: "gemma3-4b", label: "Gemma 3 4B FADA" },
  { key: "qwen2-vl-7b", label: "Qwen2-VL 7B FADA" },
  { key: "qwen3-vl-4b", label: "Qwen3-VL 4B FADA" },
  { key: "qwen35-9b", label: "Qwen3.5 9B FADA" },
  { key: "qwen25-vl-3b", label: "Qwen2.5-VL 3B FADA" },
  { key: "internvl35-2b", label: "InternVL3.5 2B FADA" },
  { key: "gemma4-e2b", label: "Gemma 4 E2B FADA" },
];

function normalizeHfModelKey(value: string | null | undefined): string {
  return HF_MODEL_OPTIONS.find((option) => option.key === value)?.key ?? DEFAULT_HF_MODEL_KEY;
}

type OpenAiChatResponse = {
  choices?: Array<{
    message?: {
      role?: string;
      content?: string;
    };
  }>;
  error?: string;
  detail?: string;
};

type RunpodJobResponse = {
  id?: string;
  status?: string;
  output?: OpenAiChatResponse;
  error?: string | { message?: string };
  detail?: string;
};

type GradioQueueResponse = {
  event_id?: string;
  error?: string;
  detail?: string;
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getErrorMessage(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  const record = payload as Record<string, unknown>;
  if (typeof record.detail === "string") {
    return record.detail;
  }
  if (typeof record.error === "string") {
    return record.error;
  }
  if (record.error && typeof record.error === "object") {
    const nested = record.error as Record<string, unknown>;
    if (typeof nested.message === "string") {
      return nested.message;
    }
  }
  return fallback;
}

function isGradioSpaceEndpoint(endpoint: string) {
  return endpoint.includes(".hf.space") || endpoint.includes("/gradio_api");
}

function getGradioBaseUrl(endpoint: string) {
  const trimmed = endpoint.replace(/\/$/, "");
  const gradioIndex = trimmed.indexOf("/gradio_api");
  return gradioIndex === -1 ? trimmed : trimmed.slice(0, gradioIndex);
}

function getLatestImageUrl(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const imageUrl = imageParts(messages[index].content)[0]?.image_url;
    if (imageUrl) {
      return imageUrl;
    }
  }
  return null;
}

function describeForGradio(content: ChatMessage["content"]) {
  const text = contentText(content).trim();
  const hasImage = imageParts(content).length > 0;

  if (text && hasImage) {
    return `${text}\n[Ultrasound image attached]`;
  }
  if (text) {
    return text;
  }
  if (hasImage) {
    return "Describe this ultrasound image.";
  }
  return "";
}

function buildGradioQuestion(messages: ChatMessage[]) {
  const recentMessages = messages.slice(-GRADIO_HISTORY_WINDOW);

  if (recentMessages.length === 0) {
    return "Describe the visible anatomy and key observations in this ultrasound image.";
  }

  if (recentMessages.length === 1) {
    return describeForGradio(recentMessages[0].content);
  }

  const transcript = recentMessages
    .map((message) => {
      const speaker =
        message.role === "assistant"
          ? "Assistant"
          : message.role === "system"
            ? "System"
            : "User";
      return `${speaker}: ${describeForGradio(message.content) || "(no text)"}`;
    })
    .join("\n\n");

  return [
    "Continue this conversation about a fetal ultrasound image.",
    "Use the latest available ultrasound image from the conversation when it is relevant.",
    "",
    transcript,
    "",
    "Answer the latest user request directly.",
  ].join("\n");
}

function toGradioImageInput(imageUrl: string | null) {
  if (!imageUrl) {
    return null;
  }

  const mimeMatch = /^data:([^;]+);base64,/.exec(imageUrl);
  const mimeType = mimeMatch?.[1] ?? null;
  const extension = mimeType?.split("/")[1]?.replace(/[^a-zA-Z0-9]/g, "") || "png";

  return {
    url: imageUrl,
    orig_name: `upload.${extension}`,
    mime_type: mimeType,
    is_stream: false,
    meta: {
      _type: "gradio.FileData",
    },
  };
}

function parseSseBlocks(body: string) {
  return body
    .trim()
    .split(/\r?\n\r?\n/)
    .map((block) => {
      const lines = block.split(/\r?\n/);
      const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() ?? "";
      const data = lines
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      return { event, data };
    });
}

function parseGradioSseResponse(body: string) {
  const events = parseSseBlocks(body);
  const complete = [...events].reverse().find((event) => event.event === "complete");

  if (!complete?.data) {
    const errorEvent = [...events]
      .reverse()
      .find((event) => event.event === "error" || event.event === "failed");
    if (errorEvent?.data) {
      throw new Error(errorEvent.data);
    }
    throw new Error("HF Space returned an incomplete response");
  }

  const parsed = JSON.parse(complete.data) as unknown;
  const output =
    Array.isArray(parsed) && typeof parsed[0] === "string"
      ? parsed[0]
      : typeof parsed === "string"
        ? parsed
        : null;

  if (!output) {
    throw new Error("HF Space returned an empty response");
  }

  return output.replace(/<turn\|>/g, "").trim();
}

function getChatRequest(endpoint: string, model: string, messages: ChatMessage[]) {
  const trimmed = endpoint.replace(/\/$/, "");
  if (isGradioSpaceEndpoint(trimmed)) {
    const baseUrl = getGradioBaseUrl(trimmed);
    return {
      kind: "gradio" as const,
      url: `${baseUrl}/gradio_api/call/analyze_image`,
      body: {
        data: [
          model,
          toGradioImageInput(getLatestImageUrl(messages)),
          buildGradioQuestion(messages),
          DEFAULT_MAX_TOKENS,
          DEFAULT_TEMPERATURE,
        ],
      },
    };
  }

  const endpointMessages = toEndpointMessages(messages);
  if (trimmed.includes("api.runpod.ai/v2/")) {
    const runpodBase = trimmed.replace(/\/openai\/v1$/, "").replace(/\/chat\/completions$/, "");
    return {
      kind: "runpod" as const,
      url: `${runpodBase}/run`,
      statusUrl: `${runpodBase}/status`,
      body: {
        input: {
          model,
          messages: endpointMessages,
          max_tokens: DEFAULT_MAX_TOKENS,
          temperature: DEFAULT_TEMPERATURE,
        },
        policy: {
          ttl: RUNPOD_JOB_TTL_MS,
        },
      },
    };
  }

  const url = trimmed.endsWith("/chat/completions")
    ? trimmed
    : trimmed.endsWith("/openai/v1")
      ? `${trimmed}/chat/completions`
      : `${trimmed}/v1/chat/completions`;

  return {
    kind: "openai" as const,
    url,
    body: {
      model,
      messages: endpointMessages,
      max_tokens: DEFAULT_MAX_TOKENS,
      temperature: DEFAULT_TEMPERATURE,
    },
  };
}

async function resolveRunpodResponse(
  initialJob: RunpodJobResponse,
  statusUrl: string,
  apiKey: string,
) {
  let job = initialJob;
  const deadline = Date.now() + RUNPOD_POLL_TIMEOUT_MS;

  while (Date.now() < deadline) {
    if (job.status === "COMPLETED") {
      if (job.output) {
        return job.output;
      }
      throw new Error("RunPod completed without an output payload");
    }

    if (job.status === "FAILED" || job.status === "CANCELLED" || job.status === "TIMED_OUT") {
      throw new Error(getErrorMessage(job, `RunPod job ${job.status.toLowerCase()}`));
    }

    if (!job.id) {
      throw new Error("RunPod did not return a job ID");
    }

    await sleep(RUNPOD_POLL_INTERVAL_MS);

    const response = await fetch(`${statusUrl}/${job.id}`, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    });
    const data = (await response.json()) as RunpodJobResponse;
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "Unable to read RunPod job status"));
    }
    job = data;
  }

  throw new Error("Model is still warming up. Please retry in a moment.");
}

async function resolveGradioResponse(
  initialEvent: GradioQueueResponse,
  url: string,
  apiKey: string | null,
) {
  if (!initialEvent.event_id) {
    throw new Error(getErrorMessage(initialEvent, "HF Space did not return an event ID"));
  }

  const headers: HeadersInit = {};
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const response = await fetch(`${url}/${initialEvent.event_id}`, { headers });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(body || "Unable to read HF Space response");
  }

  return parseGradioSseResponse(body);
}

type ChatAppProps = {
  endpoint: string | null;
  model: string;
  apiKey: string | null;
};

export function ChatApp({ endpoint, model, apiKey }: ChatAppProps) {
  const {
    conversations,
    activeConversationId,
    createConversation,
    selectConversation,
    deleteConversation,
    addMessage,
  } = useChatStore();
  const [prompt, setPrompt] = useState("");
  const [query, setQuery] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState(() => normalizeHfModelKey(model));
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const chatEndpoint = useMemo(() => endpoint?.replace(/\/$/, "") ?? null, [endpoint]);
  const isGradioEndpoint = useMemo(
    () => Boolean(chatEndpoint && isGradioSpaceEndpoint(chatEndpoint)),
    [chatEndpoint],
  );
  const activeModel = isGradioEndpoint ? selectedModel : model;
  const activeModelLabel =
    HF_MODEL_OPTIONS.find((option) => option.key === selectedModel)?.label ?? activeModel;

  useEffect(() => {
    const savedModel = window.localStorage.getItem(MODEL_STORAGE_KEY);
    if (savedModel) {
      setSelectedModel(normalizeHfModelKey(savedModel));
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(MODEL_STORAGE_KEY, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    if (!activeConversationId && conversations.length === 0) {
      createConversation();
    }
  }, [activeConversationId, conversations.length, createConversation]);

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId) ?? null,
    [activeConversationId, conversations],
  );

  const filteredConversations = useMemo(() => {
    const savedConversations = conversations.filter((conversation) => conversation.messages.length > 0);
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return savedConversations;
    }
    return savedConversations.filter((conversation) =>
      conversation.title.toLowerCase().includes(normalized),
    );
  }, [conversations, query]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [activeConversation?.messages.length]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (isSending) {
      return;
    }

    const trimmed = prompt.trim();
    if (!trimmed && !image) {
      return;
    }

    const conversationId = activeConversation?.id ?? createConversation();
    const content: string | ChatContentPart[] = image
      ? [
          ...(trimmed ? [{ type: "text" as const, text: trimmed }] : []),
          { type: "image_url" as const, image_url: image },
        ]
      : trimmed;

    const userMessage: ChatMessage = {
      id: createId("message"),
      role: "user",
      content,
      createdAt: Date.now(),
    };

    addMessage(conversationId, userMessage);
    setPrompt("");
    setImage(null);
    setError(null);
    setIsSending(true);

    if (!chatEndpoint) {
      setError("LLM endpoint is not configured");
      setIsSending(false);
      return;
    }

    const latestMessages = [
      ...(activeConversation?.messages ?? []),
      userMessage,
    ];

    try {
      const endpointRequest = getChatRequest(
        chatEndpoint,
        activeModel,
        latestMessages,
      );
      const headers: HeadersInit = {
        "Content-Type": "application/json",
      };
      if (apiKey) {
        headers.Authorization = `Bearer ${apiKey}`;
      }

      const response = await fetch(endpointRequest.url, {
        method: "POST",
        headers,
        body: JSON.stringify(endpointRequest.body),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(getErrorMessage(data, "Chat request failed"));
      }

      let assistantContent: string | null = null;

      if (endpointRequest.kind === "gradio") {
        assistantContent = await resolveGradioResponse(
          data as GradioQueueResponse,
          endpointRequest.url,
          apiKey,
        );
      } else {
        const chatResponse =
          endpointRequest.kind === "runpod"
            ? await resolveRunpodResponse(
                data as RunpodJobResponse,
                endpointRequest.statusUrl,
                apiKey ?? "",
              )
            : (data as OpenAiChatResponse);
        assistantContent = chatResponse?.choices?.[0]?.message?.content ?? null;
      }

      if (!assistantContent) {
        throw new Error("LLM endpoint returned an empty response");
      }

      addMessage(conversationId, {
        id: createId("message"),
        role: "assistant",
        content: assistantContent,
        createdAt: Date.now(),
      });
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to reach the LLM endpoint",
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="flex h-dvh overflow-hidden bg-background text-foreground">
      <aside
        className={cn(
          "flex w-80 shrink-0 flex-col border-r border-border bg-card transition-all duration-300",
          !isSidebarOpen && "-ml-80",
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-border px-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-muted-foreground">
              FADA
            </p>
            <h1 className="text-lg font-semibold">Conversations</h1>
          </div>
          <Button size="icon" onClick={createConversation} aria-label="New chat">
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <div className="border-b border-border p-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search chats"
              className="pl-9"
            />
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {filteredConversations.map((conversation) => (
            <button
              key={conversation.id}
              type="button"
              onClick={() => selectConversation(conversation.id)}
              className={cn(
                "group mb-1 flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition-colors hover:bg-muted",
                conversation.id === activeConversationId && "bg-muted",
              )}
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <MessageCircle className="h-4 w-4" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">
                  {conversation.title}
                </span>
                <span className="text-xs text-muted-foreground">
                  {conversation.messages.length} messages ·{" "}
                  {formatRelative(conversation.updatedAt)}
                </span>
              </span>
              <span
                role="button"
                tabIndex={0}
                onClick={(event) => {
                  event.stopPropagation();
                  deleteConversation(conversation.id);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.stopPropagation();
                    deleteConversation(conversation.id);
                  }
                }}
                className="rounded-lg p-2 text-muted-foreground opacity-0 transition-opacity hover:bg-background hover:text-destructive group-hover:opacity-100"
                aria-label={`Delete ${conversation.title}`}
              >
                <Trash2 className="h-4 w-4" />
              </span>
            </button>
          ))}
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 items-center justify-between border-b border-border bg-card px-4">
          <div className="flex min-w-0 items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen((value) => !value)}
              aria-label="Toggle conversations"
            >
              <PanelLeft className="h-5 w-5" />
            </Button>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">
                {activeConversation?.title ?? "New chat"}
              </p>
              <p className="text-xs text-muted-foreground">
                Model: {isGradioEndpoint ? activeModelLabel : model}
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {isGradioEndpoint && (
              <label className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                Model
                <select
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                  disabled={isSending}
                  className="h-10 max-w-[220px] rounded-2xl border border-border bg-background px-3 text-sm text-foreground outline-none transition-colors focus:border-primary"
                >
                  {HF_MODEL_OPTIONS.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <Button variant="secondary" onClick={createConversation}>
              <Plus className="h-4 w-4" />
              New chat
            </Button>
          </div>
        </header>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {!activeConversation?.messages.length ? (
              <div className="rounded-3xl border border-border bg-card p-8 shadow-sm">
                <div className="inline-flex rounded-full border border-border bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
                  {isGradioEndpoint ? activeModelLabel : model}
                </div>
                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  Start a conversation.
                </h2>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">
                  Upload an ultrasound image, ask a follow-up question, or continue
                  from one of your saved chats.
                </p>
                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  {[
                    "Describe this ultrasound image.",
                    "Compare these two views.",
                    "Summarize the visible anatomy.",
                    "What quality issues are present?",
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => setPrompt(suggestion)}
                      className="rounded-2xl border border-border bg-background px-4 py-3 text-left text-sm text-muted-foreground transition-colors hover:border-primary hover:text-foreground"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              activeConversation.messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex",
                    message.role === "user" ? "justify-end" : "justify-start",
                  )}
                >
                  <div
                    className={cn(
                      "max-w-[82%] rounded-3xl px-4 py-3 text-sm leading-6 shadow-sm",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "border border-border bg-card text-card-foreground",
                    )}
                  >
                    {imageParts(message.content).map((part) => (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        key={part.image_url}
                        src={part.image_url}
                        alt="Uploaded prompt"
                        className="mb-3 max-h-72 rounded-2xl object-contain"
                      />
                    ))}
                    {message.role === "assistant" && typeof message.content === "string" ? (
                      <MarkdownMessage content={message.content} />
                    ) : (
                      <p className="whitespace-pre-wrap">{contentText(message.content)}</p>
                    )}
                  </div>
                </div>
              ))
            )}
            {isSending && (
              <div className="flex justify-start">
                <div className="flex items-center gap-2 rounded-3xl border border-border bg-card px-4 py-3 text-sm text-muted-foreground shadow-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Thinking...
                </div>
              </div>
            )}
          </div>
        </div>

        <form
          onSubmit={onSubmit}
          className="border-t border-border bg-card/95 p-4 backdrop-blur"
        >
          <div className="mx-auto max-w-3xl">
            {error && (
              <div className="mb-3 rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            )}
            {image && (
              <div className="mb-3 flex items-center gap-3 rounded-2xl border border-border bg-background p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={image}
                  alt="Selected upload"
                  className="h-16 w-16 rounded-xl object-cover"
                />
                <span className="flex-1 text-sm text-muted-foreground">
                  Image attached
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => setImage(null)}
                  aria-label="Remove image"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            )}
            <div className="flex items-end gap-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={async (event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    setImage(await fileToDataUri(file));
                  }
                  event.target.value = "";
                }}
              />
              <Button
                type="button"
                variant="secondary"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                aria-label="Attach image"
              >
                <ImagePlus className="h-5 w-5" />
              </Button>
              <Textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="Message FADA..."
                className="min-h-14 flex-1"
              />
              <Button
                type="submit"
                size="icon"
                disabled={isSending || (!prompt.trim() && !image)}
                aria-label="Send message"
              >
                {isSending ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </Button>
            </div>
          </div>
        </form>
      </section>
    </main>
  );
}
