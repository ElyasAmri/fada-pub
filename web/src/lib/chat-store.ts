"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export type ChatContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: string };

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  content: string | ChatContentPart[];
  createdAt: number;
};

export type Conversation = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
};

type ChatStore = {
  conversations: Conversation[];
  activeConversationId: string | null;
  createConversation: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  addMessage: (conversationId: string, message: ChatMessage) => void;
  renameConversation: (conversationId: string, title: string) => void;
};

export function createId(prefix: string) {
  return `${prefix}-${crypto.randomUUID()}`;
}

function persistedConversations(conversations: Conversation[]) {
  return conversations.filter((conversation) => conversation.messages.length > 0);
}

function firstTitleFrom(content: ChatMessage["content"]) {
  const text =
    typeof content === "string"
      ? content
      : content.find((part) => part.type === "text")?.text ?? "Image prompt";
  return text.trim().slice(0, 42) || "New chat";
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      conversations: [],
      activeConversationId: null,
      createConversation: () => {
        let id = "";
        set((state) => {
          const existingDraft = state.conversations.find((conversation) => conversation.messages.length === 0);
          if (existingDraft) {
            id = existingDraft.id;
            return {
              conversations: state.conversations,
              activeConversationId: existingDraft.id,
            };
          }

          const now = Date.now();
          id = createId("thread");
          const conversation: Conversation = {
            id,
            title: "New chat",
            messages: [],
            createdAt: now,
            updatedAt: now,
          };
          return {
            conversations: [conversation, ...state.conversations],
            activeConversationId: id,
          };
        });
        return id;
      },
      selectConversation: (id) =>
        set((state) => ({
          conversations: state.conversations.filter(
            (conversation) => conversation.messages.length > 0 || conversation.id === id,
          ),
          activeConversationId: id,
        })),
      deleteConversation: (id) =>
        set((state) => {
          const conversations = state.conversations.filter((item) => item.id !== id);
          return {
            conversations,
            activeConversationId:
              state.activeConversationId === id
                ? conversations[0]?.id ?? null
                : state.activeConversationId,
          };
        }),
      addMessage: (conversationId, message) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) => {
            if (conversation.id !== conversationId) {
              return conversation;
            }
            const shouldRetitle =
              conversation.title === "New chat" && message.role === "user";
            return {
              ...conversation,
              title: shouldRetitle ? firstTitleFrom(message.content) : conversation.title,
              messages: [...conversation.messages, message],
              updatedAt: Date.now(),
            };
          }),
        })),
      renameConversation: (conversationId, title) =>
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? { ...conversation, title, updatedAt: Date.now() }
              : conversation,
          ),
        })),
    }),
    {
      name: "fada-chat-store",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        conversations: persistedConversations(state.conversations),
        activeConversationId: persistedConversations(state.conversations).some(
          (conversation) => conversation.id === state.activeConversationId,
        )
          ? state.activeConversationId
          : null,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) {
          return;
        }
        state.conversations = persistedConversations(state.conversations);
        if (!state.activeConversationId && state.conversations[0]) {
          state.activeConversationId = state.conversations[0].id;
        }
      },
    },
  ),
);

export function toEndpointMessages(messages: ChatMessage[]) {
  return messages.map(({ role, content }) => ({ role, content }));
}
