/**
 * Katherine Client - State Store (Zustand)
 */

import { create } from 'zustand'
import type { Message, MemorySearchResult, HealthStatus } from './types'

interface ChatState {
  // Conversation
  conversationId: string | null
  messages: Message[]
  isLoading: boolean
  isStreaming: boolean
  isThinking: boolean  // AI finished public response, generating internal monologue
  streamingContent: string
  
  // Memories
  retrievedMemories: MemorySearchResult[]
  
  // System status
  health: HealthStatus | null
  error: string | null
  
  // Actions
  setConversationId: (id: string | null) => void
  addMessage: (message: Message) => void
  setMessages: (messages: Message[]) => void
  updateMessage: (messageId: string, content: string) => void
  deleteMessage: (messageId: string) => void
  setLoading: (loading: boolean) => void
  setStreaming: (streaming: boolean) => void
  setThinking: (thinking: boolean) => void
  appendStreamContent: (content: string) => void
  clearStreamContent: () => void
  setRetrievedMemories: (memories: MemorySearchResult[]) => void
  setHealth: (health: HealthStatus) => void
  setError: (error: string | null) => void
  clearChat: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  // Initial state
  conversationId: null,
  messages: [],
  isLoading: false,
  isStreaming: false,
  isThinking: false,
  streamingContent: '',
  retrievedMemories: [],
  health: null,
  error: null,

  // Actions
  setConversationId: (id) => set({ conversationId: id }),
  
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),
  
  setMessages: (messages) => set({ messages }),
  
  updateMessage: (messageId, content) => set((state) => ({
    messages: state.messages.map((m) =>
      m.id === messageId ? { ...m, content } : m
    )
  })),
  
  deleteMessage: (messageId) => set((state) => ({
    messages: state.messages.filter((m) => m.id !== messageId)
  })),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setStreaming: (streaming) => set({ isStreaming: streaming }),
  
  setThinking: (thinking) => set({ isThinking: thinking }),
  
  appendStreamContent: (content) => set((state) => ({
    streamingContent: state.streamingContent + content
  })),
  
  clearStreamContent: () => set({ streamingContent: '' }),
  
  setRetrievedMemories: (memories) => set({ retrievedMemories: memories }),
  
  setHealth: (health) => set({ health }),
  
  setError: (error) => set({ error }),
  
  clearChat: () => set({
    conversationId: null,
    messages: [],
    streamingContent: '',
    retrievedMemories: [],
    error: null,
    isThinking: false,
  }),
}))
