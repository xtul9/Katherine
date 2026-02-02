/**
 * Katherine Client - API Layer
 */

import type { ChatRequest, ChatResponse, HealthStatus, Memory, MemorySearchResult, StreamEvent, Conversation, ConversationDetail, ContextWindowStats, ExpiredMessagesResponse, ArchivalStats, ArchivalResult, FullArchivalResult, AutoArchivalStatus } from './types'

const API_BASE = '/api'

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.text()
    throw new ApiError(response.status, error)
  }

  return response.json()
}

export const api = {
  /**
   * Check the health of the orchestrator
   */
  async health(): Promise<HealthStatus> {
    return request<HealthStatus>('/health')
  },

  /**
   * Send a chat message and get a response
   */
  async chat(req: ChatRequest): Promise<ChatResponse> {
    return request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify(req),
    })
  },

  /**
   * Stream a chat response using SSE
   */
  async *chatStream(req: ChatRequest): AsyncGenerator<StreamEvent> {
    const url = `${API_BASE}/chat/stream`
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req),
    })

    if (!response.ok) {
      const error = await response.text()
      throw new ApiError(response.status, error)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          try {
            const event = JSON.parse(data) as StreamEvent
            yield event
          } catch {
            // Ignore parse errors
          }
        }
      }
    }
  },

  /**
   * Save a memory manually
   */
  async saveMemory(memory: {
    content: string
    summary?: string
    emotional_tone?: string
    importance?: number
    tags?: string[]
  }): Promise<Memory> {
    return request<Memory>('/memories', {
      method: 'POST',
      body: JSON.stringify(memory),
    })
  },

  /**
   * Search memories
   */
  async searchMemories(
    query: string,
    topK: number = 5,
    minSimilarity: number = 0.3
  ): Promise<{ results: MemorySearchResult[] }> {
    return request('/memories/search', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: topK,
        min_similarity: minSimilarity,
      }),
    })
  },

  /**
   * Get all memories
   */
  async listMemories(limit: number = 100): Promise<Memory[]> {
    return request<Memory[]>(`/memories?limit=${limit}`)
  },

  /**
   * Get a single memory
   */
  async getMemory(memoryId: string): Promise<Memory> {
    return request<Memory>(`/memories/${memoryId}`)
  },

  /**
   * Update a memory
   */
  async updateMemory(
    memoryId: string,
    updates: {
      content?: string
      summary?: string
      emotional_tone?: string
      importance?: number
      tags?: string[]
    }
  ): Promise<Memory> {
    return request<Memory>(`/memories/${memoryId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    })
  },

  /**
   * Delete a memory
   */
  async deleteMemory(memoryId: string): Promise<void> {
    await request(`/memories/${memoryId}`, { method: 'DELETE' })
  },

  /**
   * Extract memories from conversation
   */
  async extractMemories(conversationId: string): Promise<{
    status: string
    memories_extracted?: number
    memories?: Memory[]
  }> {
    return request(`/conversations/${conversationId}/extract-memories`, {
      method: 'POST',
    })
  },

  // =========================================================================
  // Conversations
  // =========================================================================

  /**
   * List all conversations
   */
  async listConversations(limit: number = 50, offset: number = 0): Promise<{
    conversations: Conversation[]
    total: number
  }> {
    return request(`/conversations?limit=${limit}&offset=${offset}`)
  },

  /**
   * Get a single conversation with messages
   */
  async getConversation(conversationId: string): Promise<ConversationDetail> {
    return request(`/conversations/${conversationId}`)
  },

  /**
   * Update conversation (e.g., title)
   */
  async updateConversation(conversationId: string, title: string): Promise<void> {
    await request(`/conversations/${conversationId}?title=${encodeURIComponent(title)}`, {
      method: 'PUT',
    })
  },

  /**
   * Delete a conversation
   */
  async deleteConversation(conversationId: string): Promise<void> {
    await request(`/conversations/${conversationId}`, { method: 'DELETE' })
  },

  // =========================================================================
  // Messages
  // =========================================================================

  /**
   * Update a message's content
   */
  async updateMessage(messageId: string, content: string): Promise<{ status: string; message: import('./types').Message }> {
    return request(`/messages/${messageId}?content=${encodeURIComponent(content)}`, {
      method: 'PUT',
    })
  },

  /**
   * Delete a message
   */
  async deleteMessage(messageId: string): Promise<void> {
    await request(`/messages/${messageId}`, { method: 'DELETE' })
  },

  // =========================================================================
  // Context Window (Time-based)
  // =========================================================================

  /**
   * Get context window statistics for a conversation
   */
  async getContextWindowStats(conversationId: string): Promise<ContextWindowStats> {
    return request(`/conversations/${conversationId}/context-window`)
  },

  /**
   * Get all expired messages (outside context window)
   */
  async getExpiredMessages(limit: number = 100): Promise<ExpiredMessagesResponse> {
    return request(`/conversations/expired-messages?limit=${limit}`)
  },

  // =========================================================================
  // Archival
  // =========================================================================

  /**
   * Get archival statistics
   */
  async getArchivalStats(): Promise<ArchivalStats> {
    return request('/archival/stats')
  },

  /**
   * Run archival process
   */
  async runArchival(
    mode: 'simple' | 'llm' = 'simple',
    batchSize: number = 50,
    conversationId?: string
  ): Promise<ArchivalResult> {
    const params = new URLSearchParams({
      mode,
      batch_size: batchSize.toString(),
    })
    if (conversationId) {
      params.append('conversation_id', conversationId)
    }
    return request(`/archival/run?${params}`, { method: 'POST' })
  },

  /**
   * Run full archival (all expired messages)
   */
  async runFullArchival(mode: 'simple' | 'llm' = 'simple'): Promise<FullArchivalResult> {
    return request(`/archival/run-all?mode=${mode}`, { method: 'POST' })
  },

  // =========================================================================
  // Auto-Archival
  // =========================================================================

  /**
   * Get auto-archival status
   */
  async getAutoArchivalStatus(): Promise<AutoArchivalStatus> {
    return request('/archival/auto/status')
  },

  /**
   * Enable auto-archival
   */
  async enableAutoArchival(): Promise<AutoArchivalStatus & { status: string }> {
    return request('/archival/auto/enable', { method: 'POST' })
  },

  /**
   * Disable auto-archival
   */
  async disableAutoArchival(): Promise<AutoArchivalStatus & { status: string }> {
    return request('/archival/auto/disable', { method: 'POST' })
  },
}

export { ApiError }
