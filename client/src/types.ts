/**
 * Katherine Client - Type Definitions
 */

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  personal_heuristics?: string
}

export interface Memory {
  id: string
  content: string
  summary?: string
  emotional_tone?: string
  importance: number
  created_at: string
  tags: string[]
}

export interface MemorySearchResult {
  memory: Memory
  similarity: number
  relevance_explanation?: string
}

export interface ChatRequest {
  message: string
  conversation_id?: string
  include_memories?: boolean
}

export interface ChatResponse {
  response: string
  conversation_id: string
  retrieved_memories: MemorySearchResult[]
  tokens_used?: number
}

export interface HealthStatus {
  status: string
  embedding_model_loaded: boolean
  chroma_connected: boolean
  vllm_reachable: boolean
  memory_count: number
  conversation_count: number
  message_count: number
}

export interface Conversation {
  id: string
  title?: string
  created_at: string
  updated_at: string
  message_count?: number
}

export interface ConversationDetail extends Conversation {
  messages: Message[]
}

export interface StreamMemory {
  id: string
  content: string
  emotional_tone?: string
  similarity: number
}

export interface StreamEvent {
  type: 'meta' | 'content' | 'error' | 'done' | 'thinking'
  content?: string
  conversation_id?: string
  memories?: StreamMemory[]  // Full memory data from streaming
  error?: string
  personal_heuristics?: string  // Personal heuristics extracted from internal monologue
}

export interface ContextWindowStats {
  conversation_id: string
  window_hours: number
  cutoff_time: string
  active_messages: number
  expired_messages: number
  oldest_active_message?: string
  newest_expired_message?: string
}

export interface ExpiredMessagesResponse {
  count: number
  window_hours: number
  cutoff_time: string
  messages: Array<{
    message: Message
    conversation_id: string
    conversation_title?: string
  }>
}

export interface ArchivalStats {
  pending_messages: number
  window_hours: number
  cutoff_time: string
  is_running: boolean
  memory_count: number
}

export interface ArchivalResult {
  status: string
  messages_processed: number
  memories_created: number
  messages_archived: number
  errors: string[]
  memories: Memory[]
}

export interface FullArchivalResult {
  status: string
  batches_run: number
  total_messages_processed: number
  total_memories_created: number
  errors: string[]
}

export interface AutoArchivalStatus {
  enabled: boolean
  running: boolean
  interval_seconds: number
  threshold: number
  pending_messages: number
  will_trigger: boolean
}

export interface SelfDevelopmentAssessment {
  assessment: 'yes' | 'partial' | 'no' | 'unknown'
  reason: string
  timestamp: string
}

export interface SelfDevelopmentStatus {
  enabled: boolean
  window_size: number
  current_assessments: number
  negative_ratio: number
  threshold: number
  reflection_triggered: boolean
  last_reflection_time: string | null
  recent_assessments: SelfDevelopmentAssessment[]
}

export interface UserTag {
  tag: string
  display_order: number
  created_at: string
  updated_at: string
}
