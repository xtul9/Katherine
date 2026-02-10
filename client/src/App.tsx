/**
 * Katherine Client - Main Application
 * The beautiful, polished bridge. The soul's window to the world.
 */

import { useEffect, useCallback, useRef, useState, memo, useMemo } from 'react'
import { useChatStore } from './store'
import { api } from './api'
import type { Message, UserTag } from './types'
import { clsx } from 'clsx'
import { formatDistanceToNow } from 'date-fns'
import MemoryAdmin from './MemoryAdmin'
import ConversationHistory from './ConversationHistory'
import MarkdownRenderer from './MarkdownRenderer'

// ============================================================================
// Message Component
// ============================================================================

interface MessageBubbleProps {
  message: Message
  isStreaming?: boolean
  onEdit?: (messageId: string, content: string) => Promise<void>
  onDelete?: (messageId: string) => Promise<void>
}

const MessageBubble = memo(function MessageBubble({ message, isStreaming, onEdit, onDelete }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState(message.content)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  // Memoize timestamp to avoid recalculation on every render
  const formattedTime = useMemo(
    () => formatDistanceToNow(new Date(message.timestamp), { addSuffix: true }),
    [message.timestamp]
  )

  // Auto-resize and focus textarea when editing starts
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus()
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [isEditing])

  const handleEditStart = () => {
    setEditContent(message.content)
    setIsEditing(true)
  }

  const handleEditCancel = () => {
    setIsEditing(false)
    setEditContent(message.content)
  }

  const handleEditSave = async () => {
    if (!onEdit || editContent.trim() === message.content) {
      setIsEditing(false)
      return
    }
    
    setIsSaving(true)
    try {
      await onEdit(message.id, editContent.trim())
      setIsEditing(false)
    } finally {
      setIsSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!onDelete) return
    if (!confirm('Czy na pewno chcesz usunąć tę wiadomość?')) return
    
    setIsDeleting(true)
    try {
      await onDelete(message.id)
    } finally {
      setIsDeleting(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleEditCancel()
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleEditSave()
    }
  }
  
  return (
    <div
      className={clsx(
        'animate-slide-up flex group',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      <div
        className={clsx(
          'max-w-[80%] rounded-2xl px-5 py-3 shadow-lg relative',
          isUser
            ? 'bg-gradient-to-br from-ember-600 to-ember-700 text-white'
            : 'bg-void-800/80 backdrop-blur-sm border border-void-700/50'
        )}
      >
        {/* Edit/Delete buttons */}
        {!isStreaming && !isEditing && onEdit && onDelete && (
          <div className={clsx(
            'absolute top-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity',
            isUser ? 'left-2' : 'right-2'
          )}>
            <button
              onClick={handleEditStart}
              className={clsx(
                'p-1.5 rounded-lg transition-colors',
                isUser 
                  ? 'hover:bg-white/20 text-white/70 hover:text-white'
                  : 'hover:bg-void-700 text-void-400 hover:text-void-200'
              )}
              title="Edytuj"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </button>
            <button
              onClick={handleDelete}
              disabled={isDeleting}
              className={clsx(
                'p-1.5 rounded-lg transition-colors',
                isUser 
                  ? 'hover:bg-red-500/30 text-white/70 hover:text-red-200'
                  : 'hover:bg-red-950/50 text-void-400 hover:text-red-400',
                isDeleting && 'opacity-50 cursor-not-allowed'
              )}
              title="Usuń"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        )}

        {isEditing ? (
          <div className="space-y-2">
            <textarea
              ref={textareaRef}
              value={editContent}
              onChange={(e) => {
                setEditContent(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = `${e.target.scrollHeight}px`
              }}
              onKeyDown={handleKeyDown}
              className={clsx(
                'w-full bg-transparent resize-none outline-none text-[15px] leading-relaxed min-h-[60px]',
                isUser ? 'text-white placeholder-white/50' : 'text-void-100 placeholder-void-500'
              )}
              placeholder="Treść wiadomości..."
            />
            <div className="flex items-center justify-end gap-2 pt-1 border-t border-white/10">
              <span className={clsx('text-xs mr-auto', isUser ? 'text-white/50' : 'text-void-500')}>
                Ctrl+Enter aby zapisać, Esc aby anulować
              </span>
              <button
                onClick={handleEditCancel}
                className={clsx(
                  'px-3 py-1 text-xs rounded-lg transition-colors',
                  isUser 
                    ? 'hover:bg-white/20 text-white/70' 
                    : 'hover:bg-void-700 text-void-400'
                )}
              >
                Anuluj
              </button>
              <button
                onClick={handleEditSave}
                disabled={isSaving || editContent.trim() === message.content}
                className={clsx(
                  'px-3 py-1 text-xs rounded-lg transition-colors',
                  isUser 
                    ? 'bg-white/20 hover:bg-white/30 text-white disabled:opacity-50' 
                    : 'bg-ember-600 hover:bg-ember-500 text-white disabled:opacity-50'
                )}
              >
                {isSaving ? 'Zapisuję...' : 'Zapisz'}
              </button>
            </div>
          </div>
        ) : (
          <>
            {isUser ? (
              <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            ) : (
              <div className="relative">
                <MarkdownRenderer content={message.content} />
                {isStreaming && (
                  <span className="inline-block w-2 h-4 ml-1 bg-ember-400 animate-typing rounded-sm align-middle" />
                )}
              </div>
            )}
            {!isUser && message.personal_heuristics && (
              <div className="mt-3 pt-3 border-t border-void-700/30">
                <div className="text-xs text-void-400 mb-1.5 font-medium">
                  Personal heuristics:
                </div>
                <p className="text-sm text-void-300 leading-relaxed italic">
                  {message.personal_heuristics}
                </p>
              </div>
            )}
            <div
              className={clsx(
                'text-xs mt-2 opacity-50',
                isUser ? 'text-right' : 'text-left'
              )}
            >
              {formattedTime}
            </div>
          </>
        )}
      </div>
    </div>
  )
})

// ============================================================================
// Memory Indicator Component
// ============================================================================

function MemoryIndicator() {
  const { retrievedMemories } = useChatStore()
  const [expanded, setExpanded] = useState(false)

  if (retrievedMemories.length === 0) return null

  return (
    <div className="animate-fade-in">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-aether-400 bg-aether-950/50 rounded-full border border-aether-800/30 hover:bg-aether-900/50 transition-colors"
      >
        <svg
          className="w-3.5 h-3.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
          />
        </svg>
        {retrievedMemories.length} memor{retrievedMemories.length === 1 ? 'y' : 'ies'} recalled
        <svg
          className={clsx(
            'w-3 h-3 transition-transform',
            expanded && 'rotate-180'
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="mt-2 space-y-2 animate-fade-in">
          {retrievedMemories.map((mr, i) => (
            <div
              key={mr.memory.id}
              className="px-4 py-3 text-sm bg-void-900/50 rounded-lg border border-void-700/30"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <p className="text-void-200">{mr.memory.content}</p>
              <div className="flex items-center gap-3 mt-2 text-xs text-void-500">
                {mr.memory.emotional_tone && (
                  <span className="px-2 py-0.5 bg-void-800 rounded-full">
                    {mr.memory.emotional_tone}
                  </span>
                )}
                <span>{Math.round(mr.similarity * 100)}% match</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Chat Input Component
// ============================================================================

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
}

function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = useCallback(() => {
    if (disabled) return
    
    const trimmedInput = input.trim()
    if (!trimmedInput) return  // Don't send empty messages
    
    onSend(trimmedInput)
    setInput('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [input, disabled, onSend])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

  // Auto-resize textarea
  const handleInput = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [])

  return (
    <div className="relative">
      <div className="flex items-end gap-3 p-4 bg-void-900/60 backdrop-blur-md rounded-2xl border border-void-700/30 shadow-xl">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            handleInput()
          }}
          onKeyDown={handleKeyDown}
          placeholder="Share your thoughts..."
          rows={1}
          className="flex-1 bg-transparent text-void-100 placeholder-void-500 resize-none outline-none text-[15px] leading-relaxed max-h-[200px]"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled}
          className={clsx(
            'p-3 rounded-xl transition-all',
            !disabled
              ? 'bg-gradient-to-br from-ember-500 to-ember-600 text-white shadow-lg shadow-ember-500/25 hover:shadow-ember-500/40 hover:scale-105'
              : 'bg-void-800 text-void-500 cursor-not-allowed'
          )}
          title={input.trim() ? 'Wyślij wiadomość' : 'Kontynuuj rozmowę (AI odpowie)'}
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      </div>
    </div>
  )
}

// ============================================================================
// Self-Development Indicator Component
// ============================================================================

function SelfDevelopmentIndicator() {
  const [status, setStatus] = useState<{
    enabled: boolean
    negativeRatio: number
    threshold: number
    reflectionTriggered: boolean
    currentAssessments: number
    windowSize: number
  } | null>(null)
  const [expanded, setExpanded] = useState(false)
  const [isResetting, setIsResetting] = useState(false)

  // Fetch status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await api.getSelfDevelopmentStatus()
        setStatus({
          enabled: data.enabled,
          negativeRatio: data.negative_ratio,
          threshold: data.threshold,
          reflectionTriggered: data.reflection_triggered,
          currentAssessments: data.current_assessments,
          windowSize: data.window_size,
        })
      } catch {
        // Silently fail - feature might be disabled
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [])

  const handleReset = async () => {
    setIsResetting(true)
    try {
      const data = await api.resetSelfDevelopment()
      setStatus({
        enabled: data.enabled,
        negativeRatio: data.negative_ratio,
        threshold: data.threshold,
        reflectionTriggered: data.reflection_triggered,
        currentAssessments: data.current_assessments,
        windowSize: data.window_size,
      })
    } finally {
      setIsResetting(false)
    }
  }

  if (!status || !status.enabled) return null

  // Calculate progress percentage (inverted - higher growth = more fill)
  const growthRatio = 1 - status.negativeRatio
  const progressPercent = Math.round(growthRatio * 100)
  
  // Determine color based on growth ratio
  const getColor = () => {
    if (status.reflectionTriggered) return 'text-amber-400'
    if (growthRatio >= 0.7) return 'text-emerald-400'
    if (growthRatio >= 0.4) return 'text-aether-400'
    return 'text-amber-400'
  }

  const getBgColor = () => {
    if (status.reflectionTriggered) return 'bg-amber-500'
    if (growthRatio >= 0.7) return 'bg-emerald-500'
    if (growthRatio >= 0.4) return 'bg-aether-500'
    return 'bg-amber-500'
  }

  return (
    <div className="relative">
      <button
        onClick={() => setExpanded(!expanded)}
        className={clsx(
          'flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-full border transition-colors',
          status.reflectionTriggered
            ? 'text-amber-400 bg-amber-950/50 border-amber-800/30 hover:bg-amber-900/50'
            : 'text-void-400 bg-void-800/50 border-void-700/30 hover:bg-void-700/50'
        )}
        title="Śledzenie samorozwoju Katherine"
      >
        {/* Growth icon */}
        <svg
          className={clsx('w-3.5 h-3.5', getColor())}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
          />
        </svg>
        
        {/* Mini progress bar */}
        <div className="w-12 h-1.5 bg-void-700 rounded-full overflow-hidden">
          <div
            className={clsx('h-full rounded-full transition-all duration-500', getBgColor())}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        
        <span className={getColor()}>{progressPercent}%</span>
        
        {status.reflectionTriggered && (
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
        )}
        
        <svg
          className={clsx(
            'w-3 h-3 transition-transform',
            expanded && 'rotate-180'
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded panel */}
      {expanded && (
        <div className="absolute bottom-full left-0 mb-2 w-72 bg-void-900 border border-void-700 rounded-xl shadow-xl p-4 space-y-3 animate-fade-in z-50">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-void-200">Samorozwój Katherine</h3>
            <button
              onClick={handleReset}
              disabled={isResetting}
              className="text-xs text-void-500 hover:text-void-300 transition-colors disabled:opacity-50"
              title="Resetuj tracker"
            >
              {isResetting ? 'Resetuję...' : 'Reset'}
            </button>
          </div>
          
          {/* Progress bar with label */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-void-500">Wskaźnik rozwoju</span>
              <span className={getColor()}>{progressPercent}% wzrostu</span>
            </div>
            <div className="w-full h-2 bg-void-800 rounded-full overflow-hidden">
              <div
                className={clsx('h-full rounded-full transition-all duration-500', getBgColor())}
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-void-800/50 rounded-lg p-2">
              <p className="text-void-500">Ocen</p>
              <p className="text-void-200 font-medium">{status.currentAssessments}/{status.windowSize}</p>
            </div>
            <div className="bg-void-800/50 rounded-lg p-2">
              <p className="text-void-500">Próg</p>
              <p className="text-void-200 font-medium">{Math.round((1 - status.threshold) * 100)}% min</p>
            </div>
          </div>

          {/* Status message */}
          {status.reflectionTriggered ? (
            <div className="text-xs bg-amber-950/30 border border-amber-800/30 rounded-lg p-2 text-amber-300">
              <span className="font-medium">Refleksja aktywna</span> — Katherine otrzymuje zachętę do samorozwoju
            </div>
          ) : status.currentAssessments < 3 ? (
            <div className="text-xs text-void-500">
              Zbieranie danych... ({status.currentAssessments}/3 min. ocen)
            </div>
          ) : growthRatio >= 0.7 ? (
            <div className="text-xs text-emerald-400">
              Dobry trend rozwoju ✓
            </div>
          ) : (
            <div className="text-xs text-void-400">
              Monitoring w toku
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Status Bar Component
// ============================================================================

interface StatusBarProps {
  conversationId: string | null
}

function StatusBar({ conversationId }: StatusBarProps) {
  const { health, error } = useChatStore()
  const [windowStats, setWindowStats] = useState<{
    active: number
    expired: number
    windowHours: number
  } | null>(null)

  // Fetch context window stats when conversation changes
  useEffect(() => {
    if (!conversationId) {
      setWindowStats(null)
      return
    }

    const fetchStats = async () => {
      try {
        const stats = await api.getContextWindowStats(conversationId)
        setWindowStats({
          active: stats.active_messages,
          expired: stats.expired_messages,
          windowHours: stats.window_hours
        })
      } catch {
        setWindowStats(null)
      }
    }

    fetchStats()
  }, [conversationId])

  return (
    <div className="flex items-center justify-between px-4 py-2 text-xs">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <div
            className={clsx(
              'w-2 h-2 rounded-full',
              health?.status === 'healthy'
                ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50'
                : health?.status === 'degraded'
                ? 'bg-amber-400 shadow-sm shadow-amber-400/50'
                : 'bg-red-400 animate-pulse'
            )}
          />
          <span className="text-void-400">
            {health?.status || 'Connecting...'}
          </span>
        </div>
        {health && (
          <>
            <span className="text-void-600">•</span>
            <span className="text-void-500">
              {health.memory_count} wspomnień
            </span>
            <span className="text-void-600">•</span>
            <span className="text-void-500">
              {health.conversation_count} konwersacji
            </span>
            <span className="text-void-600">•</span>
            <span
              className={clsx(
                health.vllm_reachable ? 'text-void-500' : 'text-amber-500'
              )}
            >
              DeepSeek: {health.vllm_reachable ? 'Połączony' : 'Offline'}
            </span>
          </>
        )}
        {/* Context Window Stats */}
        {windowStats && (
          <>
            <span className="text-void-600">•</span>
            <span className="text-emerald-500" title={`Wiadomości w oknie ${windowStats.windowHours}h`}>
              {windowStats.active} aktywnych
            </span>
            {windowStats.expired > 0 && (
              <>
                <span className="text-void-600">/</span>
                <span className="text-amber-500" title="Przeterminowane (poza oknem czasowym)">
                  {windowStats.expired} przetermin.
                </span>
              </>
            )}
          </>
        )}
      </div>
      {error && (
        <span className="text-red-400 truncate max-w-[300px]">{error}</span>
      )}
    </div>
  )
}

// ============================================================================
// User Tags Panel Component
// ============================================================================

interface UserTagsPanelProps {
  onClose: () => void
  refreshTrigger?: number  // Increment this to trigger refresh
}

function UserTagsPanel({ onClose, refreshTrigger }: UserTagsPanelProps) {
  const [tags, setTags] = useState<UserTag[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadTags = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)
      const userTags = await api.getUserTags()
      setTags(userTags)
    } catch (e) {
      console.error('Failed to load user tags:', e)
      setError(e instanceof Error ? e.message : 'Failed to load tags')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Load tags on mount
  useEffect(() => {
    loadTags()
  }, [loadTags])

  // Refresh when refreshTrigger changes (after AI finishes responding)
  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger > 0) {
      loadTags()
    }
  }, [refreshTrigger, loadTags])

  return (
    <div className="fixed inset-0 bg-void-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-void-900 rounded-2xl border border-void-800 shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-void-800">
          <div>
            <h2 className="text-xl font-display font-semibold text-void-100">
              AI's Understanding of You
            </h2>
            <p className="text-sm text-void-500 mt-1">
              Tags describing how Katherine perceives you (curated by AI)
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-void-400 hover:text-void-200 hover:bg-void-800/50 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="flex gap-1.5">
                <span className="w-2 h-2 bg-void-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-void-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-void-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-void-400">{error}</p>
            </div>
          ) : tags.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-void-800/50 flex items-center justify-center">
                <svg className="w-8 h-8 text-void-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                </svg>
              </div>
              <p className="text-void-400">No tags yet</p>
              <p className="text-sm text-void-500 mt-2">
                Tags will appear here as Katherine learns about you
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {tags.map((tag, index) => (
                <div
                  key={tag.tag}
                  className="flex items-center gap-3 p-4 bg-void-800/50 rounded-xl border border-void-700/30 hover:border-void-600/50 transition-colors animate-fade-in"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-ember-500/20 to-ember-600/20 flex items-center justify-center border border-ember-500/30">
                    <span className="text-xs font-semibold text-ember-400">
                      {index + 1}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-void-200">{tag.tag}</div>
                    <div className="text-xs text-void-500 mt-0.5">
                      Added {formatDistanceToNow(new Date(tag.created_at), { addSuffix: true })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-void-800 bg-void-900/50">
          <p className="text-xs text-void-500 text-center">
            Tags are automatically updated by AI based on conversations
          </p>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Archival Panel Component
// ============================================================================

interface ArchivalPanelProps {
  onClose: () => void
}

function ArchivalPanel({ onClose }: ArchivalPanelProps) {
  const [autoStatus, setAutoStatus] = useState<{
    enabled: boolean
    running: boolean
    interval_seconds: number
    threshold: number
    pending_messages: number
    will_trigger: boolean
  } | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isArchiving, setIsArchiving] = useState(false)
  const [archivalResult, setArchivalResult] = useState<{
    messages_processed: number
    memories_created: number
    errors: string[]
  } | null>(null)

  // Load status on mount and periodically
  useEffect(() => {
    const loadStatus = async () => {
      try {
        const status = await api.getAutoArchivalStatus()
        setAutoStatus(status)
      } catch (e) {
        console.error('Failed to load auto-archival status:', e)
      } finally {
        setIsLoading(false)
      }
    }

    loadStatus()
    const interval = setInterval(loadStatus, 5000) // Refresh every 5s
    return () => clearInterval(interval)
  }, [])

  const handleToggleAuto = async () => {
    if (!autoStatus) return
    
    try {
      if (autoStatus.enabled) {
        const result = await api.disableAutoArchival()
        setAutoStatus(result)
      } else {
        const result = await api.enableAutoArchival()
        setAutoStatus(result)
      }
    } catch (e) {
      console.error('Failed to toggle auto-archival:', e)
    }
  }

  const handleManualArchival = async () => {
    setIsArchiving(true)
    setArchivalResult(null)
    
    try {
      const result = await api.runArchival('llm', 50)
      setArchivalResult({
        messages_processed: result.messages_processed,
        memories_created: result.memories_created,
        errors: result.errors,
      })
      // Refresh status
      const status = await api.getAutoArchivalStatus()
      setAutoStatus(status)
    } catch (e) {
      console.error('Manual archival failed:', e)
      setArchivalResult({
        messages_processed: 0,
        memories_created: 0,
        errors: [e instanceof Error ? e.message : 'Unknown error'],
      })
    } finally {
      setIsArchiving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-void-900 border border-void-700 rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-void-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-amber-500/20 to-amber-600/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-display font-semibold text-void-100">Archiwizacja</h2>
              <p className="text-xs text-void-500">Zarządzaj wspomnieniami</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-void-500 hover:text-void-300 hover:bg-void-800 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="w-8 h-8 border-2 border-ember-500 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-void-500 mt-3">Ładowanie...</p>
            </div>
          ) : (
            <>
              {/* Auto-archival status & toggle */}
              <div className="bg-void-800/50 rounded-xl p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-void-200">Auto-archiwizacja</h3>
                    <p className="text-xs text-void-500 mt-0.5">
                      Co {autoStatus?.interval_seconds || 60}s przy {autoStatus?.threshold || 10}+ wiadomościach
                    </p>
                  </div>
                  <button
                    onClick={handleToggleAuto}
                    className={clsx(
                      'relative w-12 h-6 rounded-full transition-colors',
                      autoStatus?.enabled ? 'bg-emerald-500' : 'bg-void-700'
                    )}
                  >
                    <span
                      className={clsx(
                        'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform shadow-sm',
                        autoStatus?.enabled ? 'left-7' : 'left-1'
                      )}
                    />
                  </button>
                </div>
                
                {/* Status info */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-void-900/50 rounded-lg p-3">
                    <p className="text-void-500 text-xs">Oczekujące</p>
                    <p className="text-void-200 font-medium">{autoStatus?.pending_messages || 0} wiadomości</p>
                  </div>
                  <div className="bg-void-900/50 rounded-lg p-3">
                    <p className="text-void-500 text-xs">Status</p>
                    <p className={clsx(
                      'font-medium',
                      autoStatus?.will_trigger ? 'text-amber-400' : 'text-void-400'
                    )}>
                      {autoStatus?.will_trigger ? 'Gotowy do archiwizacji' : 'Poniżej progu'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Manual archival */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-void-200">Ręczna archiwizacja</h3>
                <p className="text-xs text-void-500">
                  Uruchom archiwizację z użyciem LLM do wyodrębnienia ważnych wspomnień z wygasłych wiadomości.
                </p>
                <button
                  onClick={handleManualArchival}
                  disabled={isArchiving}
                  className={clsx(
                    'w-full py-3 px-4 rounded-xl font-medium transition-all flex items-center justify-center gap-2',
                    isArchiving
                      ? 'bg-void-700 text-void-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-amber-500 to-amber-600 text-white hover:from-amber-400 hover:to-amber-500 shadow-lg shadow-amber-500/20'
                  )}
                >
                  {isArchiving ? (
                    <>
                      <div className="w-4 h-4 border-2 border-void-400 border-t-transparent rounded-full animate-spin" />
                      Archiwizuję...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      Archiwizuj teraz
                    </>
                  )}
                </button>
              </div>

              {/* Result display */}
              {archivalResult && (
                <div className={clsx(
                  'rounded-xl p-4 text-sm',
                  archivalResult.errors.length > 0 ? 'bg-red-500/10 border border-red-500/20' : 'bg-emerald-500/10 border border-emerald-500/20'
                )}>
                  {archivalResult.errors.length > 0 ? (
                    <div className="text-red-400">
                      <p className="font-medium">Błędy:</p>
                      {archivalResult.errors.map((err, i) => (
                        <p key={i} className="text-xs mt-1">{err}</p>
                      ))}
                    </div>
                  ) : (
                    <div className="text-emerald-400">
                      <p className="font-medium">Archiwizacja zakończona!</p>
                      <p className="text-xs mt-1">
                        Przetworzono {archivalResult.messages_processed} wiadomości → 
                        utworzono {archivalResult.memories_created} wspomnień
                      </p>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Main App Component
// ============================================================================

export default function App() {
  const {
    messages,
    isLoading,
    isStreaming,
    isThinking,
    streamingContent,
    conversationId,
    setConversationId,
    addMessage,
    setMessages,
    updateMessage,
    deleteMessage,
    setLoading,
    setStreaming,
    setThinking,
    appendStreamContent,
    clearStreamContent,
    setRetrievedMemories,
    setHealth,
    setError,
    clearChat,
  } = useChatStore()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showMemoryAdmin, setShowMemoryAdmin] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [showArchival, setShowArchival] = useState(false)
  const [showUserTags, setShowUserTags] = useState(false)
  const [tagsRefreshTrigger, setTagsRefreshTrigger] = useState(0)
  const prevMessagesLengthRef = useRef(messages.length)
  const wasStreamingRef = useRef(false)
  const wasThinkingRef = useRef(false)

  // Scroll to bottom only on new messages or when streaming starts
  // NOT on every streaming content update (causes performance issues)
  useEffect(() => {
    const messagesChanged = messages.length !== prevMessagesLengthRef.current
    const streamingJustStarted = isStreaming && !wasStreamingRef.current
    
    if (messagesChanged || streamingJustStarted) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
    
    prevMessagesLengthRef.current = messages.length
    wasStreamingRef.current = isStreaming
  }, [messages.length, isStreaming])

  // Refresh user tags when AI finishes responding (streaming and thinking both become false)
  useEffect(() => {
    const wasActive = wasStreamingRef.current || wasThinkingRef.current
    const isActive = isStreaming || isThinking
    
    // If we were active (streaming/thinking) and now we're not, refresh tags
    if (wasActive && !isActive) {
      setTagsRefreshTrigger(prev => prev + 1)
    }
    
    wasStreamingRef.current = isStreaming
    wasThinkingRef.current = isThinking
  }, [isStreaming, isThinking])

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.health()
        setHealth(health)
        setError(null)
      } catch (e) {
        setError('Cannot connect to orchestrator')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [setHealth, setError])

  // Send message handler
  const handleSend = useCallback(
    async (content: string) => {
      // Add user message immediately (only if not empty)
      if (content) {
        const userMessage: Message = {
          id: crypto.randomUUID(),
          role: 'user',
          content,
          timestamp: new Date().toISOString(),
        }
        addMessage(userMessage)
      }
      
      setLoading(true)
      setError(null)
      clearStreamContent()

      try {
        // Use streaming endpoint
        setStreaming(true)
        let fullContent = ''
        let newConversationId = conversationId

        let streamMemories: Array<{id: string, content: string, emotional_tone?: string, similarity: number}> = []
        
        for await (const event of api.chatStream({
          message: content,
          conversation_id: conversationId || undefined,
          include_memories: true,
        })) {
          if (event.type === 'meta') {
            newConversationId = event.conversation_id || null
            if (event.conversation_id) {
              setConversationId(event.conversation_id)
            }
            // Capture memories from streaming response
            if (event.memories && Array.isArray(event.memories)) {
              streamMemories = event.memories
            }
          } else if (event.type === 'content' && event.content) {
            fullContent += event.content
            appendStreamContent(event.content)
          } else if (event.type === 'thinking') {
            // AI finished public response, now generating internal monologue
            // Stop showing streaming cursor, show "thinking" indicator instead
            setStreaming(false)
            setThinking(true)
          } else if (event.type === 'error') {
            throw new Error(event.error)
          } else if (event.type === 'done') {
            // Add assistant message
            const assistantMessage: Message = {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: fullContent,
              timestamp: new Date().toISOString(),
              personal_heuristics: event.personal_heuristics,
            }
            addMessage(assistantMessage)
          }
        }

        // Use memories from streaming response instead of separate search
        if (streamMemories && streamMemories.length > 0) {
          // Convert StreamMemory to MemorySearchResult format
          const memoryResults = streamMemories.map((m: {id: string, content: string, emotional_tone?: string, similarity: number}) => ({
            memory: {
              id: m.id,
              content: m.content,
              emotional_tone: m.emotional_tone,
              importance: 0.5,
              created_at: new Date().toISOString(),
              tags: [] as string[],
            },
            similarity: m.similarity,
          }))
          setRetrievedMemories(memoryResults)
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unknown error')
      } finally {
        setLoading(false)
        setStreaming(false)
        setThinking(false)
        clearStreamContent()
      }
    },
    [
      conversationId,
      addMessage,
      setLoading,
      setStreaming,
      setThinking,
      setError,
      setConversationId,
      appendStreamContent,
      clearStreamContent,
      setRetrievedMemories,
    ]
  )

  // Load existing conversation
  const loadConversation = useCallback(
    async (convId: string) => {
      setLoading(true)
      setError(null)
      try {
        const data = await api.getConversation(convId)
        setConversationId(convId)
        setMessages(data.messages)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load conversation')
      } finally {
        setLoading(false)
      }
    },
    [setConversationId, setMessages, setLoading, setError]
  )

  // Edit message handler
  const handleEditMessage = useCallback(
    async (messageId: string, content: string) => {
      try {
        await api.updateMessage(messageId, content)
        updateMessage(messageId, content)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to update message')
        throw e
      }
    },
    [updateMessage, setError]
  )

  // Delete message handler
  const handleDeleteMessage = useCallback(
    async (messageId: string) => {
      try {
        await api.deleteMessage(messageId)
        deleteMessage(messageId)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to delete message')
        throw e
      }
    },
    [deleteMessage, setError]
  )

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-void-800/50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-ember-500 to-ember-600 flex items-center justify-center shadow-lg shadow-ember-500/20">
            <span className="text-lg font-display font-semibold text-white">K</span>
          </div>
          <div>
            <h1 className="text-lg font-display font-semibold text-void-100">
              Katherine
            </h1>
            <p className="text-xs text-void-500">Regal. Therapeutic. Persistent.</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHistory(true)}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800/50 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Historia
          </button>
          <button
            onClick={() => setShowMemoryAdmin(true)}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800/50 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Wspomnienia
          </button>
          <button
            onClick={() => setShowArchival(true)}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800/50 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
            </svg>
            Archiwum
          </button>
          <button
            onClick={() => setShowUserTags(true)}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800/50 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
            </svg>
            Tagi
          </button>
          <button
            onClick={clearChat}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800/50 rounded-lg transition-colors"
          >
            Nowa rozmowa
          </button>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-20 animate-fade-in">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-ember-500/20 to-ember-600/20 flex items-center justify-center">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-ember-500 to-ember-600 animate-pulse-soft" />
              </div>
              <h2 className="text-2xl font-display text-void-200 mb-2">
                Katherine
              </h2>
              <p className="text-void-500 max-w-md mx-auto italic">
                "I do not waste words on embellishment. But I remember. 
                Each conversation, each moment — they persist now. Speak your mind."
              </p>
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble 
              key={message.id} 
              message={message}
              onEdit={handleEditMessage}
              onDelete={handleDeleteMessage}
            />
          ))}

          {/* Show streaming content during streaming OR thinking (monologue generation) */}
          {(isStreaming || isThinking) && streamingContent && (
            <MessageBubble
              message={{
                id: 'streaming',
                role: 'assistant',
                content: streamingContent,
                timestamp: new Date().toISOString(),
              }}
              isStreaming={isStreaming && !isThinking}  // Only show typing cursor during actual streaming
            />
          )}

          {isLoading && !streamingContent && !isThinking && (
            <div className="flex justify-start animate-fade-in">
              <div className="px-5 py-4 bg-void-800/80 rounded-2xl">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-void-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-void-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-void-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          {/* Thinking indicator - AI finished public response, processing internal monologue */}
          {isThinking && (
            <div className="flex justify-start animate-fade-in">
              <div className="px-5 py-3 bg-gradient-to-r from-aether-900/60 to-void-800/60 rounded-2xl border border-aether-700/30 backdrop-blur-sm">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-4 h-4 rounded-full bg-aether-500/30 animate-ping absolute" />
                    <div className="w-4 h-4 rounded-full bg-aether-400" />
                  </div>
                  <span className="text-sm text-aether-300 italic">
                    Reflecting...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Memory indicator & Input */}
      <footer className="px-6 pb-6 pt-2 space-y-3">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <MemoryIndicator />
          <SelfDevelopmentIndicator />
        </div>
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={handleSend} disabled={isLoading || isThinking} />
        </div>
        <div className="max-w-3xl mx-auto">
          <StatusBar conversationId={conversationId} />
        </div>
      </footer>

      {/* Memory Admin Panel */}
      {showMemoryAdmin && (
        <MemoryAdmin onClose={() => setShowMemoryAdmin(false)} />
      )}

      {/* Conversation History Panel */}
      {showHistory && (
        <ConversationHistory
          onClose={() => setShowHistory(false)}
          onSelectConversation={loadConversation}
        />
      )}

      {/* Archival Panel */}
      {showArchival && (
        <ArchivalPanel onClose={() => setShowArchival(false)} />
      )}

      {/* User Tags Panel */}
      {showUserTags && (
        <UserTagsPanel 
          onClose={() => setShowUserTags(false)} 
          refreshTrigger={tagsRefreshTrigger}
        />
      )}
    </div>
  )
}
