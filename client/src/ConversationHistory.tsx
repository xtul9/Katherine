/**
 * Katherine Client - Conversation History Panel
 * Browse, continue, and manage past conversations.
 */

import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import type { Conversation, ContextWindowStats } from './types'
import { formatDistanceToNow, format } from 'date-fns'
import { pl } from 'date-fns/locale'

// ============================================================================
// Conversation Card
// ============================================================================

interface ConversationCardProps {
  conversation: Conversation
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

function ConversationCard({ conversation, onSelect, onDelete }: ConversationCardProps) {
  const [isDeleting, setIsDeleting] = useState(false)
  const [windowStats, setWindowStats] = useState<ContextWindowStats | null>(null)
  const [showStats, setShowStats] = useState(false)

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Czy na pewno chcesz usunąć tę konwersację?')) return
    
    setIsDeleting(true)
    try {
      await onDelete(conversation.id)
    } finally {
      setIsDeleting(false)
    }
  }

  const handleToggleStats = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!showStats && !windowStats) {
      try {
        const stats = await api.getContextWindowStats(conversation.id)
        setWindowStats(stats)
      } catch {
        // Ignore errors
      }
    }
    setShowStats(!showStats)
  }

  return (
    <div
      onClick={() => onSelect(conversation.id)}
      className="p-4 bg-void-900/60 rounded-xl border border-void-700/30 hover:border-ember-500/30 hover:bg-void-800/60 cursor-pointer transition-all group"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-void-200 truncate">
            {conversation.title || 'Bez tytułu'}
          </h3>
          <div className="flex items-center gap-3 mt-1 text-xs text-void-500">
            <span>{conversation.message_count || 0} wiadomości</span>
            <span>•</span>
            <span>{formatDistanceToNow(new Date(conversation.updated_at), { addSuffix: true, locale: pl })}</span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {/* Context Window Stats Button */}
          <button
            onClick={handleToggleStats}
            className="p-1.5 text-void-600 hover:text-aether-400 hover:bg-aether-950/30 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
            title="Statystyki okna czasowego"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          {/* Delete Button */}
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="p-1.5 text-void-600 hover:text-red-400 hover:bg-red-950/30 rounded-lg opacity-0 group-hover:opacity-100 transition-all disabled:opacity-50"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>
      
      {/* Context Window Stats Panel */}
      {showStats && windowStats && (
        <div className="mt-3 pt-3 border-t border-void-700/30 animate-fade-in" onClick={e => e.stopPropagation()}>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="px-3 py-2 bg-emerald-950/30 rounded-lg border border-emerald-800/20">
              <div className="text-emerald-400 font-medium">{windowStats.active_messages}</div>
              <div className="text-emerald-600">Aktywnych (w oknie {windowStats.window_hours}h)</div>
            </div>
            <div className="px-3 py-2 bg-amber-950/30 rounded-lg border border-amber-800/20">
              <div className="text-amber-400 font-medium">{windowStats.expired_messages}</div>
              <div className="text-amber-600">Przeterminowanych</div>
            </div>
          </div>
          <div className="mt-2 text-xs text-void-600">
            Granica okna: {format(new Date(windowStats.cutoff_time), 'dd MMM yyyy, HH:mm', { locale: pl })}
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Expired Messages Panel
// ============================================================================

interface ExpiredMessagesPanelProps {
  onClose: () => void
}

function ExpiredMessagesPanel({ onClose }: ExpiredMessagesPanelProps) {
  const [messages, setMessages] = useState<Array<{
    message: { id: string; role: string; content: string; timestamp: string }
    conversation_id: string
    conversation_title?: string
  }>>([])
  const [isLoading, setIsLoading] = useState(true)
  const [windowHours, setWindowHours] = useState(24)
  const [cutoffTime, setCutoffTime] = useState<string | null>(null)
  const [isArchiving, setIsArchiving] = useState(false)
  const [archiveResult, setArchiveResult] = useState<{
    success: boolean
    message: string
    memoriesCreated?: number
  } | null>(null)

  const loadExpired = async () => {
    setIsLoading(true)
    try {
      const data = await api.getExpiredMessages(100)
      setMessages(data.messages)
      setWindowHours(data.window_hours)
      setCutoffTime(data.cutoff_time)
    } catch {
      // Ignore errors
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadExpired()
  }, [])

  const handleArchive = async (mode: 'simple' | 'llm') => {
    setIsArchiving(true)
    setArchiveResult(null)
    try {
      const result = await api.runFullArchival(mode)
      setArchiveResult({
        success: true,
        message: `Zarchiwizowano ${result.total_messages_processed} wiadomości`,
        memoriesCreated: result.total_memories_created
      })
      // Reload expired messages
      await loadExpired()
    } catch (e) {
      setArchiveResult({
        success: false,
        message: e instanceof Error ? e.message : 'Archiwizacja nie powiodła się'
      })
    } finally {
      setIsArchiving(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-void-950/95 backdrop-blur-sm z-50 overflow-hidden flex flex-col">
      <header className="px-6 py-4 border-b border-void-800/50 flex items-center justify-between bg-void-900/80 backdrop-blur-md">
        <div>
          <h1 className="text-xl font-display text-void-100">Przeterminowane wiadomości</h1>
          <p className="text-sm text-void-500">
            Wiadomości starsze niż {windowHours}h
            {cutoffTime && (
              <span className="ml-2 text-void-600">
                (przed {format(new Date(cutoffTime), 'dd MMM, HH:mm', { locale: pl })})
              </span>
            )}
          </p>
        </div>
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800 rounded-lg transition-colors"
        >
          ← Powrót
        </button>
      </header>

      <main className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="flex gap-1.5">
              <span className="w-3 h-3 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-3 h-3 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-3 h-3 bg-amber-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-950/30 flex items-center justify-center">
              <svg className="w-8 h-8 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-void-400">Wszystkie wiadomości są w oknie czasowym!</p>
            <p className="text-void-600 text-sm mt-1">Brak przeterminowanych wiadomości do archiwizacji.</p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-3">
            {/* Archive Controls */}
            <div className="px-4 py-4 bg-amber-950/20 border border-amber-800/20 rounded-lg mb-6">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm text-amber-300">
                  <strong>{messages.length}</strong> wiadomości poza oknem {windowHours}h.
                </p>
              </div>
              
              {archiveResult && (
                <div className={`mb-3 px-3 py-2 rounded-lg text-sm ${
                  archiveResult.success 
                    ? 'bg-emerald-950/30 border border-emerald-800/20 text-emerald-300'
                    : 'bg-red-950/30 border border-red-800/20 text-red-300'
                }`}>
                  {archiveResult.message}
                  {archiveResult.memoriesCreated !== undefined && (
                    <span className="ml-2 text-emerald-400">
                      ({archiveResult.memoriesCreated} wspomnień utworzonych)
                    </span>
                  )}
                </div>
              )}
              
              <div className="flex items-center gap-3">
                <button
                  onClick={() => handleArchive('simple')}
                  disabled={isArchiving || messages.length === 0}
                  className="px-4 py-2 text-sm bg-amber-600 hover:bg-amber-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isArchiving ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Archiwizuję...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                      </svg>
                      Archiwizuj (prosty)
                    </>
                  )}
                </button>
                
                <button
                  onClick={() => handleArchive('llm')}
                  disabled={isArchiving || messages.length === 0}
                  className="px-4 py-2 text-sm bg-aether-600 hover:bg-aether-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  title="Użyj LLM do wyodrębnienia znaczących wspomnień (wolniejsze, ale mądrzejsze)"
                >
                  {isArchiving ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Archiwizuję...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      Archiwizuj (LLM)
                    </>
                  )}
                </button>
                
                <span className="text-xs text-void-600 ml-2">
                  Prosty = szybki, LLM = inteligentny (droższy)
                </span>
              </div>
            </div>
            
            {messages.map((item) => (
              <div
                key={item.message.id}
                className="p-4 bg-void-900/60 rounded-xl border border-void-700/30"
              >
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    item.message.role === 'user' 
                      ? 'bg-ember-600/20 text-ember-400' 
                      : 'bg-aether-600/20 text-aether-400'
                  }`}>
                    {item.message.role === 'user' ? 'M' : 'K'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 text-xs text-void-500 mb-1">
                      <span className="font-medium">
                        {item.message.role === 'user' ? 'Michael' : 'Katherine'}
                      </span>
                      <span>•</span>
                      <span>{format(new Date(item.message.timestamp), 'dd MMM yyyy, HH:mm', { locale: pl })}</span>
                      {item.conversation_title && (
                        <>
                          <span>•</span>
                          <span className="truncate text-void-600">{item.conversation_title}</span>
                        </>
                      )}
                    </div>
                    <p className="text-sm text-void-300 line-clamp-3">
                      {item.message.content}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

// ============================================================================
// Main Component
// ============================================================================

interface ConversationHistoryProps {
  onClose: () => void
  onSelectConversation: (conversationId: string) => void
}

export default function ConversationHistory({ onClose, onSelectConversation }: ConversationHistoryProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)
  const [showExpired, setShowExpired] = useState(false)

  const loadConversations = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.listConversations(100)
      setConversations(data.conversations)
      setTotal(data.total)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load conversations')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  const handleDelete = async (conversationId: string) => {
    try {
      await api.deleteConversation(conversationId)
      setConversations(prev => prev.filter(c => c.id !== conversationId))
      setTotal(prev => prev - 1)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete conversation')
    }
  }

  const handleSelect = (conversationId: string) => {
    onSelectConversation(conversationId)
    onClose()
  }

  // Show expired messages panel
  if (showExpired) {
    return <ExpiredMessagesPanel onClose={() => setShowExpired(false)} />
  }

  return (
    <div className="fixed inset-0 bg-void-950/90 backdrop-blur-sm z-40 overflow-hidden flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b border-void-800/50 flex items-center justify-between bg-void-900/80 backdrop-blur-md">
        <div>
          <h1 className="text-xl font-display text-void-100">Historia konwersacji</h1>
          <p className="text-sm text-void-500">
            {total} konwersacji
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowExpired(true)}
            className="px-4 py-2 text-sm text-amber-400 hover:text-amber-300 hover:bg-amber-950/30 rounded-lg transition-colors flex items-center gap-2"
            title="Pokaż wiadomości poza oknem czasowym"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Przeterminowane
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800 rounded-lg transition-colors"
          >
            ← Powrót do czatu
          </button>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-4 px-4 py-3 bg-red-950/30 border border-red-800/30 rounded-lg">
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="flex gap-1.5">
              <span className="w-3 h-3 bg-ember-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-3 h-3 bg-ember-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-3 h-3 bg-ember-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-void-800/50 flex items-center justify-center">
              <svg className="w-8 h-8 text-void-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <p className="text-void-500">Brak zapisanych konwersacji.</p>
            <p className="text-void-600 text-sm mt-1">Rozpocznij rozmowę z Katherine.</p>
          </div>
        ) : (
          <div className="grid gap-3 max-w-2xl mx-auto">
            {conversations.map((conv) => (
              <ConversationCard
                key={conv.id}
                conversation={conv}
                onSelect={handleSelect}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
