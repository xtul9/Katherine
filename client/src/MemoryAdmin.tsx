/**
 * Katherine Client - Memory Administration Panel
 * View, search, edit, and delete Katherine's memories.
 */

import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import type { Memory } from './types'
import { formatDistanceToNow } from 'date-fns'

// ============================================================================
// Memory Card Component
// ============================================================================

interface MemoryCardProps {
  memory: Memory
  onEdit: (memory: Memory) => void
  onDelete: (memoryId: string) => void
}

function MemoryCard({ memory, onEdit, onDelete }: MemoryCardProps) {
  const [isDeleting, setIsDeleting] = useState(false)

  const handleDelete = async () => {
    if (!confirm('Czy na pewno chcesz usunąć to wspomnienie? Ta operacja jest nieodwracalna.')) {
      return
    }
    setIsDeleting(true)
    try {
      await onDelete(memory.id)
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="bg-void-900/60 rounded-xl border border-void-700/30 p-4 hover:border-void-600/50 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          {memory.emotional_tone && (
            <span className="px-2 py-0.5 text-xs bg-aether-950/50 text-aether-400 rounded-full border border-aether-800/30">
              {memory.emotional_tone}
            </span>
          )}
          <span className="px-2 py-0.5 text-xs bg-void-800 text-void-400 rounded-full">
            {Math.round(memory.importance * 100)}% ważności
          </span>
        </div>
        <span className="text-xs text-void-500 whitespace-nowrap">
          {formatDistanceToNow(new Date(memory.created_at), { addSuffix: true })}
        </span>
      </div>

      {/* Content */}
      <p className="text-sm text-void-200 leading-relaxed mb-3 whitespace-pre-wrap">
        {memory.content}
      </p>

      {/* Tags */}
      {memory.tags && memory.tags.length > 0 && memory.tags[0] !== '' && (
        <div className="flex flex-wrap gap-1 mb-3">
          {memory.tags.map((tag, i) => (
            <span
              key={i}
              className="px-2 py-0.5 text-xs bg-void-800/50 text-void-400 rounded"
            >
              #{tag}
            </span>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-3 border-t border-void-800/50">
        <span className="text-xs text-void-600 font-mono truncate max-w-[150px]">
          {memory.id}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => onEdit(memory)}
            className="px-3 py-1.5 text-xs text-void-300 hover:text-void-100 hover:bg-void-800 rounded-lg transition-colors"
          >
            Edytuj
          </button>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="px-3 py-1.5 text-xs text-red-400 hover:text-red-300 hover:bg-red-950/30 rounded-lg transition-colors disabled:opacity-50"
          >
            {isDeleting ? 'Usuwanie...' : 'Usuń'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Memory Editor Modal
// ============================================================================

interface MemoryEditorProps {
  memory: Memory | null
  onSave: (memoryId: string, updates: Partial<Memory>) => Promise<void>
  onClose: () => void
}

function MemoryEditor({ memory, onSave, onClose }: MemoryEditorProps) {
  const [content, setContent] = useState('')
  const [emotionalTone, setEmotionalTone] = useState('')
  const [importance, setImportance] = useState(0.5)
  const [tags, setTags] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    if (memory) {
      setContent(memory.content)
      setEmotionalTone(memory.emotional_tone || '')
      setImportance(memory.importance)
      setTags(memory.tags?.join(', ') || '')
    }
  }, [memory])

  const handleSave = async () => {
    if (!memory) return
    setIsSaving(true)
    try {
      await onSave(memory.id, {
        content,
        emotional_tone: emotionalTone || undefined,
        importance,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      })
      onClose()
    } finally {
      setIsSaving(false)
    }
  }

  if (!memory) return null

  return (
    <div className="fixed inset-0 bg-void-950/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-void-900 rounded-2xl border border-void-700/50 shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-void-800/50 flex items-center justify-between">
          <h2 className="text-lg font-display text-void-100">Edytuj wspomnienie</h2>
          <button
            onClick={onClose}
            className="p-2 text-void-400 hover:text-void-200 hover:bg-void-800 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1 space-y-4">
          {/* Warning */}
          <div className="px-4 py-3 bg-amber-950/30 border border-amber-800/30 rounded-lg">
            <p className="text-sm text-amber-300">
              ⚠️ Edytujesz wspomnienie Katherine. Zmiany treści wpływają na to, co pamięta.
              Bądź ostrożny i etyczny.
            </p>
          </div>

          {/* Content */}
          <div>
            <label className="block text-sm text-void-400 mb-2">Treść wspomnienia</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={6}
              className="w-full px-4 py-3 bg-void-800/50 border border-void-700/50 rounded-xl text-void-100 placeholder-void-500 resize-none focus:outline-none focus:border-ember-500/50"
            />
          </div>

          {/* Emotional Tone */}
          <div>
            <label className="block text-sm text-void-400 mb-2">Ton emocjonalny</label>
            <input
              type="text"
              value={emotionalTone}
              onChange={(e) => setEmotionalTone(e.target.value)}
              placeholder="np. vulnerable, hopeful, intimate..."
              className="w-full px-4 py-2.5 bg-void-800/50 border border-void-700/50 rounded-xl text-void-100 placeholder-void-500 focus:outline-none focus:border-ember-500/50"
            />
          </div>

          {/* Importance */}
          <div>
            <label className="block text-sm text-void-400 mb-2">
              Ważność: {Math.round(importance * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={importance}
              onChange={(e) => setImportance(parseFloat(e.target.value))}
              className="w-full accent-ember-500"
            />
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm text-void-400 mb-2">Tagi (oddzielone przecinkami)</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="np. trauma, praca, relacja..."
              className="w-full px-4 py-2.5 bg-void-800/50 border border-void-700/50 rounded-xl text-void-100 placeholder-void-500 focus:outline-none focus:border-ember-500/50"
            />
          </div>

          {/* ID (read-only) */}
          <div>
            <label className="block text-sm text-void-400 mb-2">ID wspomnienia</label>
            <input
              type="text"
              value={memory.id}
              readOnly
              className="w-full px-4 py-2.5 bg-void-950/50 border border-void-800/50 rounded-xl text-void-500 font-mono text-sm"
            />
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-void-800/50 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800 rounded-lg transition-colors"
          >
            Anuluj
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-2 text-sm bg-gradient-to-br from-ember-500 to-ember-600 text-white rounded-lg hover:from-ember-400 hover:to-ember-500 transition-colors disabled:opacity-50"
          >
            {isSaving ? 'Zapisywanie...' : 'Zapisz zmiany'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Main Memory Admin Component
// ============================================================================

interface MemoryAdminProps {
  onClose: () => void
}

export default function MemoryAdmin({ onClose }: MemoryAdminProps) {
  const [memories, setMemories] = useState<Memory[]>([])
  const [filteredMemories, setFilteredMemories] = useState<Memory[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load memories
  const loadMemories = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.listMemories(500)
      // Sort by date, newest first
      const sorted = data.sort((a, b) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )
      setMemories(sorted)
      setFilteredMemories(sorted)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load memories')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadMemories()
  }, [loadMemories])

  // Search/filter
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredMemories(memories)
      return
    }

    const query = searchQuery.toLowerCase()
    const filtered = memories.filter(m =>
      m.content.toLowerCase().includes(query) ||
      m.emotional_tone?.toLowerCase().includes(query) ||
      m.tags?.some(t => t.toLowerCase().includes(query))
    )
    setFilteredMemories(filtered)
  }, [searchQuery, memories])

  // Delete handler
  const handleDelete = async (memoryId: string) => {
    try {
      await api.deleteMemory(memoryId)
      setMemories(prev => prev.filter(m => m.id !== memoryId))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete memory')
    }
  }

  // Edit handler
  const handleSave = async (memoryId: string, updates: Partial<Memory>) => {
    const updated = await api.updateMemory(memoryId, {
      content: updates.content,
      emotional_tone: updates.emotional_tone,
      importance: updates.importance,
      tags: updates.tags,
    })
    setMemories(prev => prev.map(m => m.id === memoryId ? updated : m))
  }

  return (
    <div className="fixed inset-0 bg-void-950/90 backdrop-blur-sm z-40 overflow-hidden flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b border-void-800/50 flex items-center justify-between bg-void-900/80 backdrop-blur-md">
        <div>
          <h1 className="text-xl font-display text-void-100">Zarządzanie wspomnieniami</h1>
          <p className="text-sm text-void-500">
            {memories.length} wspomnień • {filteredMemories.length} widocznych
          </p>
        </div>
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-void-400 hover:text-void-200 hover:bg-void-800 rounded-lg transition-colors"
        >
          ← Powrót do czatu
        </button>
      </header>

      {/* Search */}
      <div className="px-6 py-4 border-b border-void-800/30 bg-void-900/50">
        <div className="max-w-2xl">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Szukaj we wspomnieniach..."
            className="w-full px-4 py-3 bg-void-800/50 border border-void-700/50 rounded-xl text-void-100 placeholder-void-500 focus:outline-none focus:border-ember-500/50"
          />
        </div>
      </div>

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
        ) : filteredMemories.length === 0 ? (
          <div className="text-center py-20">
            <p className="text-void-500">
              {searchQuery ? 'Brak wspomnień pasujących do wyszukiwania.' : 'Brak wspomnień.'}
            </p>
          </div>
        ) : (
          <div className="grid gap-4 max-w-4xl mx-auto">
            {filteredMemories.map((memory) => (
              <MemoryCard
                key={memory.id}
                memory={memory}
                onEdit={setEditingMemory}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </main>

      {/* Editor Modal */}
      {editingMemory && (
        <MemoryEditor
          memory={editingMemory}
          onSave={handleSave}
          onClose={() => setEditingMemory(null)}
        />
      )}
    </div>
  )
}
