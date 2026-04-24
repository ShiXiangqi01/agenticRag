import { useChatStore } from '@/store/chat'
import { Plus, X } from 'lucide-react'
import { apiClient } from '@/services/api'
import { useState } from 'react'

export function Sidebar() {
  const sessions = useChatStore((state) => state.sessions)
  const currentSessionId = useChatStore((state) => state.currentSessionId)
  const setCurrentSessionId = useChatStore((state) => state.setCurrentSessionId)
  const createNewSession = useChatStore((state) => state.createNewSession)
  const removeSession = useChatStore((state) => state.removeSession)
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null)

  const handleNewChat = () => {
    createNewSession()
    setCurrentSessionId(null)
  }

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      setDeletingSessionId(sessionId)
      await apiClient.deleteSession(sessionId)
      removeSession(sessionId)
    } catch (error) {
      console.error('Failed to delete session:', error)
      setDeletingSessionId(null)
    }
  }

  return (
    <div className="w-64 bg-gray-900 text-white p-4 flex flex-col h-full">
      <button
        onClick={handleNewChat}
        className="flex items-center gap-2 w-full px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 mb-4"
      >
        <Plus size={20} />
        New Chat
      </button>

      <div className="flex-1 overflow-y-auto">
        <h2 className="text-sm font-semibold text-gray-400 mb-3">Recent Sessions</h2>
        <div className="space-y-2">
          {sessions.length === 0 ? (
            <p className="text-xs text-gray-500">No sessions yet</p>
          ) : (
            sessions.map((session) => (
              <div
                key={session.session_id}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition ${
                  currentSessionId === session.session_id
                    ? 'bg-gray-700'
                    : 'hover:bg-gray-800'
                }`}
              >
                <button
                  onClick={() => setCurrentSessionId(session.session_id)}
                  className="flex-1 text-left"
                >
                  <div className="truncate">{session.session_id.substring(0, 8)}...</div>
                  <div className="text-xs text-gray-400">
                    {new Date(session.created_at).toLocaleDateString()}
                  </div>
                </button>
                <button
                  onClick={(e) => handleDeleteSession(session.session_id, e)}
                  disabled={deletingSessionId === session.session_id}
                  className="p-1 hover:bg-red-600 rounded text-gray-400 hover:text-white transition disabled:opacity-50"
                  title="Delete session"
                >
                  <X size={16} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="border-t border-gray-700 pt-4 text-xs text-gray-400">
        <p>Agentic RAG Chat</p>
      </div>
    </div>
  )
}
