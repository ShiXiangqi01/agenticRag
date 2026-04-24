import { create } from 'zustand'
import type { ChatMessage, SessionHandle, AgentModel } from '@/types/api'

interface ChatStore {
  // Chat state
  messages: ChatMessage[]
  currentSessionId: string | null
  selectedModel: AgentModel | null
  isLoading: boolean
  error: string | null

  // Sessions
  sessions: SessionHandle[]
  availableModels: AgentModel[]

  // Actions
  addMessage: (message: ChatMessage) => void
  setMessages: (messages: ChatMessage[]) => void
  setCurrentSessionId: (sessionId: string | null) => void
  setSelectedModel: (model: AgentModel | null) => void
  setIsLoading: (isLoading: boolean) => void
  setError: (error: string | null) => void
  setSessions: (sessions: SessionHandle[]) => void
  setAvailableModels: (models: AgentModel[]) => void
  clearChat: () => void
  createNewSession: () => void
  removeSession: (sessionId: string) => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  currentSessionId: null,
  selectedModel: null,
  isLoading: false,
  error: null,
  sessions: [],
  availableModels: [],

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  setMessages: (messages) => set({ messages }),

  setCurrentSessionId: (sessionId) => set({ currentSessionId: sessionId }),

  setSelectedModel: (model) => set({ selectedModel: model }),

  setIsLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error }),

  setSessions: (sessions) => set({ sessions }),

  setAvailableModels: (models) => set({ availableModels: models }),

  clearChat: () =>
    set({
      messages: [],
      currentSessionId: null,
      error: null,
    }),

  removeSession: (sessionId: string) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.session_id !== sessionId),
      currentSessionId: state.currentSessionId === sessionId ? null : state.currentSessionId,
    })),

  createNewSession: () =>
    set({
      messages: [],
      currentSessionId: null,
    }),
}))
