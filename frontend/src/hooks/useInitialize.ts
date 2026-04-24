import { useEffect } from 'react'
import { useChatStore } from '@/store/chat'
import { apiClient } from '@/services/api'

export function useInitialize() {
  const setSessions = useChatStore((state) => state.setSessions)
  const setAvailableModels = useChatStore((state) => state.setAvailableModels)
  const setSelectedModel = useChatStore((state) => state.setSelectedModel)

  useEffect(() => {
    const initializeApp = async () => {
      try {
        const [models, sessions] = await Promise.all([
          apiClient.getAvailableModels(),
          apiClient.getSessions(),
        ])

        setAvailableModels(models)
        setSessions(sessions)

        if (models.length > 0) {
          setSelectedModel(models[0])
        }
      } catch (error) {
        console.error('Failed to initialize app:', error)
      }
    }

    initializeApp()
  }, [setAvailableModels, setSessions, setSelectedModel])
}
