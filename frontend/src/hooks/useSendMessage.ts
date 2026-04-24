import { useCallback } from 'react'
import { useChatStore } from '@/store/chat'
import { apiClient } from '@/services/api'
import type { AgentResponse } from '@/types/api'

export function useSendMessage() {
  const addMessage = useChatStore((state) => state.addMessage)
  const setIsLoading = useChatStore((state) => state.setIsLoading)
  const setError = useChatStore((state) => state.setError)
  const currentSessionId = useChatStore((state) => state.currentSessionId)
  const selectedModel = useChatStore((state) => state.selectedModel)
  const setCurrentSessionId = useChatStore((state) => state.setCurrentSessionId)

  const sendMessage = useCallback(
    async (text: string, userImageFile?: File | null, userImagePreviewUrl?: string | null) => {
      if (!text.trim()) return

      const textContent = text.trim()

      setError(null)
      setIsLoading(true)

      // Add user message
      addMessage({
        id: Date.now().toString(),
        role: 'user',
        content: textContent,
        userImagePreviewUrl: userImagePreviewUrl ?? undefined,
        timestamp: new Date(),
      })

      try {
        let userImageId: string | null = null
        if (userImageFile) {
          const uploadResult = await apiClient.uploadUserImage(userImageFile)
          userImageId = uploadResult.user_image_id
        }

        const response: AgentResponse = await apiClient.sendMessage({
          message: textContent,
          user_image_id: userImageId,
          model_name: selectedModel?.name,
          session_id: currentSessionId,
        })

        // Update session ID if new
        if (!currentSessionId && response.session?.session_id) {
          setCurrentSessionId(response.session.session_id)
        }

        // Add assistant message
        addMessage({
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.message,
          images: response.images_list,
          timestamp: new Date(),
        })
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to send message'
        setError(errorMessage)
        console.error('Error sending message:', err)
      } finally {
        setIsLoading(false)
      }
    },
    [addMessage, setIsLoading, setError, currentSessionId, selectedModel, setCurrentSessionId]
  )

  return { sendMessage }
}
