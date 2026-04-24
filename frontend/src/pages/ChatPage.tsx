import { useChatStore } from '@/store/chat'
import { useSendMessage } from '@/hooks/useSendMessage'
import { useInitialize } from '@/hooks/useInitialize'
import { Sidebar } from '@/components/Sidebar'
import { ChatHistory } from '@/components/ChatHistory'
import { MessageInput } from '@/components/MessageInput'
import { ModelSelector } from '@/components/ModelSelector'
import { ErrorMessage } from '@/components/ErrorMessage'

export function ChatPage() {
  useInitialize()

  const messages = useChatStore((state) => state.messages)
  const isLoading = useChatStore((state) => state.isLoading)
  const error = useChatStore((state) => state.error)
  const setError = useChatStore((state) => state.setError)
  const { sendMessage } = useSendMessage()

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header with Model Selector */}
        <ModelSelector />

        {/* Chat History */}
        <ChatHistory messages={messages} isLoading={isLoading} />

        {/* Error Message */}
        {error && (
          <ErrorMessage
            error={error}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Message Input */}
        <MessageInput onSend={sendMessage} isLoading={isLoading} />
      </div>
    </div>
  )
}
