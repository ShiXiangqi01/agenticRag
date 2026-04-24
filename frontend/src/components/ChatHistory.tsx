import { useEffect, useRef } from 'react'
import { MessageItem } from './MessageItem'
import type { ChatMessage } from '@/types/api'

interface ChatHistoryProps {
  messages: ChatMessage[]
  isLoading: boolean
}

export function ChatHistory({ messages, isLoading }: ChatHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 bg-white">
      {messages.length === 0 ? (
        <div className="h-full flex items-center justify-center text-gray-400">
          <p className="text-center">
            No messages yet. <br />
            Start a conversation!
          </p>
        </div>
      ) : (
        <>
          {messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="flex justify-start mb-4">
              <div className="bg-gray-200 text-gray-900 px-4 py-3 rounded-lg rounded-bl-none">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce delay-100"></div>
                  <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce delay-200"></div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
