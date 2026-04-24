import type { ChatMessage } from '@/types/api'

interface MessageItemProps {
  message: ChatMessage
}

export function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user'
  const bubbleClassName = isUser
    ? 'max-w-xs lg:max-w-md'
    : 'max-w-[85vw] lg:max-w-[50vw]'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`${bubbleClassName} px-4 py-3 rounded-lg ${
          isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-gray-200 text-gray-900 rounded-bl-none'
        }`}
      >
        {isUser && message.userImagePreviewUrl && (
          <div className="mb-3">
            <img src={message.userImagePreviewUrl} alt="Uploaded" className="max-w-xs rounded-lg" />
          </div>
        )}

        <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>

        {message.images && message.images.length > 0 && (
          <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-2">
            {message.images.map((img) => (
              <div key={img.vector_id} className="flex flex-col gap-1">
                {img.base64_image && (
                  <img
                    src={`data:image/jpeg;base64,${img.base64_image}`}
                    alt={img.image_name}
                    className="w-full rounded-lg object-cover"
                  />
                )}
                <span className="text-xs opacity-75 truncate">{img.image_name}</span>
              </div>
            ))}
          </div>
        )}

        <span className={`text-xs mt-2 block ${isUser ? 'opacity-75' : 'opacity-60'}`}>
          {message.timestamp.toLocaleTimeString()}
        </span>
      </div>
    </div>
  )
}
