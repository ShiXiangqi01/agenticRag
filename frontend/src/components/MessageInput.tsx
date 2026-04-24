import { Send, Loader2 } from 'lucide-react'
import { useRef, useState } from 'react'

interface MessageInputProps {
  onSend: (message: string, imageFile?: File | null, imagePreviewUrl?: string | null) => void
  isLoading?: boolean
}

export function MessageInput({ onSend, isLoading = false }: MessageInputProps) {
  const [message, setMessage] = useState('')
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && !isLoading) {
      const messageToSend = message.trim()
      const imageFileToSend = selectedImageFile
      const previewToSend = previewUrl

      setMessage('')
      setSelectedImageFile(null)
      setPreviewUrl(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      await onSend(messageToSend, imageFileToSend, previewToSend)
    }
  }

  const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Show preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreviewUrl(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    setSelectedImageFile(file)
  }

  const handleClearImage = () => {
    setSelectedImageFile(null)
    setPreviewUrl(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3 p-4 border-t">
      {previewUrl && (
        <div className="relative inline-block max-w-xs">
          <img src={previewUrl} alt="Preview" className="max-w-full rounded-lg" />
          <button
            type="button"
            onClick={handleClearImage}
            className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1"
          >
            ×
          </button>
        </div>
      )}

      <div className="flex gap-2">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <label className="px-4 py-2 border rounded-lg cursor-pointer hover:bg-gray-50 flex items-center gap-2">
          📷
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            disabled={isLoading}
            className="hidden"
          />
        </label>

        <button
          type="submit"
          disabled={isLoading || !message.trim()}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 flex items-center gap-2"
        >
          {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
        </button>
      </div>
    </form>
  )
}
