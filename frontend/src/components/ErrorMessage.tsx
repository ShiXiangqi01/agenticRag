import { AlertCircle } from 'lucide-react'

interface ErrorMessageProps {
  error: string | null
  onDismiss: () => void
}

export function ErrorMessage({ error, onDismiss }: ErrorMessageProps) {
  if (!error) return null

  return (
    <div className="mx-4 mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg flex items-start gap-2">
      <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
      <div className="flex-1 text-sm">{error}</div>
      <button
        onClick={onDismiss}
        className="text-red-700 hover:text-red-900 font-bold"
      >
        ×
      </button>
    </div>
  )
}
