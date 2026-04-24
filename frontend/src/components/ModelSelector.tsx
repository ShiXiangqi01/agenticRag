import { useChatStore } from '@/store/chat'

export function ModelSelector() {
  const availableModels = useChatStore((state) => state.availableModels)
  const selectedModel = useChatStore((state) => state.selectedModel)
  const setSelectedModel = useChatStore((state) => state.setSelectedModel)

  return (
    <div className="px-4 py-3 border-b bg-gray-50">
      <label className="block text-sm font-medium text-gray-700 mb-2">Select Model</label>
      <select
        value={selectedModel?.name || ''}
        onChange={(e) => {
          const model = availableModels.find((m) => m.name === e.target.value)
          if (model) setSelectedModel(model)
        }}
        className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {availableModels.map((model) => (
          <option key={model.name} value={model.name}>
            {model.display_name}
          </option>
        ))}
      </select>
    </div>
  )
}
