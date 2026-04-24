export interface AgentResponseImage {
  vector_id: string
  image_name: string
  image_path: string
  base64_image: string
}

export interface SessionHandle {
  session_id: string
  created_at: string
}

export interface AgentModel {
  name: string
  display_name: string
}

export interface AgentResponse {
  message: string
  session: SessionHandle
  images_list: AgentResponseImage[]
}

export interface MessageRequest {
  message: string
  user_image_id?: string | null
  model_name?: string
  session_id?: string | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  images?: AgentResponseImage[]
  userImagePreviewUrl?: string
  timestamp: Date
}
