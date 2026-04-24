import axios, { AxiosInstance } from 'axios'
import type { AgentResponse, AgentModel, MessageRequest, SessionHandle } from '@/types/api'

class APIClient {
  private client: AxiosInstance

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }

  async sendMessage(request: MessageRequest): Promise<AgentResponse> {
    const response = await this.client.post<AgentResponse>('/agents/send_message', request)
    return response.data
  }

  async getSessions(): Promise<SessionHandle[]> {
    const response = await this.client.get<SessionHandle[]>('/agents/sessions')
    return response.data
  }

  async getAvailableModels(): Promise<AgentModel[]> {
    const response = await this.client.get<AgentModel[]>('/agents/available_models')
    return response.data
  }

  async uploadUserImage(file: File): Promise<{ user_image_id: string }> {
    const formData = new FormData()
    // Backend expects form field name `image` at /data/user_image/store
    formData.append('image', file)
    const response = await this.client.post<string>('/data/user_image/store', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return { user_image_id: response.data }
  }

  async getUserImage(imageId: string): Promise<{ base64_image: string }> {
    const response = await this.client.get<string>(`/data/user_image/${imageId}`)
    return { base64_image: response.data }
  }

  async deleteSession(sessionId: string): Promise<{ success: boolean }> {
    const response = await this.client.delete<{ success: boolean }>(`/agents/sessions/${sessionId}`)
    return response.data
  }
}

export const apiClient = new APIClient()
