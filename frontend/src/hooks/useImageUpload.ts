import { useCallback, useState } from 'react'
import { apiClient } from '@/services/api'

export function useImageUpload() {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const uploadImage = useCallback(async (file: File): Promise<string | null> => {
    setIsUploading(true)
    setUploadError(null)

    try {
      const result = await apiClient.uploadUserImage(file)
      return result.user_image_id
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload image'
      setUploadError(errorMessage)
      console.error('Error uploading image:', error)
      return null
    } finally {
      setIsUploading(false)
    }
  }, [])

  return { uploadImage, isUploading, uploadError }
}
