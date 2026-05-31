/**
 * ═══════════════════════════════════
 * 📄 FILE 26/42: mobile/services/api.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Backend API Client
 * Wraps networking calls for camera uploads, PDF files, AI translations,
 * TTS audio downloads, and history synchronization.
 */

import { Platform } from 'react-native';
import { useAppStore, ScanItem, AppStats } from '../store/useAppStore';

// Helper to get active API URL from the Zustand store
const getBaseUrl = () => {
  return useAppStore.getState().apiUrl;
};

export const ApiClient = {
  // ------------------------------------------------------------------
  // SCAN OPERATIONS
  // ------------------------------------------------------------------

  async scanImage(
    fileUri: string,
    options: {
      correct?: boolean;
      translateTo?: string | null;
      saveHistory?: boolean;
      saveAnnotated?: boolean;
    } = {}
  ): Promise<any> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/scan/image`;

    const formData = new FormData();
    
    // Formatting the file upload for React Native FormData
    const filename = fileUri.split('/').pop() || 'scan.jpg';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image/jpeg`;
    
    if (Platform.OS === 'web') {
      const response = await fetch(fileUri);
      const blob = await response.blob();
      formData.append('file', blob, filename);
    } else {
      formData.append('file', {
        uri: fileUri,
        name: filename,
        type,
      } as any);
    }

    formData.append('correct', String(options.correct ?? true));
    if (options.translateTo) {
      formData.append('translate_to', options.translateTo);
    }
    formData.append('save_history', String(options.saveHistory ?? true));
    formData.append('save_annotated', String(options.saveAnnotated ?? false));

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorMsg = await response.text();
      throw new Error(errorMsg || `Scan image request failed with status ${response.status}`);
    }

    return response.json();
  },

  async scanLive(fileUri: string): Promise<any> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/scan/live`;

    const formData = new FormData();
    const filename = fileUri.split('/').pop() || 'frame.jpg';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image/jpeg`;

    if (Platform.OS === 'web') {
      const response = await fetch(fileUri);
      const blob = await response.blob();
      formData.append('frame', blob, filename);
    } else {
      formData.append('frame', {
        uri: fileUri,
        name: filename,
        type,
      } as any);
    }

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorMsg = await response.text();
      throw new Error(errorMsg || 'Live frame processing failed');
    }

    return response.json();
  },

  async scanPdf(
    fileUri: string,
    options: {
      correct?: boolean;
      translateTo?: string | null;
      generateAudio?: boolean;
    } = {}
  ): Promise<any> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/scan/pdf`;

    const formData = new FormData();
    const filename = fileUri.split('/').pop() || 'document.pdf';
    
    if (Platform.OS === 'web') {
      const response = await fetch(fileUri);
      const blob = await response.blob();
      formData.append('file', blob, filename);
    } else {
      formData.append('file', {
        uri: fileUri,
        name: filename,
        type: 'application/pdf',
      } as any);
    }

    formData.append('correct', String(options.correct ?? true));
    if (options.translateTo) {
      formData.append('translate_to', options.translateTo);
    }
    formData.append('generate_audio', String(options.generateAudio ?? false));

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorMsg = await response.text();
      throw new Error(errorMsg || 'PDF processing failed');
    }

    return response.json();
  },

  // ------------------------------------------------------------------
  // TRANSLATION OPERATIONS
  // ------------------------------------------------------------------

  async translateText(text: string, targetLang: string): Promise<any> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/translate/`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, target_lang: targetLang }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Translation failed');
    }

    return response.json();
  },

  async fetchLanguages(): Promise<Record<string, string>> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/translate/languages`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch supported languages');
    }
    const data = await response.json();
    return data.languages;
  },

  // ------------------------------------------------------------------
  // TTS (TEXT TO SPEECH) OPERATIONS
  // ------------------------------------------------------------------

  async fetchVoices(): Promise<any[]> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/tts/voices`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch TTS voices');
    }
    const data = await response.json();
    return data.voices;
  },

  /**
   * Fetch binary speech data (base64) from the TTS engine.
   * Useful for downloading audio to local file system.
   */
  async generateSpeechBase64(text: string, lang: string, rate: string = '+0%'): Promise<string> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/tts/speak`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, lang, rate }),
    });

    if (!response.ok) {
      throw new Error('Speech synthesis request failed');
    }

    // Read response stream as array buffer and convert to base64
    const buffer = await response.arrayBuffer();
    return this._arrayBufferToBase64(buffer);
  },

  async generateGuidanceBase64(message: string, lang: string): Promise<string> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/tts/guidance`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, lang }),
    });

    if (!response.ok) {
      throw new Error('Guidance synthesis failed');
    }

    const buffer = await response.arrayBuffer();
    return this._arrayBufferToBase64(buffer);
  },

  // ------------------------------------------------------------------
  // HISTORY CRUD OPERATIONS
  // ------------------------------------------------------------------

  async fetchHistory(limit = 50, offset = 0): Promise<ScanItem[]> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/history/?limit=${limit}&offset=${offset}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to load scan history');
    }

    return response.json();
  },

  async fetchStats(): Promise<AppStats> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/history/stats`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to load dashboard statistics');
    }

    return response.json();
  },

  async deleteHistoryItem(id: number): Promise<boolean> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/history/${id}`;

    const response = await fetch(url, {
      method: 'DELETE',
    });

    return response.ok;
  },

  async clearHistory(): Promise<boolean> {
    const baseUrl = getBaseUrl();
    const url = `${baseUrl}/history/`;

    const response = await fetch(url, {
      method: 'DELETE',
    });

    return response.ok;
  },

  // ------------------------------------------------------------------
  // UTILITIES
  // ------------------------------------------------------------------

  _arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    // Using global btoa helper (available in React Native environment)
    return btoa(binary);
  },
};
export default ApiClient;
