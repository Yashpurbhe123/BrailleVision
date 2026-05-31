/**
 * ═══════════════════════════════════
 * 📄 FILE 25/42: mobile/store/useAppStore.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Zustand App State Store
 * Manages reactive UI state for camera scanning, scan history, settings,
 * audio playback, accessibility high-contrast toggles, and backend base URL.
 */

import { Platform } from 'react-native';
import { create } from 'zustand';

export interface ScanItem {
  id: number;
  raw_text: string;
  corrected_text: string;
  translated_text: string | null;
  target_language: string | null;
  avg_confidence: number;
  cell_count: number;
  source_type: string;
  correction_method: string | null;
  side_detected: string | null;
  processing_time_ms: number | null;
  created_at: string | null;
  audio_path: string | null;
}

export interface AppStats {
  total_scans: number;
  total_words: number;
  avg_confidence: number;
  scans_today: number;
}

interface AppState {
  // Config & Settings
  apiUrl: string;
  isOnboardingCompleted: boolean;
  targetLanguage: string; // BCP-47 code (e.g. 'en', 'hi')
  ttsEnabled: boolean;
  speechRate: string; // e.g. '+0%', '+25%', '-10%'
  highContrast: boolean;

  // Active Scanner State
  isScanning: boolean;
  isLoading: boolean;
  isUploading: boolean;  // true while upload tab is processing a file
  currentResult: {
    success: boolean;
    rawText: string;
    correctedText: string;
    translatedText: string | null;
    cells: Array<{
      pattern: number[];
      confidence: number;
      x: number;
      y: number;
      bbox: number[];
      dot_count: number;
    }>;
    avgConfidence: number;
    cellCount: number;
    dotCount: number;
    guidance: string;
    sideDetected: string;
    detectionQuality: string;
    correctionMethod: string;
    wasCorrected: boolean;
    annotatedImageBase64: string | null;
    processingTimeMs: number;
  } | null;

  // Audio Playback
  isPlayingAudio: boolean;
  activeAudioKey: string | null; // e.g. 'scan-result-123'

  // History & Statistics
  historyItems: ScanItem[];
  stats: AppStats | null;

  // Actions
  setApiUrl: (url: string) => void;
  completeOnboarding: () => void;
  resetOnboarding: () => void;
  setTargetLanguage: (lang: string) => void;
  setTtsEnabled: (enabled: boolean) => void;
  setSpeechRate: (rate: string) => void;
  setHighContrast: (enabled: boolean) => void;
  
  setScanning: (scanning: boolean) => void;
  setLoading: (loading: boolean) => void;
  setUploading: (uploading: boolean) => void;
  setCurrentResult: (result: AppState['currentResult']) => void;
  clearCurrentResult: () => void;
  
  setPlayingAudio: (playing: boolean, key?: string | null) => void;
  
  // Async operations (State synchronization)
  setHistoryItems: (items: ScanItem[]) => void;
  setStats: (stats: AppStats) => void;
}

import Constants from 'expo-constants';

const getInitialApiUrl = (): string => {
  if (Platform.OS === 'web') {
    return `http://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8000`;
  }
  
  // Dynamically resolve developer's machine IP on physical devices
  const hostUri = Constants.expoConfig?.hostUri || '';
  const host = hostUri.split(':')[0];
  if (host) {
    return `http://${host}:8000`;
  }
  
  return 'http://10.0.2.2:8000'; // Fallback for emulator
};

export const useAppStore = create<AppState>((set) => ({
  // Defaults
  apiUrl: getInitialApiUrl(),
  isOnboardingCompleted: false,
  targetLanguage: 'en',
  ttsEnabled: true,
  speechRate: '+0%',
  highContrast: false,

  isScanning: false,
  isLoading: false,
  isUploading: false,
  currentResult: null,

  isPlayingAudio: false,
  activeAudioKey: null,

  historyItems: [],
  stats: null,

  // Simple setters
  setApiUrl: (url) => set({ apiUrl: url }),
  completeOnboarding: () => set({ isOnboardingCompleted: true }),
  resetOnboarding: () => set({ isOnboardingCompleted: false }),
  setTargetLanguage: (lang) => set({ targetLanguage: lang }),
  setTtsEnabled: (enabled) => set({ ttsEnabled: enabled }),
  setSpeechRate: (rate) => set({ speechRate: rate }),
  setHighContrast: (enabled) => set({ highContrast: enabled }),
  
  setScanning: (scanning) => set({ isScanning: scanning }),
  setLoading: (loading) => set({ isLoading: loading }),
  setUploading: (uploading) => set({ isUploading: uploading }),
  setCurrentResult: (result) => set({ currentResult: result }),
  clearCurrentResult: () => set({ currentResult: null }),
  
  setPlayingAudio: (playing, key = null) => set({ isPlayingAudio: playing, activeAudioKey: key }),

  setHistoryItems: (items) => set({ historyItems: items }),
  setStats: (stats) => set({ stats }),
}));
