/**
 * ═══════════════════════════════════
 * 📄 FILE 28/42: mobile/hooks/useBrailleScanner.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Custom Scanner Hook
 * Wraps camera frame-capture logic, handles periodic live upload, and triggers
 * speech-based feedback directions.
 */

import { useRef, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import ApiClient from '../services/api';
import VoiceService from '../services/voice';

// Interval between live scan frames (in milliseconds)
const LIVE_SCAN_INTERVAL_MS = 1500;

export const useBrailleScanner = (cameraRef: any) => {
  const isScanning = useAppStore((state) => state.isScanning);
  const isLoading = useAppStore((state) => state.isLoading);
  const currentResult = useAppStore((state) => state.currentResult);

  const scanIntervalId = useRef<any | null>(null);
  const isProcessingFrame = useRef<boolean>(false);
  const lastGuidanceMsg = useRef<string>('');
  const lastGuidanceTime = useRef<number>(0);

  // Trigger audio guidance with basic throttling
  const triggerGuidance = useCallback((msg: string) => {
    if (!msg) return;
    const now = Date.now();
    // Throttle guidance announcements to once every 4 seconds to prevent overlapping chatter
    if (msg !== lastGuidanceMsg.current || now - lastGuidanceTime.current > 4000) {
      lastGuidanceMsg.current = msg;
      lastGuidanceTime.current = now;
      VoiceService.speakGuidance(msg);
    }
  }, []);

  // ------------------------------------------------------------------
  // LIVE PERIODIC FRAME SCANNER
  // ------------------------------------------------------------------

  const startLiveScanner = useCallback(() => {
    if (scanIntervalId.current) return; // already active

    const state = useAppStore.getState();
    state.setScanning(true);
    lastGuidanceMsg.current = '';
    lastGuidanceTime.current = 0;

    logger("Starting live Braille scanner loop...");

    scanIntervalId.current = setInterval(async () => {
      // Pause live scan while the upload tab is actively processing a file
      const currentStore = useAppStore.getState();
      if (currentStore.isUploading) return;

      if (isProcessingFrame.current || !cameraRef.current) return;
      isProcessingFrame.current = true;

      try {
        // Capture frame as lightweight JPEG
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.4,
          skipProcessing: true,
        });

        if (photo?.uri) {
          const result = await ApiClient.scanLive(photo.uri);
          
          if (result.success) {
            // Update store with minimal real-time statistics
            const latestStore = useAppStore.getState();
            latestStore.setCurrentResult({
              success: true,
              rawText: result.raw_text,
              correctedText: result.corrected_text,
              translatedText: null,
              cells: [],
              avgConfidence: result.avg_confidence,
              cellCount: result.cell_count,
              dotCount: result.dot_count,
              guidance: result.guidance,
              sideDetected: result.side_detected,
              detectionQuality: result.detection_quality,
              correctionMethod: 'none',
              wasCorrected: false,
              annotatedImageBase64: null,
              processingTimeMs: result.processing_time_ms,
            });

            // Announce camera placement directions
            triggerGuidance(result.guidance);
          }
        }
      } catch (err) {
        console.warn("useBrailleScanner live frame error:", err);
      } finally {
        isProcessingFrame.current = false;
      }
    }, LIVE_SCAN_INTERVAL_MS);
  }, [cameraRef, triggerGuidance]);

  const stopLiveScanner = useCallback(() => {
    if (scanIntervalId.current) {
      clearInterval(scanIntervalId.current);
      scanIntervalId.current = null;
    }
    const state = useAppStore.getState();
    state.setScanning(false);
    state.setLoading(false);
    isProcessingFrame.current = false;
    VoiceService.stop();
  }, []);

  // ------------------------------------------------------------------
  // FULL HIGH-RESOLUTION SNAPSHOT SCANNER
  // ------------------------------------------------------------------

  const captureFullImage = useCallback(async (options: {
    correct?: boolean;
    translateTo?: string | null;
  } = {}) => {
    if (!cameraRef.current) return;
    
    const initialStore = useAppStore.getState();
    // Stop live scanner temporarily to focus resources
    const wasScanning = initialStore.isScanning;
    if (wasScanning) {
      if (scanIntervalId.current) {
        clearInterval(scanIntervalId.current);
        scanIntervalId.current = null;
      }
      isProcessingFrame.current = false;
    }

    initialStore.setLoading(true);
    VoiceService.speakGuidance("Processing image. Please hold still.");

    try {
      // Capture high-quality photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.95,
        skipProcessing: false,
      });

      if (photo?.uri) {
        const currentStore = useAppStore.getState();
        const translateLang = options.translateTo || (currentStore.targetLanguage !== 'en' ? currentStore.targetLanguage : null);
        
        const result = await ApiClient.scanImage(photo.uri, {
          correct: options.correct ?? true,
          translateTo: translateLang,
          saveHistory: true,
          saveAnnotated: true,
        });

        if (result.success) {
          const successStore = useAppStore.getState();
          successStore.setCurrentResult({
            success: true,
            rawText: result.raw_text,
            correctedText: result.corrected_text,
            translatedText: result.translated_text,
            cells: result.cells || [],
            avgConfidence: result.avg_confidence,
            cellCount: result.cell_count,
            dotCount: result.dot_count,
            guidance: result.guidance,
            sideDetected: result.side_detected,
            detectionQuality: result.detection_quality,
            correctionMethod: result.correction_method,
            wasCorrected: result.was_corrected,
            annotatedImageBase64: result.annotated_image_base64,
            processingTimeMs: result.processing_time_ms,
          });
          
          // Speak the final decoded output aloud
          const outputText = result.translated_text || result.corrected_text || result.raw_text;
          if (outputText) {
            VoiceService.speak(outputText, `scan-${Date.now()}`);
          }
        } else {
          VoiceService.speakGuidance("Detection failed. " + (result.error || "Please reposition camera."));
        }
      }
    } catch (err: any) {
      console.error("useBrailleScanner full capture error:", err);
      VoiceService.speakGuidance("Error processing capture. Please try again.");
    } finally {
      const finalStore = useAppStore.getState();
      finalStore.setLoading(false);
      // Restart live scanning loop if it was active
      if (wasScanning) {
        finalStore.setScanning(false); // reset state to trigger start
        startLiveScanner();
      }
    }
  }, [cameraRef, startLiveScanner]);

  return {
    isScanning,
    isLoading,
    currentResult,
    startLiveScanner,
    stopLiveScanner,
    captureFullImage,
  };
};

function logger(msg: string) {
  console.log(`[useBrailleScanner] ${msg}`);
}
export default useBrailleScanner;
