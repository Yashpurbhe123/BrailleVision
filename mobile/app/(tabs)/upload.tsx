/**
 * ═══════════════════════════════════
 * 📄 FILE 31/42: mobile/app/(tabs)/upload.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Photo Library & Document Uploader
 * Integrates Expo ImagePicker and DocumentPicker. Sends media to scan endpoints,
 * renders annotated bounding box images, and binds speech feedback players.
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  AccessibilityInfo,
  Platform,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Image as ImageIcon, AlertCircle, Sparkles } from 'lucide-react-native';

import ApiClient from '../../services/api';
import VoiceService from '../../services/voice';
import { useAppStore } from '../../store/useAppStore';
import { ConfidenceDisplay } from '../../components/ConfidenceDisplay';
import { TTSPlayer } from '../../components/TTSPlayer';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../../constants/theme';
import { useUploadVoiceCommands } from '../../hooks/useVoiceCommands';
import { useFocusEffect } from 'expo-router';

export default function UploadScreen() {
  const store = useAppStore();
  const [loading, setLoading] = useState<boolean>(false);
  const [fileType, setFileType] = useState<'image' | null>(null);
  const [uploadedImageUri, setUploadedImageUri] = useState<string | null>(null);
  
  // Results
  const [imageResult, setImageResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const announceToAccessibility = (text: string) => {
    AccessibilityInfo.announceForAccessibility(text);
  };

  // Announce screen on focus
  useFocusEffect(
    React.useCallback(() => {
      VoiceService.speakGuidance("Upload screen. Say open gallery to upload a photo from your library.");
    }, [])
  );

  // ------------------------------------------------------------------
  // IMAGE PICKER FLOW
  // ------------------------------------------------------------------
  const pickImage = async () => {
    setErrorMsg(null);
    setImageResult(null);
    setFileType(null);
    setUploadedImageUri(null);

    if (Platform.OS !== 'web') {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permissionResult.granted) {
        const msg = "Media library permissions required.";
        setErrorMsg(msg);
        VoiceService.speakGuidance(msg);
        return;
      }
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: (ImagePicker as any).MediaTypeOptions?.Images || 'images',
      quality: 0.9,
    });

    if (result.canceled || !result.assets || result.assets.length === 0) {
      return;
    }

    const uri = result.assets[0].uri;
    setUploadedImageUri(uri);          // ← show image immediately
    setFileType('image');
    setLoading(true);
    store.setUploading(true);  // pause live scanner
    VoiceService.speakGuidance("Uploading Braille image. Please wait.");

    try {
      const scanData = await ApiClient.scanImage(uri, {
        correct: true,
        saveHistory: true,
        saveAnnotated: true,
      });

      if (scanData.success) {
        setImageResult(scanData);
        const finalTxt = scanData.corrected_text || scanData.raw_text;
        VoiceService.speakGuidance(`Upload success. Decoded text: ${finalTxt}`);
        announceToAccessibility(`Upload completed. Confidence score ${Math.round(scanData.avg_confidence * 100)} percent.`);
      } else {
        const err = scanData.error || "Decoding failed.";
        setErrorMsg(err);
        VoiceService.speakGuidance(err);
      }
    } catch (err: any) {
      console.error(err);
      const msg = "Network error. Failed to reach scanning server.";
      setErrorMsg(msg);
      VoiceService.speakGuidance(msg);
    } finally {
      setLoading(false);
      store.setUploading(false);  // resume live scanner
    }
  };

  // ── Voice commands for blind users ──
  useUploadVoiceCommands({
    onPickImage: pickImage,
    onReadResult: () => {
      const text = imageResult?.translated_text || imageResult?.corrected_text || imageResult?.raw_text;
      if (text) VoiceService.speak(text, 'upload-voice');
      else VoiceService.speakGuidance('No result to read yet. Upload an image first.');
    },
    onStopAudio: () => VoiceService.stop(),
  });

  return (
    <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
      <Text style={styles.title} accessibilityRole="header">Upload Braille</Text>
      <Text style={styles.subtitle}>Select a photo from your gallery to decode.</Text>

      {/* Select button */}
      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={styles.pickerBtn}
          onPress={pickImage}
          disabled={loading}
          accessibilityRole="button"
          accessibilityLabel="Upload image from photo library"
        >
          <View style={styles.iconCircle}>
            <ImageIcon color={COLORS.primary} size={24} strokeWidth={2} />
          </View>
          <Text style={styles.btnText}>PHOTO LIBRARY</Text>
        </TouchableOpacity>
      </View>

      {/* Loading state */}
      {loading && !uploadedImageUri && (
        <View style={styles.loaderContainer}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.loaderText}>Processing File. Please wait...</Text>
        </View>
      )}

      {/* Error state */}
      {errorMsg && (
        <View style={styles.errorContainer} accessibilityRole="alert">
          <AlertCircle color={COLORS.error} size={16} strokeWidth={2.5} style={{ marginRight: 6 }} />
          <Text style={styles.errorText}>{errorMsg}</Text>
        </View>
      )}

      {/* IMAGE SCANNING DETAILS */}
      {fileType === 'image' && (uploadedImageUri || imageResult) && (
        <View style={styles.resultCard}>

          {/* ── Uploaded image preview (always shown) ── */}
          {uploadedImageUri && !imageResult?.annotated_image_base64 && (
            <Image
              style={styles.annotatedPreview}
              source={{ uri: uploadedImageUri }}
              resizeMode="contain"
              accessibilityLabel="Uploaded Braille image"
            />
          )}

          {/* ── Annotated image overlay (shown when backend provides it) ── */}
          {imageResult?.annotated_image_base64 && (
            <Image
              style={styles.annotatedPreview}
              source={{ uri: `data:image/jpeg;base64,${imageResult.annotated_image_base64}` }}
              resizeMode="contain"
              accessibilityLabel="Annotated view of detected Braille cells"
            />
          )}

          {/* Loading overlay on the image while processing */}
          {loading && (
            <View style={styles.imageLoadingOverlay}>
              <ActivityIndicator size="large" color={COLORS.primary} />
              <Text style={styles.imageLoadingText}>Analysing Braille...</Text>
            </View>
          )}

          {/* Only show decode results when we have them */}
          {imageResult && (
            <>
              {/* Letter confidence score indicator */}
              <ConfidenceDisplay
                text={imageResult.corrected_text || imageResult.raw_text}
                confidences={imageResult.confidences || []}
                avgConfidence={imageResult.avg_confidence}
              />

              {/* Core Text output */}
              <View style={styles.textDetails}>
                <Text style={styles.sectionHeader}>DECODED OUTPUT</Text>
                <Text style={styles.mainOutputText}>
                  {imageResult.corrected_text || imageResult.raw_text}
                </Text>

                {imageResult.was_corrected && (
                  <View style={styles.correctionBadge}>
                    <Sparkles color={COLORS.primary} size={11} strokeWidth={2.5} style={{ marginRight: 4 }} />
                    <Text style={styles.correctionBadgeText}>
                      AI SPELL-CORRECTED ({imageResult.correction_method})
                    </Text>
                  </View>
                )}
              </View>

              {/* Audio voice player */}
              <TTSPlayer
                text={imageResult.translated_text || imageResult.corrected_text || imageResult.raw_text}
                playbackKey="upload-image"
              />
            </>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: COLORS.background,
    paddingTop: 54,
    paddingBottom: 116, // offsets perfectly above the bottom floating tabs
    paddingHorizontal: SPACING.lg,
  },
  title: {
    fontSize: 28,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    marginBottom: 6,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 13,
    color: COLORS.textSecondary,
    lineHeight: 20,
    marginBottom: SPACING.xl,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: SPACING.xl,
  },
  pickerBtn: {
    width: '100%',
    backgroundColor: 'rgba(240, 86, 200, 0.04)',
    borderRadius: 24,
    paddingVertical: SPACING.lg,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1.8,
    borderColor: 'rgba(240, 86, 200, 0.25)',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 4,
  },
  iconCircle: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
    borderWidth: 1.5,
    borderColor: 'rgba(240, 86, 200, 0.2)',
  },
  iconCircleSecondary: {
    backgroundColor: 'rgba(6, 182, 212, 0.08)',
    borderColor: 'rgba(6, 182, 212, 0.2)',
  },
  btnText: {
    color: COLORS.primary,
    fontSize: 10.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 1.2,
  },
  btnTextSecondary: {
    color: COLORS.accent, // cyan
  },
  loaderContainer: {
    marginVertical: SPACING.xl,
    alignItems: 'center',
  },
  loaderText: {
    color: COLORS.textSecondary,
    fontSize: 13,
    marginTop: SPACING.md,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.5,
  },
  errorContainer: {
    backgroundColor: 'rgba(239, 68, 68, 0.06)',
    borderWidth: 1.8,
    borderColor: COLORS.error,
    borderRadius: 20,
    padding: SPACING.md,
    marginBottom: SPACING.xl,
    flexDirection: 'row',
    alignItems: 'center',
  },
  errorText: {
    color: COLORS.error,
    fontWeight: TYPOGRAPHY.weight.heavy,
    fontSize: 13,
  },
  resultCard: {
    backgroundColor: 'rgba(18, 12, 32, 0.82)', // Frosted glass
    borderRadius: 28,
    borderWidth: 2,
    borderColor: COLORS.border,
    padding: 18,
    width: '100%',
    marginBottom: SPACING.xl,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    elevation: 6,
  },
  annotatedPreview: {
    width: '100%',
    height: 240,
    backgroundColor: '#040209',
    borderRadius: 20,
    marginBottom: SPACING.md,
    borderWidth: 1.5,
    borderColor: 'rgba(6, 182, 212, 0.25)',
  },
  imageLoadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 240,
    borderRadius: 20,
    backgroundColor: 'rgba(4, 2, 9, 0.65)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  imageLoadingText: {
    color: COLORS.primary,
    fontSize: 12,
    fontWeight: TYPOGRAPHY.weight.heavy,
    marginTop: 10,
    letterSpacing: 0.8,
  },
  textDetails: {
    marginVertical: 12,
  },
  sectionHeader: {
    fontSize: 10,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.primary,
    letterSpacing: 1.5,
    marginBottom: 8,
  },
  mainOutputText: {
    fontSize: 19,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    lineHeight: 26,
  },
  correctionBadge: {
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
    marginTop: 10,
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: COLORS.border,
  },
  correctionBadgeText: {
    color: COLORS.primary,
    fontSize: 9.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.8,
  },
  pdfHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1.5,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
    paddingBottom: 12,
    marginBottom: 16,
  },
  pdfHeaderTitle: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.text,
    letterSpacing: 1.0,
  },
  pdfPagesText: {
    fontSize: 10.5,
    color: COLORS.primary,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.5,
  },
  pdfOutputText: {
    fontSize: 15,
    color: COLORS.text,
    lineHeight: 23,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  pageBreakdown: {
    marginTop: 12,
    marginBottom: 12,
    paddingTop: 12,
    borderTopWidth: 1.5,
    borderTopColor: 'rgba(255, 255, 255, 0.06)',
  },
  sectionSubHeader: {
    fontSize: 12,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.textSecondary,
    marginBottom: 10,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  pageItemRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.03)',
  },
  pageItemLabel: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  pageItemConfidence: {
    color: COLORS.success,
    fontSize: 13,
    fontWeight: TYPOGRAPHY.weight.heavy,
  },
});
