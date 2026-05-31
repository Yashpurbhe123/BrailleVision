/**
 * ═══════════════════════════════════
 * 📄 FILE 32/42: mobile/app/(tabs)/history.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Scan History & Usage Analytics
 * Loads SQLite scan records via aiosqlite pagination, computes aggregate metrics
 * (Confidence, Scans, Words), and includes floating speech playback widgets.
 */

import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useFocusEffect } from 'expo-router';
import { Camera, MessageSquare, Award, Trash2, Clock, Languages, Inbox } from 'lucide-react-native';

import { useAppStore, ScanItem, AppStats } from '../../store/useAppStore';
import ApiClient from '../../services/api';
import VoiceService from '../../services/voice';
import { TTSPlayer } from '../../components/TTSPlayer';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from '../../constants/theme';
import { useGenericVoiceCommands } from '../../hooks/useVoiceCommands';

export default function HistoryScreen() {
  const store = useAppStore();
  const [loading, setLoading] = useState<boolean>(false);
  const [refreshing, setRefreshing] = useState<boolean>(false);

  const loadData = async (showLoadingIndicator = true) => {
    if (showLoadingIndicator) setLoading(true);
    try {
      // Fetch scan history and statistics in parallel
      const [history, stats] = await Promise.all([
        ApiClient.fetchHistory(50, 0),
        ApiClient.fetchStats(),
      ]);

      store.setHistoryItems(history);
      store.setStats(stats);
    } catch (error) {
      console.warn("HistoryScreen load error:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Reload history when screen comes into focus
  useFocusEffect(
    React.useCallback(() => {
      VoiceService.speakGuidance("History log opened. Say go to scanner to start scanning, or say stop to stop audio.");
      loadData(true);
      return () => {
        VoiceService.stop();
      };
    }, [])
  );

  // ── Voice commands for blind users ──
  useGenericVoiceCommands({
    onStopAudio: () => VoiceService.stop(),
  });

  const onRefresh = () => {
    setRefreshing(true);
    loadData(false);
  };

  const deleteItem = async (id: number) => {
    try {
      const success = await ApiClient.deleteHistoryItem(id);
      if (success) {
        VoiceService.speakGuidance("Item deleted.");
        loadData(false);
      } else {
        Alert.alert("Error", "Failed to delete history item.");
      }
    } catch (e) {
      console.error(e);
    }
  };

  const confirmClearAll = () => {
    Alert.alert(
      "Clear Scan History",
      "Are you sure you want to permanently delete all scan records?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Clear All", 
          style: "destructive",
          onPress: async () => {
            try {
              await ApiClient.clearHistory();
              VoiceService.speakGuidance("All history cleared.");
              loadData(false);
            } catch (e) {
              console.error(e);
            }
          }
        }
      ]
    );
  };

  const renderStatsHeader = () => {
    const s: AppStats = store.stats || {
      total_scans: 0,
      total_words: 0,
      avg_confidence: 0.0,
      scans_today: 0,
    };

    return (
      <View style={styles.statsContainer}>
        {/* Total Scans Card */}
        <View style={styles.statCard}>
          <View style={styles.iconCircle}>
            <Camera color={COLORS.primary} size={15} strokeWidth={2.5} />
          </View>
          <Text style={styles.statValue}>{s.total_scans}</Text>
          <Text style={styles.statLabel}>Total Scans</Text>
        </View>

        {/* Word Count Card */}
        <View style={styles.statCard}>
          <View style={styles.iconCircle}>
            <MessageSquare color={COLORS.primary} size={15} strokeWidth={2.5} />
          </View>
          <Text style={styles.statValue}>{s.total_words}</Text>
          <Text style={styles.statLabel}>Words Read</Text>
        </View>

        {/* Avg Confidence Card */}
        <View style={styles.statCard}>
          <View style={styles.iconCircle}>
            <Award color={COLORS.primary} size={15} strokeWidth={2.5} />
          </View>
          <Text style={styles.statValue}>{Math.round(s.avg_confidence * 100)}%</Text>
          <Text style={styles.statLabel}>Avg Accuracy</Text>
        </View>
      </View>
    );
  };

  const getConfidenceBorderColor = (conf: number) => {
    if (conf >= 0.8) return COLORS.success;
    if (conf >= 0.5) return COLORS.warning;
    return COLORS.error;
  };

  const renderHistoryItem = ({ item }: { item: ScanItem }) => {
    const outputText = item.translated_text || item.corrected_text || item.raw_text;
    const formattedDate = item.created_at ? new Date(item.created_at).toLocaleDateString() : '';

    return (
      <View style={[styles.itemCard, { borderLeftWidth: 4.5, borderLeftColor: getConfidenceBorderColor(item.avg_confidence) }]}>
        {/* Info header */}
        <View style={styles.itemHeader}>
          <View style={styles.itemMeta}>
            <Clock color={COLORS.textMuted} size={11} strokeWidth={2.5} style={{ marginRight: 4 }} />
            <Text style={styles.itemDate}>{formattedDate}</Text>
            <View style={styles.sourceBadge}>
              <Text style={styles.sourceBadgeText}>{item.source_type.toUpperCase()}</Text>
            </View>
          </View>
          {/* Delete button */}
          <TouchableOpacity
            style={styles.deleteBtn}
            onPress={() => deleteItem(item.id)}
            accessibilityRole="button"
            accessibilityLabel={`Delete history record from date ${formattedDate}`}
          >
            <Trash2 color={COLORS.error} size={14} strokeWidth={2} />
          </TouchableOpacity>
        </View>

        {/* Main Decoded text */}
        <Text style={styles.itemText} numberOfLines={3}>
          {outputText}
        </Text>

        {/* Translation tag */}
        {item.translated_text && (
          <View style={styles.translationTag}>
            <Languages color={COLORS.info} size={11} strokeWidth={2.5} style={{ marginRight: 4 }} />
            <Text style={styles.translationTagText}>
              Translated to: {item.target_language?.toUpperCase()}
            </Text>
          </View>
        )}

        {/* Media controls */}
        <TTSPlayer text={outputText} playbackKey={`history-item-${item.id}`} />
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {/* Top Title Bar */}
      <View style={styles.headerBar}>
        <Text style={styles.title} accessibilityRole="header">History log</Text>
        {store.historyItems.length > 0 && (
          <TouchableOpacity
            style={styles.clearAllBtn}
            onPress={confirmClearAll}
            accessibilityRole="button"
            accessibilityLabel="Clear all history records"
          >
            <Text style={styles.clearAllText}>CLEAR ALL</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Main List */}
      {loading ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={COLORS.primary} />
        </View>
      ) : (
        <FlatList
          data={store.historyItems}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderHistoryItem}
          ListHeaderComponent={renderStatsHeader}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={COLORS.primary}
            />
          }
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <View style={styles.emptyIconCircle}>
                <Inbox color={COLORS.textMuted} size={36} strokeWidth={1.5} />
              </View>
              <Text style={styles.emptyTitle}>History is empty</Text>
              <Text style={styles.emptySubtitle}>Your scans and uploaded documents will appear here once processed.</Text>
            </View>
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 54,
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.md,
  },
  title: {
    fontSize: 28,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    letterSpacing: -0.5,
  },
  clearAllBtn: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(239, 68, 68, 0.05)',
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: COLORS.error,
  },
  clearAllText: {
    color: COLORS.error,
    fontSize: 10,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.8,
  },
  listContent: {
    paddingHorizontal: SPACING.lg,
    paddingBottom: 116, // offsets beautifully above the bottom floating tabs
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: SPACING.md,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'rgba(18, 12, 32, 0.72)',
    borderRadius: 20,
    paddingVertical: 16,
    alignItems: 'center',
    marginHorizontal: 4,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 3,
  },
  iconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(240, 86, 200, 0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
    borderWidth: 1,
    borderColor: 'rgba(240, 86, 200, 0.15)',
  },
  statValue: {
    fontSize: 17,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.text,
  },
  statLabel: {
    fontSize: 9.5,
    color: COLORS.textSecondary,
    marginTop: 4,
    textAlign: 'center',
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  itemCard: {
    backgroundColor: 'rgba(18, 12, 32, 0.80)',
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.04)',
    padding: 16,
    marginBottom: SPACING.md,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 10,
    elevation: 4,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  itemMeta: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  itemDate: {
    color: COLORS.textSecondary,
    fontSize: 11.5,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
  sourceBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    marginLeft: SPACING.sm,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.02)',
  },
  sourceBadgeText: {
    fontSize: 7.5,
    color: COLORS.textSecondary,
    fontWeight: '900',
    letterSpacing: 0.8,
  },
  deleteBtn: {
    padding: 6,
  },
  itemText: {
    fontSize: 15,
    fontWeight: TYPOGRAPHY.weight.bold,
    color: COLORS.text,
    lineHeight: 22,
    marginBottom: 10,
  },
  translationTag: {
    backgroundColor: 'rgba(6, 182, 212, 0.08)',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
    marginBottom: 10,
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(6, 182, 212, 0.25)',
  },
  translationTagText: {
    color: COLORS.accent,
    fontSize: 9.5,
    fontWeight: TYPOGRAPHY.weight.heavy,
    letterSpacing: 0.8,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 90,
  },
  emptyIconCircle: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: 'rgba(255, 255, 255, 0.01)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.md,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.03)',
  },
  emptyTitle: {
    fontSize: 17,
    fontWeight: TYPOGRAPHY.weight.heavy,
    color: COLORS.text,
    marginBottom: 6,
  },
  emptySubtitle: {
    fontSize: 12.5,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
    paddingHorizontal: SPACING.xl,
    fontWeight: TYPOGRAPHY.weight.bold,
  },
});
