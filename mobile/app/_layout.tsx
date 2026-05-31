/**
 * ═══════════════════════════════════
 * 📄 FILE 39a/42: mobile/app/_layout.tsx
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Root Stack Navigator Layout
 * Integrates Expo Router Stack Navigation, system status bar,
 * and sets up general accessibility properties.
 */

import React from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useAppStore } from '../store/useAppStore';

export default function RootLayout() {
  const highContrast = useAppStore((state) => state.highContrast);

  return (
    <SafeAreaProvider>
      {/* @ts-ignore */}
      <StatusBar style={highContrast ? 'light' : 'light'} backgroundColor={highContrast ? '#000000' : '#121212'} />
      <Stack
        screenOptions={{
          headerShown: false,
          animation: 'fade',
          contentStyle: {
            backgroundColor: highContrast ? '#000000' : '#121212',
          },
        }}
      >
        <Stack.Screen name="index" />
        <Stack.Screen name="onboarding" />
        <Stack.Screen name="(tabs)" />
      </Stack>
    </SafeAreaProvider>
  );
}
