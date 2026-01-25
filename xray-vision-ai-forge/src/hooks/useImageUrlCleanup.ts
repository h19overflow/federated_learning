/**
 * Image URL Cleanup Hook
 * Manages lifecycle of blob URLs with proper cleanup on unmount
 */

import { useEffect, useRef } from "react";

export const useImageUrlCleanup = () => {
  const urlsRef = useRef<Map<string, string>>(new Map());

  /**
   * Create and track a blob URL
   */
  const createUrl = (file: File, key?: string): string => {
    const urlKey = key || file.name;
    const url = URL.createObjectURL(file);
    urlsRef.current.set(urlKey, url);
    return url;
  };

  /**
   * Revoke a single URL
   */
  const revokeUrl = (key: string): void => {
    const url = urlsRef.current.get(key);
    if (url) {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error(`Failed to revoke URL for key "${key}":`, e);
      }
      urlsRef.current.delete(key);
    }
  };

  /**
   * Revoke all tracked URLs
   */
  const revokeAll = (): void => {
    urlsRef.current.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Failed to revoke URL:", e);
      }
    });
    urlsRef.current.clear();
  };

  /**
   * Update tracked URLs from a Map
   */
  const updateUrls = (urls: Map<string, string>): void => {
    urlsRef.current = new Map(urls);
  };

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      revokeAll();
    };
  }, []);

  return {
    urlsRef,
    createUrl,
    revokeUrl,
    revokeAll,
    updateUrls,
  };
};
