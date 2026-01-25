/**
 * Blob URL Manager Utility
 * Handles creation, tracking, and cleanup of object URLs
 */

export class BlobUrlManager {
  private urls: Map<string, string> = new Map();

  /**
   * Create and track a blob URL
   */
  createUrl(file: File, key?: string): string {
    const urlKey = key || file.name;
    const url = URL.createObjectURL(file);
    this.urls.set(urlKey, url);
    return url;
  }

  /**
   * Get a tracked URL
   */
  getUrl(key: string): string | undefined {
    return this.urls.get(key);
  }

  /**
   * Revoke a single URL
   */
  revokeUrl(key: string): void {
    const url = this.urls.get(key);
    if (url) {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error(`Failed to revoke URL for key "${key}":`, e);
      }
      this.urls.delete(key);
    }
  }

  /**
   * Revoke all tracked URLs
   */
  revokeAll(): void {
    this.urls.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Failed to revoke URL:", e);
      }
    });
    this.urls.clear();
  }

  /**
   * Get all tracked URLs as a Map
   */
  getAll(): Map<string, string> {
    return new Map(this.urls);
  }

  /**
   * Get count of tracked URLs
   */
  size(): number {
    return this.urls.size;
  }
}

/**
 * Create a new BlobUrlManager instance
 */
export const createBlobUrlManager = (): BlobUrlManager => {
  return new BlobUrlManager();
};
