/**
 * Input validation and sanitization utilities
 */

// File validation
export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  const MAX_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

  if (!ALLOWED_TYPES.includes(file.type)) {
    return { valid: false, error: 'Only PNG and JPEG images are allowed' };
  }

  if (file.size > MAX_SIZE) {
    return { valid: false, error: 'File size must be less than 10MB' };
  }

  // Check file extension matches MIME type
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (!ext || !['png', 'jpg', 'jpeg'].includes(ext)) {
    return { valid: false, error: 'Invalid file extension' };
  }

  return { valid: true };
};

// Sanitize filename to prevent path traversal
export const sanitizeFilename = (filename: string): string => {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_') // Replace special chars
    .replace(/\.{2,}/g, '.') // Remove multiple dots
    .substring(0, 255); // Limit length
};

// Validate experiment name
export const validateExperimentName = (name: string): { valid: boolean; error?: string } => {
  if (!name || name.trim().length === 0) {
    return { valid: false, error: 'Experiment name is required' };
  }

  if (name.length > 100) {
    return { valid: false, error: 'Experiment name must be less than 100 characters' };
  }

  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return { valid: false, error: 'Experiment name can only contain letters, numbers, hyphens, and underscores' };
  }

  return { valid: true };
};

// Validate URL is safe
export const isSafeUrl = (url: string): boolean => {
  try {
    const parsed = new URL(url);
    return ['http:', 'https:', 'data:'].includes(parsed.protocol);
  } catch {
    return false;
  }
};

// Sanitize data URL
export const sanitizeDataUrl = (dataUrl: string): string | null => {
  if (!dataUrl.startsWith('data:image/')) {
    return null;
  }

  // Validate base64 format
  const parts = dataUrl.split(',');
  if (parts.length !== 2) {
    return null;
  }

  return dataUrl;
};
