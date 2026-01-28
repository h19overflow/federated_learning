/**
 * Converts a numeric value to a percentage string with one decimal place.
 *
 * @param value - The numeric value to format (expected to be between 0 and 1)
 * @returns A formatted percentage string (e.g., "85.5%") or "N/A" if value is null/undefined
 *
 * @example
 * formatPercentage(0.855) // Returns "85.5%"
 * formatPercentage(null) // Returns "N/A"
 */
export const formatPercentage = (
  value: number | null | undefined
): string => {
  if (value === null || value === undefined) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
};

/**
 * Formats a duration in minutes to a human-readable string.
 *
 * @param minutes - The duration in minutes to format
 * @returns A formatted duration string (e.g., "5.2 min") or "N/A" if value is null/undefined
 *
 * @example
 * formatDuration(5.234) // Returns "5.2 min"
 * formatDuration(null) // Returns "N/A"
 */
export const formatDuration = (
  minutes: number | null | undefined
): string => {
  if (minutes === null || minutes === undefined) return "N/A";
  return `${minutes.toFixed(1)} min`;
};

/**
 * Formats an ISO date string to a localized short date format.
 *
 * @param dateString - The ISO date string to format
 * @returns A formatted date string (e.g., "Jan 27, 2026") or the original string if parsing fails
 *
 * @example
 * formatDate("2026-01-27T10:30:00Z") // Returns "Jan 27, 2026"
 * formatDate("invalid") // Returns "invalid"
 */
export const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return dateString;
  }
};

/**
 * Truncates long client identifiers for UI display.
 * 
 * @param id - The client identifier string to truncate
 * @returns A truncated identifier string
 * 
 * @example
 * truncateId("client_1234567890") // Returns "client_123456..."
 */
export const truncateId = (id: string): string => {
  if (!id) return "";
  // If it looks like a client_UUID, truncate the UUID part
  if (id.startsWith("client_") && id.length > 15) {
    return `client_${id.substring(7, 13)}...`;
  }
  return id.length > 12 ? `${id.substring(0, 10)}...` : id;
};

