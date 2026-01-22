import { z } from 'zod';

const envSchema = z.object({
  VITE_API_BASE_URL: z.string().url().default('http://127.0.0.1:8001'),
  VITE_WS_BASE_URL: z.string().default('ws://127.0.0.1:8765'),
  VITE_API_TIMEOUT: z.string().transform(Number).pipe(z.number().min(1000).max(600000)).default('300000'),
  VITE_DEBUG: z.string().transform(val => val === 'true').default('false'),
});

export type Env = z.infer<typeof envSchema>;

let validatedEnv: Env | null = null;

export const getEnv = (): Env => {
  if (validatedEnv) {
    return validatedEnv;
  }

  try {
    validatedEnv = envSchema.parse(import.meta.env);
    return validatedEnv;
  } catch (error) {
    console.error('Environment validation failed:', error);
    // Return defaults
    return {
      VITE_API_BASE_URL: 'http://127.0.0.1:8001',
      VITE_WS_BASE_URL: 'ws://127.0.0.1:8765',
      VITE_API_TIMEOUT: 300000,
      VITE_DEBUG: false,
    };
  }
};
