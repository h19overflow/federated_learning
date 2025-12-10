import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatContextType {
  messages: Message[];
  sessionId: string;
  isLoading: boolean;
  apiUrl: string;
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  setSessionId: (id: string) => void;
  setIsLoading: (loading: boolean) => void;
  setMessages: (messages: Message[]) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

interface ChatProviderProps {
  children: ReactNode;
  apiUrl?: string;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({
  children,
  apiUrl = 'http://127.0.0.1:8001',
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const value: ChatContextType = {
    messages,
    sessionId,
    isLoading,
    apiUrl,
    addMessage,
    clearMessages,
    setSessionId,
    setIsLoading,
    setMessages,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

export const useChatContext = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within ChatProvider');
  }
  return context;
};
