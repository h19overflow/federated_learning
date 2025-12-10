import React from 'react';
import {
  CopilotKit,
  useCopilotAction,
  useCopilotReadable,
} from '@copilotkit/react-core';

interface CopilotContextType {
  apiUrl: string;
}

const CopilotContext = React.createContext<CopilotContextType | undefined>(
  undefined
);

export const CopilotProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const apiUrl =
  'http://127.0.0.1:8001';

  return (
    <CopilotKit runtimeUrl={`${apiUrl}/copilot`}>
      <CopilotContextProvider apiUrl={apiUrl}>
        {children}
      </CopilotContextProvider>
    </CopilotKit>
  );
};

const CopilotContextProvider: React.FC<{
  children: React.ReactNode;
  apiUrl: string;
}> = ({ children, apiUrl }) => {
  // Add copilot actions for querying the chat API
  useCopilotAction({
    name: 'query_chat',
    description:
      'Query the federated pneumonia detection knowledge base. Ask questions about federated learning, pneumonia detection, or related topics.',
    parameters: [
      {
        name: 'query',
        description: 'The question or query to ask the knowledge base',
        type: 'string',
        required: true,
      },
    ],
    handler: async ({ query }: { query: string }) => {
      try {
        const response = await fetch(`${apiUrl}/copilot`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.answer;
      } catch (error) {
        console.error('Error querying chat API:', error);
        return 'Sorry, I encountered an error while processing your query. Please try again.';
      }
    },
  });

  // Make application context available to copilot
  useCopilotReadable({
    description:
      'The current page or section the user is viewing in the application',
    value: 'Federated Pneumonia Detection Application',
  });

  return (
    <CopilotContext.Provider value={{ apiUrl }}>
      {children}
    </CopilotContext.Provider>
  );
};

export const useCopilotContext = (): CopilotContextType => {
  const context = React.useContext(CopilotContext);
  if (!context) {
    throw new Error('useCopilotContext must be used within CopilotProvider');
  }
  return context;
};
