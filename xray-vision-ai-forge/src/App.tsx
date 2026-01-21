import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import SavedExperiments from "./pages/SavedExperiments";
import Inference from "./pages/Inference";
import { ChatProvider } from "./context/ChatContext";
import { ChatSidebar } from "./components/chat";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <ChatProvider apiUrl="http://127.0.0.1:8001">
        <div className="flex w-full h-screen overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<Landing />} />
                <Route path="/experiment" element={<Index />} />
                <Route
                  path="/saved-experiments"
                  element={<SavedExperiments />}
                />
                <Route path="/inference" element={<Inference />} />
                {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </BrowserRouter>
          </div>
          <ChatSidebar />
        </div>
      </ChatProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
