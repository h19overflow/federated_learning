import React, { useState, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Library,
  FileText,
  BookOpen,
  Trash2,
  Loader2,
  Search,
  Database,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import api from "@/services/api";
import { KnowledgeBaseDocument } from "@/types/api";

interface KnowledgeBasePanelProps {
  className?: string;
}

export const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({
  className,
}) => {
  const [documents, setDocuments] = useState<KnowledgeBaseDocument[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      setIsLoading(true);
      const response = await api.chat.getKnowledgeBase();
      setDocuments(response.documents);
    } catch (error) {
      console.error("Error fetching knowledge base:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredDocs = documents.filter(
    (doc) =>
      doc.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.paper_id?.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  return (
    <div className={cn("flex flex-col h-full bg-white", className)}>
      <div className="p-4 border-b border-[hsl(210_15%_92%)] bg-[hsl(168_25%_98%)]">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[hsl(215_15%_50%)]" />
          <Input
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 bg-white border-2 border-[hsl(210_15%_90%)] focus:border-[hsl(172_63%_35%)] rounded-xl"
          />
        </div>
      </div>

      <ScrollArea className="flex-1 p-4">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center p-12 gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-[hsl(172_63%_35%)]" />
            <p className="text-sm text-[hsl(215_15%_50%)]">
              Loading collection...
            </p>
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="text-center p-12">
            <div className="w-12 h-12 rounded-2xl bg-[hsl(210_15%_95%)] flex items-center justify-center mx-auto mb-4">
              <Library className="h-6 w-6 text-[hsl(215_15%_50%)]" />
            </div>
            <p className="text-sm font-medium text-[hsl(172_43%_15%)]">
              {searchQuery ? "No matches found" : "Collection is empty"}
            </p>
            <p className="text-xs text-[hsl(215_15%_55%)] mt-1">
              {searchQuery
                ? "Try a different search term"
                : "Embed Arxiv papers to see them here"}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredDocs.map((doc, index) => (
              <div
                key={index}
                className="group p-4 rounded-2xl transition-all border-2 bg-[hsl(210_15%_98%)] border-transparent hover:border-[hsl(210_15%_88%)] hover:bg-white"
              >
                <div className="flex items-start gap-4">
                  <div
                    className={cn(
                      "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm",
                      doc.type === "arxiv"
                        ? "bg-[hsl(172_40%_92%)] text-[hsl(172_63%_25%)]"
                        : "bg-[hsl(210_60%_92%)] text-[hsl(210_60%_40%)]",
                    )}
                  >
                    {doc.type === "arxiv" ? (
                      <BookOpen className="h-5 w-5" />
                    ) : (
                      <FileText className="h-5 w-5" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-semibold text-[hsl(172_43%_15%)] truncate">
                        {doc.display_name}
                      </p>
                      {doc.type === "arxiv" && (
                        <a
                          href={`https://arxiv.org/abs/${doc.paper_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-1 hover:bg-[hsl(172_40%_94%)] rounded-lg text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] transition-all"
                        >
                          <ExternalLink className="h-3.5 w-3.5" />
                        </a>
                      )}
                    </div>

                    <div className="flex items-center gap-3 mt-1.5">
                      <div className="flex items-center gap-1">
                        <Database className="h-3 w-3 text-[hsl(215_15%_60%)]" />
                        <p className="text-[10px] text-[hsl(215_15%_55%)]">
                          {doc.chunk_count} chunks
                        </p>
                      </div>
                      <div className="h-1 w-1 rounded-full bg-[hsl(215_15%_80%)]" />
                      <p className="text-[10px] uppercase font-bold tracking-wider text-[hsl(215_15%_60%)]">
                        {doc.type}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>

      <div className="p-4 border-t border-[hsl(210_15%_92%)] bg-[hsl(168_25%_98%)]">
        <div className="flex items-center justify-between">
          <p className="text-xs text-[hsl(215_15%_50%)]">
            {documents.length} document{documents.length !== 1 ? "s" : ""} in
            knowledge base
          </p>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchDocuments}
            className="h-7 text-[10px] font-semibold text-[hsl(172_63%_30%)] hover:bg-[hsl(172_40%_94%)]"
          >
            Refresh
          </Button>
        </div>
      </div>
    </div>
  );
};
