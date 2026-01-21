import React, { useState } from "react";
import { ChevronDown, ChevronUp, BookOpen, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

export interface Citation {
  id: string;
  source: string;
  fullPath: string;
  page: string;
  content: string;
}

interface CitationRendererProps {
  citations: Citation[];
}

export const parseCitations = (text: string): { cleanedContent: string; citations: Citation[] } => {
  const citations: Citation[] = [];
  const citationMap = new Map<string, Citation>();

  // Multiple patterns to catch different citation formats
  // Pattern 1: [Document N] Source: PATH Page: NUM Content: TEXT (with newlines)
  const patterns = [
    // Standard format with Content: marker
    /\[Document\s*(\d+)\]\s*Source:\s*(.*?)\s*Page:\s*(\d+)\s*Content:\s*([\s\S]*?)(?=\[Document\s*\d+\]|Based on|$)/gi,
    // Format without Content: marker (content follows directly)
    /\[Document\s*(\d+)\]\s*Source:\s*(.*?)\s*Page:\s*(\d+)\s*([\s\S]*?)(?=\[Document\s*\d+\]|Based on|$)/gi,
  ];

  let cleanedContent = text;

  // Try each pattern
  for (const pattern of patterns) {
    cleanedContent = cleanedContent.replace(pattern, (match, id, source, page, content) => {
      // Extract filename from source path (handle both Windows and Unix paths)
      const filename = source.trim().split(/[\\/]/).pop() || source.trim();
      
      // Clean up content - remove "Content:" prefix if present, limit length
      let cleanContent = content.trim();
      if (cleanContent.toLowerCase().startsWith('content:')) {
        cleanContent = cleanContent.slice(8).trim();
      }
      // Truncate very long content for storage (keep first 500 chars)
      const truncatedContent = cleanContent.length > 500 
        ? cleanContent.slice(0, 500) + '...' 
        : cleanContent;
      
      if (!citationMap.has(id)) {
        citationMap.set(id, {
          id,
          source: filename,
          fullPath: source.trim(),
          page: page.trim(),
          content: truncatedContent
        });
      }
      return ""; // Remove the block from the text
    });
  }

  // Convert map to array and sort by ID
  Array.from(citationMap.values())
    .sort((a, b) => parseInt(a.id) - parseInt(b.id))
    .forEach(c => citations.push(c));

  // Clean up any remaining citation markers and convert to superscript notation
  // Replace [Document N] or (Document N) references in the main text
  cleanedContent = cleanedContent.replace(/\[Document\s*(\d+)\]/gi, (match, id) => {
    return `<sup>[${id}]</sup>`;
  });

  // Clean up excessive whitespace and newlines left after removing citations
  cleanedContent = cleanedContent
    .replace(/\n{3,}/g, '\n\n')  // Max 2 newlines
    .replace(/^\s+/, '')         // Trim start
    .replace(/\s+$/, '')         // Trim end
    .trim();

  return { cleanedContent, citations };
};

const CitationItem: React.FC<{ citation: Citation; index: number }> = ({ citation, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Create a short preview (first 80 chars)
  const preview = citation.content.length > 80 
    ? citation.content.slice(0, 80) + '...' 
    : citation.content;

  return (
    <div 
      className={cn(
        "py-2 cursor-pointer transition-colors",
        index > 0 && "border-t border-[hsl(210_15%_94%)]"
      )}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-bold text-[hsl(172_63%_35%)] bg-[hsl(172_40%_94%)] px-1.5 py-0.5 rounded">
          {citation.id}
        </span>
        <FileText className="h-3 w-3 text-[hsl(172_63%_40%)]" />
        <span className="text-xs font-medium text-[hsl(172_43%_20%)] truncate flex-1">
          {citation.source}
        </span>
        <span className="text-[10px] text-[hsl(215_15%_55%)]">
          p.{citation.page}
        </span>
        {isExpanded ? (
          <ChevronUp className="h-3 w-3 text-[hsl(215_15%_55%)]" />
        ) : (
          <ChevronDown className="h-3 w-3 text-[hsl(215_15%_55%)]" />
        )}
      </div>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <div className="mt-2 ml-6 pl-3 border-l-2 border-[hsl(172_63%_80%)]">
              <p className="text-[11px] leading-relaxed text-[hsl(215_15%_40%)] italic">
                "{preview}"
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const CitationRenderer: React.FC<CitationRendererProps> = ({ citations }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (citations.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-[hsl(210_15%_90%)]">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full text-left hover:bg-[hsl(172_40%_96%)] rounded-lg px-2 py-1.5 -mx-2 transition-colors"
      >
        <BookOpen className="h-3.5 w-3.5 text-[hsl(172_63%_35%)]" />
        <span className="text-[11px] font-semibold text-[hsl(172_43%_25%)]">
          {citations.length} source{citations.length > 1 ? 's' : ''} cited
        </span>
        <div className="flex-1" />
        {isExpanded ? (
          <ChevronUp className="h-3.5 w-3.5 text-[hsl(215_15%_55%)]" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-[hsl(215_15%_55%)]" />
        )}
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <div className="mt-2 bg-[hsl(168_25%_98%)] rounded-lg border border-[hsl(168_20%_94%)] px-3 py-1">
              {citations.map((citation, index) => (
                <CitationItem key={citation.id} citation={citation} index={index} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
