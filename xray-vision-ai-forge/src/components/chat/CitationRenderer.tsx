import React, { useState } from "react";
import { ChevronDown, ChevronUp, BookOpen, FileText, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";

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
      const filename = source.trim().split(/[\\/]/).pop() || source.trim();
      
      let cleanContent = content.trim();
      if (cleanContent.toLowerCase().startsWith('content:')) {
        cleanContent = cleanContent.slice(8).trim();
      }
      
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
      return "";
    });
  }

  // Convert map to array and sort by ID
  Array.from(citationMap.values())
    .sort((a, b) => parseInt(a.id) - parseInt(b.id))
    .forEach(c => citations.push(c));

  // Use markdown link notation for citations to make them interactive
  cleanedContent = cleanedContent.replace(/\[Document\s*(\d+)\]/gi, (match, id) => {
    return `[${id}](citation:${id})`;
  });

  cleanedContent = cleanedContent
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^\s+/, '')
    .replace(/\s+$/, '')
    .trim();

  return { cleanedContent, citations };
};

export const CitationHoverCard: React.FC<{ citation: Citation; children: React.ReactNode }> = ({ citation, children }) => {
  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>
        {children}
      </HoverCardTrigger>
      <HoverCardContent className="w-80 max-w-sm shadow-xl border-[hsl(210_15%_90%)] bg-white p-4 z-[100]">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-[hsl(172_40%_94%)] flex items-center justify-center">
              <FileText className="h-3.5 w-3.5 text-[hsl(172_63%_35%)]" />
            </div>
            <h4 className="text-sm font-semibold text-[hsl(172_63%_30%)] truncate flex-1">
              {citation.source}
            </h4>
            <span className="text-[10px] font-medium text-[hsl(215_15%_55%)] bg-[hsl(210_15%_96%)] px-1.5 py-0.5 rounded">
              p.{citation.page}
            </span>
          </div>
          
          <div className="relative">
            <p className="text-sm text-[hsl(215_15%_40%)] leading-relaxed italic">
              "{citation.content.slice(0, 150)}{citation.content.length > 150 ? '...' : ''}"
            </p>
          </div>
          
          <div className="flex items-center justify-end mt-1 pt-2 border-t border-[hsl(210_15%_96%)]">
            <button 
              className="flex items-center gap-1 text-[10px] font-bold text-[hsl(172_63%_35%)] hover:text-[hsl(172_63%_25%)] transition-colors group"
              onClick={(e) => {
                e.preventDefault();
                console.log("Viewing full citation:", citation.id);
              }}
            >
              View Full Source
              <ExternalLink className="h-2.5 w-2.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </button>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
};

const CitationItem: React.FC<{ citation: Citation; index: number }> = ({ citation, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const preview = citation.content.length > 80 
    ? citation.content.slice(0, 80) + '...' 
    : citation.content;

  return (
    <div 
      className={cn(
        "py-2 transition-colors",
        index > 0 && "border-t border-[hsl(210_15%_94%)]"
      )}
    >
      <div className="flex items-center gap-2">
        <CitationHoverCard citation={citation}>
          <span className="text-[10px] font-bold text-[hsl(172_63%_35%)] bg-[hsl(172_40%_94%)] px-1.5 py-0.5 rounded cursor-help">
            {citation.id}
          </span>
        </CitationHoverCard>
        <div 
          className="flex items-center gap-2 flex-1 cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
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
                "{citation.content}"
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

