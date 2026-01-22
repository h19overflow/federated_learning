import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { cn } from "@/lib/utils";
import type { Components } from "react-markdown";
import {
  Citation,
  CitationHoverCard,
} from "@/components/chat/CitationRenderer";

interface MarkdownProps {
  content: string;
  className?: string;
  citations?: Citation[];
}

const createComponents = (citations?: Citation[]): Components => ({
  // Headings
  h1: ({ children }) => (
    <h1 className="text-lg font-bold text-[hsl(172_43%_15%)] mt-4 mb-2 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-base font-semibold text-[hsl(172_43%_18%)] mt-3 mb-1.5 first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-[hsl(172_43%_20%)] mt-2 mb-1 first:mt-0">
      {children}
    </h3>
  ),
  // Paragraphs
  p: ({ children }) => (
    <p className="text-sm leading-relaxed mb-2 last:mb-0">{children}</p>
  ),
  // Lists
  ul: ({ children }) => (
    <ul className="list-disc list-inside space-y-1 mb-2 text-sm pl-1">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-inside space-y-1 mb-2 text-sm pl-1">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="text-sm leading-relaxed">{children}</li>,
  // Emphasis
  strong: ({ children }) => (
    <strong className="font-semibold text-[hsl(172_43%_18%)]">
      {children}
    </strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,
  // Code
  code: ({ className, children, ...props }) => {
    const isInline = !className;
    if (isInline) {
      return (
        <code
          className="px-1.5 py-0.5 bg-[hsl(172_30%_92%)] text-[hsl(172_63%_25%)] rounded text-xs font-mono"
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        className={cn(
          "block p-3 bg-[hsl(220_15%_12%)] text-[hsl(0_0%_90%)] rounded-lg text-xs font-mono overflow-x-auto my-2",
          className,
        )}
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: ({ children }) => (
    <pre className="bg-[hsl(220_15%_12%)] rounded-lg overflow-hidden my-2">
      {children}
    </pre>
  ),
  // Blockquotes
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-[hsl(172_63%_35%)] pl-3 italic text-[hsl(215_15%_45%)] my-2 text-sm">
      {children}
    </blockquote>
  ),
  // Tables
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-[hsl(172_30%_94%)]">{children}</thead>
  ),
  th: ({ children }) => (
    <th className="px-2 py-1.5 text-left font-semibold text-[hsl(172_43%_18%)] border-b border-[hsl(172_30%_88%)]">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-2 py-1.5 border-b border-[hsl(210_15%_92%)]">
      {children}
    </td>
  ),
  // Links
  a: ({ href, children }) => {
    if (href?.startsWith("citation:")) {
      const id = href.split(":")[1];
      const citation = citations?.find((c) => c.id === id);

      const reference = (
        <sup className="text-[10px] font-bold text-[hsl(172_63%_30%)] ml-0.5 px-0.5 cursor-help select-none hover:text-[hsl(172_63%_20%)] transition-colors">
          [{id}]
        </sup>
      );

      if (citation) {
        return (
          <CitationHoverCard citation={citation}>{reference}</CitationHoverCard>
        );
      }

      return reference;
    }
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-[hsl(172_63%_30%)] hover:text-[hsl(172_63%_22%)] underline underline-offset-2"
      >
        {children}
      </a>
    );
  },
  // Horizontal rule
  hr: () => <hr className="border-t border-[hsl(210_15%_90%)] my-3" />,
});

export const Markdown = ({ content, className, citations }: MarkdownProps) => {
  if (!content) {
    return null;
  }

  // Memoize components to avoid unnecessary re-renders when content changes but citations don't
  const memoizedComponents = React.useMemo(
    () => createComponents(citations),
    [citations],
  );

  return (
    <div className={cn("prose prose-sm max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
        components={memoizedComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
