/**
 * Katherine Client - Markdown Renderer
 * Renders AI responses with proper markdown formatting.
 */

import { memo } from 'react'
import type { ReactNode, CSSProperties } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import type { Components } from 'react-markdown'

// remarkPlugins array - stable reference to avoid re-renders
const remarkPlugins = [remarkGfm]

interface MarkdownRendererProps {
  content: string
  className?: string
}

// ============================================================================
// Quote Highlighting
// ============================================================================

// All quote pairs: [opening, closing]
const QUOTE_PAIRS: [string, string][] = [
  ["\u201C", "\u201D"],  // " " Typographic double quotes
  ["\u2018", "\u2019"],  // ' ' Typographic single quotes  
  ["\u201E", "\u201D"],  // „ " Polish/German lower-upper
  ["\u201A", "\u2019"],  // ‚ ' Polish/German single lower-upper
  ["\u00AB", "\u00BB"],  // « » French guillemets
  ["\u00BB", "\u00AB"],  // » « Reversed guillemets (German style)
  ["\u300C", "\u300D"],  // 「 」 Japanese corner brackets
  ["\u300E", "\u300F"],  // 『 』 Japanese double corner brackets
  ["\u300A", "\u300B"],  // 《 》 Chinese double angle brackets
  ["\u3008", "\u3009"],  // 〈 〉 Chinese single angle brackets
]

// Build regex pattern for all quote types
// Matches: "text", 'text', „text", «text», etc. and simple "text" and 'text'
function buildQuotePattern(): RegExp {
  const patterns: string[] = []
  
  // Add paired quotes
  for (const [open, close] of QUOTE_PAIRS) {
    patterns.push(`${escapeRegex(open)}[^${escapeRegex(close)}]*${escapeRegex(close)}`)
  }
  
  // Add simple straight quotes (must handle nested content carefully)
  patterns.push(`"[^"]*"`)  // Simple double quotes
  patterns.push(`'[^']*'`)  // Simple single quotes (be careful with apostrophes)
  
  return new RegExp(`(${patterns.join('|')})`, 'g')
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const QUOTE_REGEX = buildQuotePattern()

/**
 * Highlights quoted text within a string, returning React elements
 */
function highlightQuotes(text: string, keyPrefix: string = ''): ReactNode[] {
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let matchIndex = 0
  
  // Reset regex state
  QUOTE_REGEX.lastIndex = 0
  
  while ((match = QUOTE_REGEX.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }
    
    // Add the highlighted quoted text
    parts.push(
      <span key={`${keyPrefix}-q-${matchIndex}`} className="text-amber-300/90">
        {match[0]}
      </span>
    )
    
    lastIndex = match.index + match[0].length
    matchIndex++
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }
  
  return parts.length > 0 ? parts : [text]
}

/**
 * Process children to highlight quotes in text nodes
 */
function processChildren(children: ReactNode, keyPrefix: string = ''): ReactNode {
  if (typeof children === 'string') {
    return highlightQuotes(children, keyPrefix)
  }
  
  if (Array.isArray(children)) {
    return children.map((child, i) => processChildren(child, `${keyPrefix}-${i}`))
  }
  
  return children
}

// Stable components object defined outside component to prevent re-renders
const markdownComponents: Components = {
  // Code blocks with syntax highlighting
  code({ node, className: codeClassName, children, ...props }) {
    const match = /language-(\w+)/.exec(codeClassName || '')
    const isInline = !match && !String(children).includes('\n')
    
    if (isInline) {
      return (
        <code
          className="px-1.5 py-0.5 bg-void-700/50 text-ember-300 rounded text-[13px] font-mono"
          {...props}
        >
          {children}
        </code>
      )
    }
    
    return (
      <SyntaxHighlighter
        style={oneDark as { [key: string]: CSSProperties }}
        language={match?.[1] || 'text'}
        PreTag="div"
        customStyle={{
          margin: '0.75rem 0',
          borderRadius: '0.5rem',
          fontSize: '13px',
          background: 'rgba(17, 17, 27, 0.8)',
        }}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    )
  },

  // Paragraphs
  p({ children }) {
    return <p className="mb-3 last:mb-0">{processChildren(children, 'p')}</p>
  },

  // Headings
  h1({ children }) {
    return <h1 className="text-lg font-semibold text-void-100 mt-4 mb-2 first:mt-0">{processChildren(children, 'h1')}</h1>
  },
  h2({ children }) {
    return <h2 className="text-base font-semibold text-void-100 mt-3 mb-2 first:mt-0">{processChildren(children, 'h2')}</h2>
  },
  h3({ children }) {
    return <h3 className="text-sm font-semibold text-void-100 mt-3 mb-1.5 first:mt-0">{processChildren(children, 'h3')}</h3>
  },
  h4({ children }) {
    return <h4 className="text-sm font-medium text-void-200 mt-2 mb-1 first:mt-0">{processChildren(children, 'h4')}</h4>
  },

  // Lists
  ul({ children }) {
    return <ul className="list-disc list-outside ml-5 mb-3 space-y-1 last:mb-0">{children}</ul>
  },
  ol({ children }) {
    return <ol className="list-decimal list-outside ml-5 mb-3 space-y-1 last:mb-0">{children}</ol>
  },
  li({ children }) {
    return <li className="text-void-200 leading-relaxed">{processChildren(children, 'li')}</li>
  },

  // Blockquotes
  blockquote({ children }) {
    return (
      <blockquote className="border-l-2 border-ember-500/50 pl-4 my-3 text-void-300 italic">
        {processChildren(children, 'bq')}
      </blockquote>
    )
  },

  // Links
  a({ href, children }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-aether-400 hover:text-aether-300 underline underline-offset-2 transition-colors"
      >
        {children}
      </a>
    )
  },

  // Strong/Bold
  strong({ children }) {
    return <strong className="font-semibold text-void-100">{children}</strong>
  },

  // Emphasis/Italic
  em({ children }) {
    return <em className="italic text-void-200">{children}</em>
  },

  // Horizontal rule
  hr() {
    return <hr className="my-4 border-void-700/50" />
  },

  // Tables (GFM)
  table({ children }) {
    return (
      <div className="overflow-x-auto my-3">
        <table className="min-w-full border-collapse text-sm">{children}</table>
      </div>
    )
  },
  thead({ children }) {
    return <thead className="bg-void-800/50">{children}</thead>
  },
  tbody({ children }) {
    return <tbody className="divide-y divide-void-700/30">{children}</tbody>
  },
  tr({ children }) {
    return <tr className="border-b border-void-700/30">{children}</tr>
  },
  th({ children }) {
    return (
      <th className="px-3 py-2 text-left text-xs font-semibold text-void-300 uppercase tracking-wider">
        {processChildren(children, 'th')}
      </th>
    )
  },
  td({ children }) {
    return <td className="px-3 py-2 text-void-200">{processChildren(children, 'td')}</td>
  },

  // Strikethrough (GFM)
  del({ children }) {
    return <del className="text-void-500 line-through">{children}</del>
  },
}

function MarkdownRendererInner({ content, className = '' }: MarkdownRendererProps) {
  return (
    <div className={`markdown-content text-[15px] leading-relaxed text-void-200 ${className}`}>
      <ReactMarkdown remarkPlugins={remarkPlugins} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

// Memoize the entire component to prevent re-renders when parent updates
const MarkdownRenderer = memo(MarkdownRendererInner)
export default MarkdownRenderer
