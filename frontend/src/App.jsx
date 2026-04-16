import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, Send, UploadCloud, X, Search, Brain, Sparkles, 
  Database, Trash2, Menu, FileText
} from 'lucide-react';
import './App.css';

// The base URL for the FastAPI backend
// In development, we use Render backend. In production, use VITE_API_BASE_URL (set during build).
// Example: VITE_API_BASE_URL=https://your-backend-domain.com npm run build
const API_BASE = 'https://rag-chatbot-6.onrender.com';

function App() {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [currentFilename, setCurrentFilename] = useState(null);
  
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toasts, setToasts] = useState([]);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Load active sessions on mount
  useEffect(() => {
    fetchSessions();
  }, []);

  const showToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3500);
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (err) {
      console.error(err);
      showToast("Failed to fetch sessions", "error");
    }
  };

  // --- Handlers ---

  const handleFileDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.pdf')) {
        setSelectedFile(file);
      } else {
        showToast("Only PDF files are accepted", "error");
      }
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const processFile = async () => {
    if (!selectedFile) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      showToast(`✅ ${data.message}`, "success");
      setCurrentSessionId(data.session_id);
      setCurrentFilename(data.filename);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      
      setMessages([]); // clear chat
      fetchSessions();
      
    } catch (err) {
      showToast(`❌ ${err.message}`, "error");
    } finally {
      setIsUploading(false);
    }
  };

  const loadSession = async (sessionId, filename) => {
    setCurrentSessionId(sessionId);
    setCurrentFilename(filename);
    if (window.innerWidth <= 768) setSidebarOpen(false);

    try {
      const res = await fetch(`${API_BASE}/history/${sessionId}`);
      const data = await res.json();
      
      const loadedMessages = [];
      data.history.forEach(entry => {
        loadedMessages.push({ role: 'user', content: entry.question });
        loadedMessages.push({ 
          role: 'assistant', 
          content: entry.answer,
          expandedQuery: entry.expanded_query,
          sourceChunks: entry.chunks_used // (simplified for history, usually backend would return actual chunks back but we'll adapt)
        });
      });
      setMessages(loadedMessages);
    } catch (err) {
      showToast("Failed to load chat history", "error");
    }
  };

  const deleteSession = async (e, sessionId) => {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE}/session/${sessionId}`, { method: 'DELETE' });
      showToast("Session deleted", "info");
      
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setCurrentFilename(null);
        setMessages([]);
      }
      fetchSessions();
    } catch (err) {
      showToast("Failed to delete session", "error");
    }
  };

  const sendQuestion = async () => {
    if (!question.trim() || !currentSessionId) return;

    const userQ = question.trim();
    setMessages(prev => [...prev, { role: 'user', content: userQ }]);
    setQuestion('');
    setIsTyping(true);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId, question: userQ })
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Failed to get answer");

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        expandedQuery: data.expanded_query,
        sourceChunks: data.source_chunks
      }]);
      
      fetchSessions(); // update Q&A count
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Error: ${err.message}` }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  };

  // --- Render Helpers ---
  const formatText = (text) => {
    if (!text) return null;
    return text.split('\n').map((line, i) => (
      <span key={i}>
        {line.split(/(\*\*.*?\*\*|\*.*?\*|`.*?`)/g).map((part, j) => {
          if (part.startsWith('**') && part.endsWith('**')) return <strong key={j}>{part.slice(2, -2)}</strong>;
          if (part.startsWith('*') && part.endsWith('*')) return <em key={j}>{part.slice(1, -1)}</em>;
          if (part.startsWith('`') && part.endsWith('`')) {
            return <code key={j} style={{ background: 'rgba(108,92,231,0.15)', padding: '2px 6px', borderRadius: '4px', fontSize: '12px' }}>{part.slice(1, -1)}</code>;
          }
          return part;
        })}
        <br />
      </span>
    ));
  };


  return (
    <div className="app-container">
      {/* ─── Sidebar ─────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <Bot className="logo-icon" size={28} color="var(--accent-light)" />
            <h1>RAG Chatbot</h1>
          </div>
          <p className="tagline">AI-Powered Document Q&A</p>
        </div>

        {/* Upload Section */}
        <div className="upload-section">
          <h2 className="section-title">📄 Upload PDF</h2>
          <div 
            className="drop-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => !selectedFile && fileInputRef.current?.click()}
          >
            {selectedFile ? (
              <div className="file-selected">
                <FileText className="file-icon" size={20} />
                <span className="file-name">{selectedFile.name}</span>
                <button className="icon-btn" onClick={(e) => { e.stopPropagation(); setSelectedFile(null); fileInputRef.current.value = ""; }}>
                  <X size={16} />
                </button>
              </div>
            ) : (
              <div className="drop-zone-content">
                <UploadCloud size={32} color="var(--text-secondary)" style={{margin: '0 auto 8px auto'}} />
                <p>Drag & drop your PDF here</p>
                <span className="drop-divider">or</span>
                <button className="browse-btn" onClick={() => fileInputRef.current?.click()}>Browse Files</button>
                <input type="file" ref={fileInputRef} accept=".pdf" hidden onChange={handleFileSelect} />
              </div>
            )}
          </div>
          <button 
            className="btn btn-primary" 
            disabled={!selectedFile || isUploading}
            onClick={processFile}
          >
            {isUploading ? <><div className="btn-loader"></div> Processing...</> : "🚀 Process PDF"}
          </button>
        </div>

        {/* Sessions Section */}
        <div className="sessions-section">
          <h2 className="section-title">📚 Sessions</h2>
          <div className="session-list">
            {sessions.length === 0 ? (
              <p className="empty-state">No sessions yet. Upload a PDF to start.</p>
            ) : (
              sessions.map(s => (
                <div 
                  key={s.session_id} 
                  className={`session-card ${s.session_id === currentSessionId ? 'active' : ''}`}
                  onClick={() => loadSession(s.session_id, s.filename)}
                >
                  <div className="session-card-header">
                    <span className="session-card-name">📄 {s.filename}</span>
                    <button className="icon-btn" onClick={(e) => deleteSession(e, s.session_id)}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <div className="session-card-meta">
                    <span>📊 {s.total_chunks} chunks</span>
                    <span>💬 {s.questions_asked} Q&A</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="tech-badge">Hybrid Search: BM25 + FAISS</div>
          <div className="tech-badge">Groq · Llama 3.1</div>
          <div className="tech-badge">React + Vite</div>
        </div>
      </aside>

      {/* ─── Main Content ────────────────────────────────────── */}
      <main className="main-content">
        <header className="top-bar">
          <button className="menu-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <Menu size={20} />
          </button>
          <div className="session-info">
            <span className={`status-dot ${currentSessionId ? 'active' : 'inactive'}`}></span>
            <span>{currentFilename ? `📄 ${currentFilename}` : 'No document loaded'}</span>
          </div>
          <div className="pipeline-badges">
            <span className="badge">Pre-retrieval</span>
            <span className="badge">Hybrid Search</span>
            <span className="badge">Post-retrieval</span>
          </div>
        </header>

        <div className="chat-area">
          {!currentSessionId ? (
            <div className="welcome-message">
              <div className="welcome-icon">💬</div>
              <h2>Welcome to PDF RAG Chatbot</h2>
              <p>Upload a PDF document and start asking questions.</p>
              <div className="feature-grid">
                <div className="feature-card">
                  <div className="icon"><Search size={24}/></div>
                  <h3>Hybrid Search</h3>
                  <p>BM25 keyword + FAISS semantic search combined with Reciprocal Rank Fusion</p>
                </div>
                <div className="feature-card">
                  <div className="icon"><Brain size={24}/></div>
                  <h3>Pre-retrieval</h3>
                  <p>Query expansion rewrites your question for better document search</p>
                </div>
                <div className="feature-card">
                  <div className="icon"><Sparkles size={24}/></div>
                  <h3>Post-retrieval</h3>
                  <p>AI synthesizes the best answer from retrieved chunks</p>
                </div>
                <div className="feature-card">
                  <div className="icon"><Database size={24}/></div>
                  <h3>Sessions</h3>
                  <p>Upload multiple PDFs, each with its own session and chat history</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-bubble">{formatText(msg.content)}</div>
                  {msg.role === 'assistant' && msg.expandedQuery && msg.expandedQuery !== messages[idx-1]?.content && (
                    <div className="expanded-query">
                      <strong>🔍 Expanded Query:</strong> {msg.expandedQuery}
                    </div>
                  )}
                  {msg.role === 'assistant' && Array.isArray(msg.sourceChunks) && msg.sourceChunks.length > 0 && (
                    <details style={{ marginTop: '8px' }}>
                      <summary className="source-toggle" style={{ listStyle: 'none' }}>
                        📄 {msg.sourceChunks.length} source chunks ▾
                      </summary>
                      <div className="source-chunks" style={{ display: 'block' }}>
                        {msg.sourceChunks.map((c, i) => (
                          <div key={i} className="source-chunk">
                            <div className="source-chunk-header">
                              <span>Chunk {i + 1}</span>
                              <span className="source-tag">{c.source} · {c.score}</span>
                            </div>
                            {c.text}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              ))}
              {isTyping && (
                <div className="message assistant">
                  <div className="typing-indicator">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              id="questionInput"
              placeholder="Ask a question about your document..."
              rows={1}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!currentSessionId || isTyping}
              style={{ height: Math.min(question.split('\n').length * 21 + 16, 120) + 'px' }}
            />
            <button 
              className="send-btn" 
              onClick={sendQuestion}
              disabled={!currentSessionId || isTyping || !question.trim()}
            >
              <Send size={18} />
            </button>
          </div>
          <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
        </div>
      </main>

      {/* Toast Notifications */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>{t.message}</div>
        ))}
      </div>
    </div>
  );
}

export default App;
