-- SQL Schemas for ACOLYTE
-- Mono-user system with hierarchy Task > Session > Message
--
-- Comment about IDs:
-- SQLite generates IDs as 32-character hex (no hyphens)
-- Python must adapt using: id.replace('-', '').lower()
-- Or generate compatible IDs: generate_id()

-- Main conversations table
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,  -- Primary unique ID (hex32)
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    content_summary TEXT,  -- Extracted keywords for search
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    context_used TEXT,     -- RAG context used (JSON)
    metadata TEXT,         -- Additional metadata (JSON)
    task_checkpoint_id TEXT,  -- Associated task (optional)
    related_sessions TEXT DEFAULT '[]',  -- Related sessions (JSON array)
    total_tokens INTEGER DEFAULT 0  -- Token counter
);

CREATE INDEX idx_session_id ON conversations(session_id);
CREATE INDEX idx_timestamp ON conversations(timestamp);
CREATE INDEX idx_role ON conversations(role);
CREATE INDEX idx_task_checkpoint ON conversations(task_checkpoint_id);

-- Tasks table (groups sessions)
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN (
        'IMPLEMENTATION', 'DEBUGGING', 'REFACTORING', 
        'DOCUMENTATION', 'RESEARCH', 'REVIEW'
    )), -- Python must use .upper() when inserting
    status TEXT NOT NULL DEFAULT 'PLANNING' CHECK (status IN (
        'PLANNING', 'IN_PROGRESS', 'COMPLETED'
    )), -- Python must use .upper() when inserting
    progress_percentage REAL DEFAULT 0.0 CHECK (
        progress_percentage >= 0.0 AND progress_percentage <= 100.0
    ),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    
    -- Context and notes
    initial_context TEXT NOT NULL,
    current_context TEXT,
    notes TEXT,
    
    -- Key decisions (JSON array)
    key_decisions TEXT DEFAULT '[]',
    
    -- Keywords for search
    keywords TEXT DEFAULT '[]'
);

CREATE INDEX idx_task_status ON tasks(status);
CREATE INDEX idx_task_created ON tasks(created_at);
CREATE INDEX idx_task_type ON tasks(task_type);

-- Task-session relationship table (many-to-many)
CREATE TABLE IF NOT EXISTS task_sessions (
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (task_id, session_id),
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
);

CREATE INDEX idx_task_sessions_task ON task_sessions(task_id);
CREATE INDEX idx_task_sessions_session ON task_sessions(session_id);

-- Useful view to get task summary
CREATE VIEW IF NOT EXISTS task_summary AS
SELECT 
    t.id,
    t.title,
    t.status,
    t.progress_percentage,
    t.created_at,
    COUNT(DISTINCT ts.session_id) as session_count,
    COUNT(DISTINCT c.session_id) as message_count,
    MAX(c.timestamp) as last_activity
FROM tasks t
LEFT JOIN task_sessions ts ON t.id = ts.task_id
LEFT JOIN conversations c ON ts.session_id = c.session_id
GROUP BY t.id;

-- Triggers to update updated_at
CREATE TRIGGER IF NOT EXISTS update_task_timestamp 
AFTER UPDATE ON tasks
BEGIN
    UPDATE tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Table for optimization system state (Dream)
CREATE TABLE IF NOT EXISTS dream_state (
    id INTEGER PRIMARY KEY DEFAULT 1,  -- Only one row
    fatigue_level REAL DEFAULT 0.0 CHECK (
        fatigue_level >= 0.0 AND fatigue_level <= 10.0
    ),
    last_optimization DATETIME,
    optimization_count INTEGER DEFAULT 0,
    avg_query_time_ms REAL DEFAULT 0.0,
    total_embeddings INTEGER DEFAULT 0,
    metrics TEXT DEFAULT '{}',  -- JSON with detailed metrics
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Only allow one row in dream_state
CREATE TRIGGER IF NOT EXISTS enforce_single_dream_state
BEFORE INSERT ON dream_state
WHEN (SELECT COUNT(*) FROM dream_state) >= 1
BEGIN
    SELECT RAISE(FAIL, 'Only one row can exist in dream_state');
END;

-- Table for insights discovered during optimization
CREATE TABLE IF NOT EXISTS dream_insights (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    session_id TEXT NOT NULL,  -- Optimization session
    insight_type TEXT NOT NULL CHECK (insight_type IN (
        'PATTERN', 'CONNECTION', 'OPTIMIZATION', 
        'ARCHITECTURE', 'BUG_RISK'
    )), -- Python must use .upper() when inserting
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    entities_involved TEXT DEFAULT '[]',  -- JSON array
    code_references TEXT DEFAULT '[]',    -- JSON array
    confidence REAL DEFAULT 0.5 CHECK (
        confidence >= 0.0 AND confidence <= 1.0
    ),
    impact TEXT DEFAULT 'MEDIUM' CHECK (impact IN ('LOW', 'MEDIUM', 'HIGH')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dream_insights_type ON dream_insights(insight_type);
CREATE INDEX idx_dream_insights_session ON dream_insights(session_id);

-- Table for important technical decisions (Decision #13 from audit)
CREATE TABLE IF NOT EXISTS technical_decisions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    task_id TEXT,  -- Optional, may not be associated with task
    session_id TEXT NOT NULL,
    decision_type TEXT NOT NULL CHECK (decision_type IN (
        'ARCHITECTURE', 'LIBRARY', 'PATTERN', 'SECURITY'
    )), -- Python must use .upper() when inserting
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    rationale TEXT NOT NULL,
    alternatives_considered TEXT DEFAULT '[]',  -- JSON array
    impact_level INTEGER NOT NULL CHECK (
        impact_level >= 1 AND impact_level <= 5
    ),
    code_references TEXT DEFAULT '[]',  -- JSON array
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
);

CREATE INDEX idx_technical_decisions_type ON technical_decisions(decision_type);
CREATE INDEX idx_technical_decisions_task ON technical_decisions(task_id);
CREATE INDEX idx_technical_decisions_impact ON technical_decisions(impact_level);

-- ==============================================================
-- NEURAL GRAPH - Code relationship system
-- ==============================================================
-- Decision #21: Maintains structural relationships between files
-- while Weaviate stores embeddings for semantic search

-- Graph nodes (files, functions, classes)
CREATE TABLE IF NOT EXISTS code_graph_nodes (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    node_type TEXT NOT NULL CHECK (node_type IN ('FILE', 'FUNCTION', 'CLASS', 'MODULE')),
    path TEXT NOT NULL,  -- file path or function path (file.py::function_name)
    name TEXT NOT NULL,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}',  -- JSON with additional info
    
    -- Unique index to avoid duplicates
    CONSTRAINT idx_unique_node UNIQUE (node_type, path)
);

CREATE INDEX idx_graph_nodes_type ON code_graph_nodes(node_type);
CREATE INDEX idx_graph_nodes_path ON code_graph_nodes(path);

-- Relationships between nodes
CREATE TABLE IF NOT EXISTS code_graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL CHECK (relation_type IN (
        'IMPORTS', 'CALLS', 'EXTENDS', 'IMPLEMENTS', 
        'USES', 'MODIFIES_TOGETHER', 'BUG_PATTERN'
    )),
    strength REAL DEFAULT 0.5 CHECK (strength >= 0.0 AND strength <= 1.0),
    discovered_by TEXT NOT NULL CHECK (discovered_by IN (
        'GIT_ACTIVITY', 'DREAM_ANALYSIS', 'USER_ACTIVITY', 'STATIC_ANALYSIS'
    )),
    last_reinforced DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}',  -- JSON with details (e.g., commit hashes, line numbers)
    
    PRIMARY KEY (source_id, target_id, relation_type),
    FOREIGN KEY (source_id) REFERENCES code_graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES code_graph_nodes(id) ON DELETE CASCADE
);

CREATE INDEX idx_graph_edges_source ON code_graph_edges(source_id);
CREATE INDEX idx_graph_edges_target ON code_graph_edges(target_id);
CREATE INDEX idx_graph_edges_type ON code_graph_edges(relation_type);
CREATE INDEX idx_graph_edges_strength ON code_graph_edges(strength);

-- Trigger to update last_reinforced
CREATE TRIGGER IF NOT EXISTS update_edge_reinforcement
AFTER UPDATE ON code_graph_edges
WHEN NEW.strength != OLD.strength
BEGIN
    UPDATE code_graph_edges 
    SET last_reinforced = CURRENT_TIMESTAMP 
    WHERE source_id = NEW.source_id 
      AND target_id = NEW.target_id 
      AND relation_type = NEW.relation_type;
END;

-- Graph metrics for Dream (singleton like dream_state)
CREATE TABLE IF NOT EXISTS code_graph_metrics (
    id INTEGER PRIMARY KEY DEFAULT 1,  -- Only one row
    total_nodes INTEGER DEFAULT 0,
    total_edges INTEGER DEFAULT 0,
    avg_connectivity REAL DEFAULT 0.0,
    strongest_clusters TEXT DEFAULT '[]',  -- JSON array of cluster info
    last_analysis DATETIME,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Only allow one row in code_graph_metrics
CREATE TRIGGER IF NOT EXISTS enforce_single_graph_metrics
BEFORE INSERT ON code_graph_metrics
WHEN (SELECT COUNT(*) FROM code_graph_metrics) >= 1
BEGIN
    SELECT RAISE(FAIL, 'Only one row can exist in code_graph_metrics');
END;

-- Trigger to update metrics automatically
CREATE TRIGGER IF NOT EXISTS update_graph_metrics_timestamp
AFTER UPDATE ON code_graph_metrics
BEGIN
    UPDATE code_graph_metrics SET updated_at = CURRENT_TIMESTAMP WHERE id = 1;
END;

-- Useful view to analyze connectivity
CREATE VIEW IF NOT EXISTS node_connectivity AS
SELECT 
    n.id,
    n.name,
    n.node_type,
    COUNT(DISTINCT e_out.target_id) as outgoing_connections,
    COUNT(DISTINCT e_in.source_id) as incoming_connections,
    AVG(COALESCE(e_out.strength, 0)) as avg_outgoing_strength,
    AVG(COALESCE(e_in.strength, 0)) as avg_incoming_strength
FROM code_graph_nodes n
LEFT JOIN code_graph_edges e_out ON n.id = e_out.source_id
LEFT JOIN code_graph_edges e_in ON n.id = e_in.target_id
GROUP BY n.id;

-- ==============================================================
-- RUNTIME STATE - Persistent but minimal runtime state
-- ==============================================================
-- For values that change rarely (device fallback, etc)
-- Only read at startup and written when changed

CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast key lookups
CREATE INDEX idx_runtime_state_key ON runtime_state(key);

-- Trigger to update timestamp
CREATE TRIGGER IF NOT EXISTS update_runtime_state_timestamp
AFTER UPDATE ON runtime_state
BEGIN
    UPDATE runtime_state SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
END;

-- Changes made:
-- 1. session_id in conversations is now PRIMARY KEY (removed redundancy with id field)
-- 2. IDs use lower(hex(...)) for Python compatibility
-- 3. Added optional fields in conversations: task_checkpoint_id, related_sessions, total_tokens
-- 4. Comments about using .upper() for uppercase types
-- 5. NEW: Neural graph tables (code_graph_nodes, code_graph_edges, code_graph_metrics)
-- 6. task_summary view updated to use session_id instead of id
-- 7. NEW: runtime_state table for minimal persistent state (device fallback, etc)
