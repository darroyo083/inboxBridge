import { useState } from "react";
import "./styles.css";

const GITHUB_URL = "https://github.com/darroyo083/inboxbridge";

type Operation = {
  number: string;
  title: string;
  description: string;
};

const operations: Operation[] = [
  { number: "01", title: "RECEIVE", description: "Gmail Push and Pub/Sub surface new Primary inbox mail." },
  { number: "02", title: "UNDERSTAND", description: "Thread history and PDF or document context are grounded together." },
  { number: "03", title: "ASK", description: "InboxBridge summarizes in Spanish and waits for the user's intent." },
  { number: "04", title: "DRAFT", description: "Reply, new mail, or forwarding instructions become a German draft." },
  { number: "05", title: "REVIEW", description: "The final draft and Spanish translation are shown side by side." },
  { number: "06", title: "CONFIRM", description: "Explicit confirmation gates the send, then Gmail delivery is verified." },
];

const capabilityGroups = [
  {
    label: "UNDERSTAND",
    module: "MODULE.01",
    items: ["Thread summaries and Q&A", "PDF and document grounding", "Attachment-aware context"],
  },
  {
    label: "WRITE",
    module: "MODULE.02",
    items: ["Replies, new mail, and forwards", "Conversational draft editing", "German final email output"],
  },
  {
    label: "MULTILINGUAL",
    module: "MODULE.03",
    items: ["Spanish incoming summaries", "Spanish review translation", "German drafting from Spanish"],
  },
  {
    label: "SAFE DELIVERY",
    module: "MODULE.04",
    items: ["Explicit confirmation required", "Duplicate-send protections", "Gmail reconciliation after send"],
  },
];

function ExternalArrow() {
  return <span aria-hidden="true" className="external-arrow">↗</span>;
}

function Header() {
  const [isOpen, setIsOpen] = useState(false);

  const closeMenu = () => setIsOpen(false);

  return (
    <header className="site-header">
      <nav className="nav-shell" aria-label="Primary navigation">
        <a className="brand-mark" href="#top" onClick={closeMenu}>InboxBridge</a>
        <button
          className="menu-toggle"
          type="button"
          aria-expanded={isOpen}
          aria-controls="site-navigation"
          onClick={() => setIsOpen((open) => !open)}
        >
          <span className="sr-only">Toggle navigation</span>
          <span aria-hidden="true">{isOpen ? "CLOSE" : "MENU"}</span>
        </button>
        <div id="site-navigation" className={`nav-links${isOpen ? " is-open" : ""}`}>
          <a href="#how-it-works" onClick={closeMenu}>How it works</a>
          <a href="#capabilities" onClick={closeMenu}>Capabilities</a>
          <a href="#engineering" onClick={closeMenu}>Engineering</a>
          <a href="#architecture" onClick={closeMenu}>Architecture</a>
        </div>
        <a className="nav-cta" href={GITHUB_URL} target="_blank" rel="noreferrer">
          <span>View on GitHub</span><ExternalArrow />
        </a>
      </nav>
    </header>
  );
}

function FlowDiagram() {
  return (
    <div className="flow-diagram" aria-label="System flow from Gmail through InboxBridge to Telegram">
      <div className="flow-label">SYS.FLOW.01</div>
      <div className="flow-track" aria-hidden="true" />
      <div className="flow-node flow-node-gmail">
        <span className="mono-label">DATA SOURCE</span>
        <strong>Gmail API</strong>
      </div>
      <div className="flow-node flow-node-core">
        <span className="mono-label">PROCESSOR</span>
        <strong>InboxBridge Core</strong>
      </div>
      <div className="flow-node flow-node-telegram">
        <span className="mono-label">CLIENT INTERFACE</span>
        <strong>Telegram Bot API</strong>
      </div>
    </div>
  );
}

function Hero() {
  return (
    <section className="hero page-shell" id="top">
      <div className="hero-copy">
        <div className="status-badge"><span className="status-dot" />Work in Progress</div>
        <h1>Your inbox,<br />conversational.</h1>
        <p>Read, understand and respond to Gmail from Telegram, with multilingual AI, attachment-aware context and explicit confirmation before anything is sent.</p>
        <div className="hero-actions">
          <a className="button button-primary" href={GITHUB_URL} target="_blank" rel="noreferrer">View on GitHub <ExternalArrow /></a>
          <a className="button button-secondary" href="#how-it-works">See how it works</a>
        </div>
      </div>
      <FlowDiagram />
    </section>
  );
}

function StatusStrip() {
  return (
    <section className="status-strip page-shell" aria-label="Project status">
      <span className="status-glyph" aria-hidden="true">[+]</span>
      <p><strong>STATUS: ACTIVE DEVELOPMENT</strong><span className="status-separator">|</span> Core flows operational. Reliability and attachment hardening in progress.</p>
    </section>
  );
}

function Operations() {
  return (
    <section className="content-section page-shell" id="how-it-works">
      <div className="section-intro">
        <h2>Sequence of Operations</h2>
        <p>A deterministic pipeline for handling communications safely.</p>
      </div>
      <div className="operation-grid">
        {operations.map((operation) => (
          <article className="operation" key={operation.number}>
            <span className="operation-number">{operation.number}</span>
            <h3>{operation.title}</h3>
            <p>{operation.description}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function TraceEntry({ side, label, children, active = false }: { side: "left" | "right"; label: string; children: React.ReactNode; active?: boolean }) {
  return (
    <div className={`trace-entry trace-entry-${side}`}>
      <div className="trace-message">
        <span className={`mono-label${active ? " label-active" : ""}`}>{label}</span>
        <div className={`trace-quote${active ? " trace-quote-active" : ""}`}>{children}</div>
      </div>
      <span className={`trace-node${active ? " trace-node-active" : ""}`} aria-hidden="true" />
      <div className="trace-spacer" />
    </div>
  );
}

function MultilingualTrace() {
  return (
    <section className="content-section trace-section page-shell">
      <div className="section-intro">
        <h2>Multilingual Trace</h2>
        <p>Cross-lingual processing execution log.</p>
      </div>
      <div className="trace-log">
        <div className="trace-line" aria-hidden="true" />
        <TraceEntry side="left" label="SOURCE: GERMAN" active>&quot;Könnten Sie uns das Dokument bis Freitag schicken?&quot;</TraceEntry>
        <TraceEntry side="right" label="SUMMARY: SPANISH">El cliente solicita el documento para el viernes.</TraceEntry>
        <TraceEntry side="left" label="INTENT: USER INPUT" active>&quot;Dile que sí, lo envío el jueves.&quot;</TraceEntry>
        <TraceEntry side="right" label="DRAFT: GERMAN">&quot;Ja, ich werde es Ihnen am Donnerstag schicken.&quot;</TraceEntry>
        <TraceEntry side="left" label="REVIEW: SPANISH">Borrador: &quot;Sí, se lo enviaré el jueves.&quot;</TraceEntry>
        <TraceEntry side="right" label="STATE" active><span className="state-text">Confirmed, sent, verified</span></TraceEntry>
      </div>
    </section>
  );
}

function Capabilities() {
  return (
    <section className="content-section page-shell" id="capabilities">
      <div className="section-intro">
        <h2>Capabilities</h2>
        <p>Core system functions.</p>
      </div>
      <div className="capability-grid">
        {capabilityGroups.map((group) => (
          <article className="capability" key={group.module}>
            <div className="capability-heading"><h3>{group.label}</h3><span>{group.module}</span></div>
            <ul>
              {group.items.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}

const safetyPrinciples = [
  { title: "Trusted State", copy: "Recipients, thread identity, and send metadata come from trusted application state." },
  { title: "No Bypass", copy: "Generated drafts cannot bypass explicit confirmation and completeness checks." },
  { title: "Verification", copy: "Sent messages are reconciled with Gmail and guarded against duplicate sending." },
];

function SafetySection() {
  return (
    <section className="content-section safety-section page-shell" id="engineering">
      <div className="safety-heading">
        <h2>AI proposes.<br />Deterministic systems decide.</h2>
        <p>Safety principles enforced at the architectural level.</p>
      </div>
      <div className="safety-grid">
        {safetyPrinciples.map((principle, index) => (
          <article className="safety-item" key={principle.title}>
            <span className="safety-index">0{index + 1}</span>
            <h3>{principle.title}</h3>
            <p>{principle.copy}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className="content-section architecture-section page-shell" id="architecture">
      <div className="section-intro"><h2>Architecture</h2></div>
      <div className="architecture-diagram">
        <div className="architecture-node">
          <span className="mono-label">INTERFACE</span>
          <strong>Telegram</strong>
          <small>Bot API</small>
        </div>
        <div className="architecture-connector" aria-hidden="true"><span>↔</span></div>
        <div className="architecture-core">
          <span className="core-tag">CORE.ENGINE</span>
          <span className="mono-label label-active">INBOXBRIDGE ASSISTANT</span>
          <div className="core-modules">
            {['Intent', 'LLM', 'Docs', 'Persistence', 'Delivery'].map((module) => <span key={module}>{module}</span>)}
          </div>
        </div>
        <div className="architecture-connector" aria-hidden="true"><span>↔</span></div>
        <div className="architecture-node">
          <span className="mono-label">EXTERNAL</span>
          <strong>Gmail</strong>
          <small>OAuth 2.0</small>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-shell page-shell">
        <div><strong>InboxBridge</strong><p>© 2026. Deterministic systems.</p></div>
        <div><span className="mono-label">WORK IN PROGRESS</span></div>
        <div className="footer-links"><a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub <ExternalArrow /></a><a href="https://github.com/darroyo083" target="_blank" rel="noreferrer">Daniel Arroyo</a></div>
      </div>
    </footer>
  );
}

function App() {
  return <><Header /><main><Hero /><StatusStrip /><Operations /><MultilingualTrace /><Capabilities /><SafetySection /><Architecture /></main><Footer /></>;
}

export default App;
