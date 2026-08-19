import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent, ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const GITHUB_URL = "https://github.com/darroyo083/inboxbridge";

const navItems = [
  { label: "Home", to: "/", end: true },
  { label: "How it works", to: "/how-it-works" },
  { label: "Capabilities", to: "/capabilities" },
  { label: "Safety", to: "/safety" },
  { label: "Architecture", to: "/architecture" },
];

type PageProps = { children: ReactNode; className?: string };

type RouterContextValue = {
  pathname: string;
  navigate: (to: string) => void;
};

const RouterContext = createContext<RouterContextValue | null>(null);

function normalizePath(pathname: string) {
  const trimmed = pathname.replace(/\/+$/, "");
  return trimmed || "/";
}

function AppRouter({ children }: { children: ReactNode }) {
  const [pathname, setPathname] = useState(() => normalizePath(window.location.pathname));

  useEffect(() => {
    const handlePopState = () => setPathname(normalizePath(window.location.pathname));
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigate = useCallback((to: string) => {
    const nextPath = normalizePath(to);
    const currentPath = normalizePath(window.location.pathname);
    if (nextPath === currentPath) return;
    window.history.pushState({}, "", nextPath);
    setPathname(nextPath);
  }, []);

  const value = useMemo<RouterContextValue>(() => ({ pathname, navigate }), [navigate, pathname]);

  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

function useRoute() {
  const context = useContext(RouterContext);
  if (!context) throw new Error("useRoute must be used inside AppRouter");
  return context;
}

function handleRouteClick(event: MouseEvent<HTMLAnchorElement>, navigate: (to: string) => void, to: string) {
  if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  navigate(to);
}

function RouteLink({ to, children, className, ariaLabel }: { to: string; children: ReactNode; className?: string; ariaLabel?: string }) {
  const { navigate } = useRoute();
  return <a className={className} href={to} aria-label={ariaLabel} onClick={(event) => handleRouteClick(event, navigate, to)}>{children}</a>;
}

function RouteNavLink({ to, children, end = false, className = "" }: { to: string; children: ReactNode; end?: boolean; className?: string }) {
  const { pathname, navigate } = useRoute();
  const isActive = end ? pathname === to : pathname.startsWith(to);
  return <a className={`${className}${isActive ? " is-active" : ""}`} href={to} aria-current={isActive ? "page" : undefined} onClick={(event) => handleRouteClick(event, navigate, to)}>{children}</a>;
}

function LogoMark({ compact = false }: { compact?: boolean }) {
  return (
    <span className={`logo-lockup${compact ? " logo-lockup-compact" : ""}`}>
      <svg className="logo-mark" viewBox="0 0 32 32" aria-hidden="true">
        <path d="M5 23V15a11 11 0 0 1 22 0v8" />
        <path d="M10 23v-7a6 6 0 0 1 12 0v7" />
        <path d="M3 26h26" />
      </svg>
      {!compact && <span className="brand-mark">InboxBridge</span>}
    </span>
  );
}

function ExternalArrow() {
  return <span aria-hidden="true" className="external-arrow">↗</span>;
}

function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const { pathname } = useRoute();

  useEffect(() => setIsOpen(false), [pathname]);

  return (
    <header className="site-header">
      <nav className="nav-shell" aria-label="Primary navigation">
        <RouteLink className="brand-link" to="/" ariaLabel="InboxBridge home">
          <LogoMark />
        </RouteLink>
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
          {navItems.map((item) => (
            <RouteNavLink className="nav-link" end={item.end} key={item.to} to={item.to}>
              {item.label}
            </RouteNavLink>
          ))}
        </div>
        <a className="nav-cta" href={GITHUB_URL} target="_blank" rel="noreferrer">
          <span>View on GitHub</span><ExternalArrow />
        </a>
      </nav>
    </header>
  );
}

function PageTransition() {
  const { pathname } = useRoute();
  const [isChanging, setIsChanging] = useState(false);

  useEffect(() => {
    setIsChanging(true);
    const frame = window.requestAnimationFrame(() => setIsChanging(false));
    return () => window.cancelAnimationFrame(frame);
  }, [pathname]);

  return <div className={`page-transition${isChanging ? " is-changing" : ""}`}><RouteView /></div>;
}

function ScrollToTop() {
  const { pathname } = useRoute();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [pathname]);
  return null;
}

function ScrollRevealObserver() {
  const { pathname } = useRoute();

  useEffect(() => {
    const page = document.querySelector<HTMLElement>(".page");
    if (!page) return undefined;

    const targets = Array.from(page.querySelectorAll<HTMLElement>([
      ".home-previews",
      ".preview-grid > .preview-link",
      ".process-section",
      ".process-list > .process-row",
      ".language-panel",
      ".language-steps > div",
      ".capabilities-section",
      ".capabilities-grid > .capability-card",
      ".capability-note",
      ".safety-intro",
      ".principles-section",
      ".principles-list > .principle-row",
      ".send-boundary",
      ".boundary-flow",
      ".system-section",
      ".system-diagram",
      ".architecture-notes",
      ".architecture-notes > div",
    ].join(", ")));
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    targets.forEach((target, index) => {
      target.classList.add("scroll-reveal");
      target.style.setProperty("--reveal-delay", `${(index % 3) * 70}ms`);
    });
    page.dataset.motionReady = "true";

    if (reducedMotion || !("IntersectionObserver" in window)) {
      targets.forEach((target) => { target.dataset.revealed = "true"; });
      return undefined;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const target = entry.target as HTMLElement;
        target.dataset.revealed = "true";
        observer.unobserve(target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });

    targets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, [pathname]);

  return null;
}

function Page({ children, className = "" }: PageProps) {
  return <main className={`page ${className}`}>{children}</main>;
}

function PageHeader({ title, intro }: { title: ReactNode; intro: string }) {
  return (
    <header className="page-header page-shell">
      <h1>{title}</h1>
      <p>{intro}</p>
    </header>
  );
}

function ButtonLink({ children, to, variant = "primary" }: { children: ReactNode; to: string; variant?: "primary" | "secondary" }) {
  return <RouteLink className={`button button-${variant}`} to={to}>{children}</RouteLink>;
}

function HalftoneField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = canvas?.parentElement;
    const context = canvas?.getContext("2d");
    if (!canvas || !host || !context) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pointer = { x: 0.5, y: 0.5, targetX: 0.5, targetY: 0.5, active: false };
    let width = 0;
    let height = 0;
    let cellSize = 16;
    let animationFrame = 0;
    let inViewport = true;
    let documentHidden = document.hidden;
    let lastFrame = 0;

    const resize = () => {
      const bounds = host.getBoundingClientRect();
      width = bounds.width;
      height = bounds.height;
      cellSize = width < 520 ? 23 : 16;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (timestamp: number) => {
      if (!inViewport || documentHidden) return;
      const elapsed = Math.min(48, timestamp - lastFrame || 16);
      lastFrame = timestamp;
      pointer.x += (pointer.targetX - pointer.x) * Math.min(1, elapsed * 0.008);
      pointer.y += (pointer.targetY - pointer.y) * Math.min(1, elapsed * 0.008);
      context.clearRect(0, 0, width, height);

      const time = reduceMotion ? 0 : timestamp * 0.0002;
      const columns = Math.ceil(width / cellSize);
      const rows = Math.ceil(height / cellSize);
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          const x = column * cellSize + cellSize * 0.5;
          const y = row * cellSize + cellSize * 0.5;
          const wave = (Math.sin(x * 0.014 + time) + Math.cos(y * 0.018 - time * 0.8) + 2) * 0.25;
          const flowDistance = ((x - width * 0.54) / (width * 0.52)) ** 2 + ((y - height * 0.48) / (height * 0.75)) ** 2;
          const flowField = Math.max(0, 1 - flowDistance);
          const pointerDistance = Math.hypot(x / width - pointer.x, y / height - pointer.y);
          const pointerField = pointer.active ? Math.max(0, 1 - pointerDistance * 4.8) : 0;
          const edgeFade = Math.min(1, Math.min(x, width - x, y, height - y) / (cellSize * 2.5));
          const energy = Math.min(1, (0.1 + wave * 0.25 + flowField * 0.42 + pointerField * 0.6) * edgeFade);
          const size = 0.8 + energy * (reduceMotion ? 1.6 : 2.7);
          context.fillStyle = `rgba(159, 190, 230, ${0.018 + energy * 0.12})`;
          context.fillRect(x - size * 0.5, y - size * 0.5, size, size);
        }
      }

      if (!reduceMotion) animationFrame = window.requestAnimationFrame(draw);
    };

    const schedule = () => {
      if (!reduceMotion && !animationFrame && inViewport && !documentHidden) animationFrame = window.requestAnimationFrame(draw);
    };
    const stop = () => {
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
    };
    const handlePointerMove = (event: PointerEvent) => {
      if (event.pointerType && event.pointerType !== "mouse") return;
      const bounds = host.getBoundingClientRect();
      pointer.targetX = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
      pointer.targetY = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
      pointer.active = true;
      schedule();
    };
    const handlePointerLeave = () => {
      pointer.targetX = 0.5;
      pointer.targetY = 0.5;
      pointer.active = false;
      schedule();
    };
    const handleVisibility = () => {
      documentHidden = document.hidden;
      if (documentHidden) stop();
      else schedule();
    };

    resize();
    if (reduceMotion) draw(0);
    const resizeObserver = new ResizeObserver(() => {
      resize();
      if (reduceMotion) draw(0);
    });
    const intersectionObserver = "IntersectionObserver" in window
      ? new IntersectionObserver(([entry]) => {
        inViewport = entry.isIntersecting;
        if (inViewport) schedule();
        else stop();
      }, { threshold: 0.05 })
      : null;
    resizeObserver.observe(host);
    intersectionObserver?.observe(host);
    host.addEventListener("pointermove", handlePointerMove, { passive: true });
    host.addEventListener("pointerleave", handlePointerLeave, { passive: true });
    document.addEventListener("visibilitychange", handleVisibility);
    schedule();

    return () => {
      stop();
      resizeObserver.disconnect();
      intersectionObserver?.disconnect();
      host.removeEventListener("pointermove", handlePointerMove);
      host.removeEventListener("pointerleave", handlePointerLeave);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, []);

  return <canvas ref={canvasRef} className="halftone-field" aria-hidden="true" />;
}

function FlowPreview() {
  return (
    <div className="hero-visual" aria-label="InboxBridge flow preview">
      <HalftoneField />
      <div className="visual-topline">
        <span>SYS.FLOW / LIVE</span>
        <span>01—03</span>
      </div>
      <div className="visual-route">
        <div className="route-side route-source">
          <span className="mono-label">SOURCE</span>
          <strong>Gmail</strong>
          <small>new thread</small>
        </div>
        <div className="route-core">
          <span className="route-core-mark"><LogoMark compact /></span>
          <strong>InboxBridge</strong>
          <small>conversational layer</small>
        </div>
        <div className="route-side route-client">
          <span className="mono-label">INTERFACE</span>
          <strong>Telegram</strong>
          <small>your review</small>
        </div>
      </div>
      <div className="visual-line" aria-hidden="true" />
      <span className="flow-packet flow-packet-left" aria-hidden="true" />
      <span className="flow-packet flow-packet-right" aria-hidden="true" />
      <div className="visual-message">
        <div className="message-heading"><span className="message-dot" />INCOMING THREAD</div>
        <p>Resumen listo. ¿Qué quieres hacer?</p>
        <span className="message-meta">SPANISH SUMMARY / ATTACHMENT AWARE</span>
      </div>
    </div>
  );
}

const homePreviews = [
  { number: "01", title: "How it works", copy: "Six deliberate handoffs from new mail to verified delivery.", to: "/how-it-works" },
  { number: "02", title: "Capabilities", copy: "Thread context, documents, drafting, and multilingual review.", to: "/capabilities" },
  { number: "03", title: "Safety", copy: "AI proposes. Deterministic systems decide.", to: "/safety" },
  { number: "04", title: "Architecture", copy: "A Telegram interface connected to Gmail through one core.", to: "/architecture" },
];

function HomePage() {
  return (
    <Page className="home-page">
      <section className="hero page-shell">
        <div className="hero-copy">
          <h1>Your inbox,<br /><em>conversational.</em></h1>
          <p>A self-hosted AI Gmail assistant for Telegram, with attachment-aware context, multilingual drafting and explicit confirmation before sending.</p>
          <div className="hero-actions">
            <a className="button button-primary" href={GITHUB_URL} target="_blank" rel="noreferrer">View on GitHub <ExternalArrow /></a>
            <ButtonLink to="/how-it-works" variant="secondary">Explore the flow</ButtonLink>
          </div>
        </div>
        <FlowPreview />
      </section>
      <section className="home-previews page-shell">
        <div className="section-heading">
          <h2>One interface.<br />Four deliberate layers.</h2>
          <p>See how InboxBridge turns a crowded inbox into a conversation you can trust.</p>
        </div>
        <div className="preview-grid">
          {homePreviews.map((preview) => (
            <RouteLink className="preview-link" to={preview.to} key={preview.to}>
              <span className="preview-number">{preview.number}</span>
              <span className="preview-body"><strong>{preview.title}</strong><span>{preview.copy}</span></span>
              <ExternalArrow />
            </RouteLink>
          ))}
        </div>
      </section>
    </Page>
  );
}

const operations = [
  { number: "01", title: "RECEIVE", description: "Gmail Push and Pub/Sub surface new mail from the Primary inbox.", detail: "GMAIL API" },
  { number: "02", title: "UNDERSTAND", description: "Thread history and PDF or document context are grounded together.", detail: "CONTEXT + DOCS" },
  { number: "03", title: "ASK", description: "A Spanish summary arrives in Telegram, ready for your intent.", detail: "TELEGRAM" },
  { number: "04", title: "DRAFT", description: "Reply, new mail, or forwarding instructions become a German draft.", detail: "STRUCTURED OUTPUT" },
  { number: "05", title: "REVIEW", description: "The draft and its Spanish translation are shown before anything moves.", detail: "USER REVIEW" },
  { number: "06", title: "CONFIRM", description: "Explicit confirmation gates Gmail send. Delivery is reconciled afterward.", detail: "VERIFIED DELIVERY" },
];

function ProcessRow({ operation }: { operation: typeof operations[number] }) {
  return (
    <article className="process-row">
      <span className="process-number">{operation.number}</span>
      <div className="process-copy"><h2>{operation.title}</h2><p>{operation.description}</p></div>
      <span className="process-detail">{operation.detail}</span>
    </article>
  );
}

function HowItWorksPage() {
  return (
    <Page className="inner-page">
      <PageHeader title={<>From new mail to<br /><em>verified delivery.</em></>} intro="A clear sequence of small decisions. InboxBridge keeps the person in the loop where it matters." />
      <section className="process-section page-shell">
        <div className="process-list">{operations.map((operation) => <ProcessRow key={operation.number} operation={operation} />)}</div>
      </section>
      <section className="language-panel page-shell">
        <div><h2>One conversation,<br /><em>multiple languages.</em></h2><p>Incoming mail is summarized in Spanish. Your instructions become a polished German draft. The Spanish review stays visible until you confirm.</p></div>
        <div className="language-steps">
          <div><span>INCOMING</span><strong>German thread</strong><small>“Könnten Sie uns das Dokument senden?”</small></div>
          <div><span>YOUR REVIEW</span><strong>Spanish summary</strong><small>“El cliente solicita el documento.”</small></div>
          <div><span>FINAL DRAFT</span><strong>German email</strong><small>“Ja, ich werde es Ihnen schicken.”</small></div>
        </div>
      </section>
    </Page>
  );
}

const capabilityGroups = [
  { index: "01", title: "Understand", copy: "Thread summaries and grounded answers that include the documents attached to the conversation.", tags: ["THREADS", "PDF", "DOCX"] },
  { index: "02", title: "Write", copy: "Replies, new messages, forwards, and conversational edits that keep the full draft current.", tags: ["REPLY", "COMPOSE", "FORWARD"] },
  { index: "03", title: "Translate", copy: "Spanish summaries and review translations, with professional German final emails.", tags: ["ES", "DE", "REVIEW"] },
  { index: "04", title: "Deliver safely", copy: "Explicit confirmation, real recipients, duplicate-send protections, and verified Gmail delivery.", tags: ["CONFIRM", "RECONCILE", "SAFE"] },
];

function CapabilityCard({ group, featured = false }: { group: typeof capabilityGroups[number]; featured?: boolean }) {
  return (
    <article className={`capability-card${featured ? " capability-card-featured" : ""}`}>
      <div className="capability-card-top"><span>{group.index}</span><span className="capability-signal" aria-hidden="true" /></div>
      <h2>{group.title}</h2>
      <p>{group.copy}</p>
      <div className="tag-list">{group.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
    </article>
  );
}

function CapabilitiesPage() {
  return (
    <Page className="inner-page">
      <PageHeader title={<>The useful parts,<br /><em>without the noise.</em></>} intro="InboxBridge is focused on the moments where email becomes work: understanding context, writing clearly, and sending deliberately." />
      <section className="capabilities-section page-shell">
        <div className="capabilities-grid">
          {capabilityGroups.map((group, index) => <CapabilityCard featured={index === 0} group={group} key={group.index} />)}
        </div>
      </section>
      <section className="capability-note page-shell"><p>Ask about a PDF. Summarize the thread. Make the draft shorter. Forward it to a saved contact. The interface stays conversational while the system stays structured.</p></section>
    </Page>
  );
}

const safetyPrinciples = [
  { number: "01", title: "Trusted state", copy: "Recipients, thread identity, and important send metadata come from trusted application state, not generated prose." },
  { number: "02", title: "No bypass", copy: "A generated draft cannot skip explicit confirmation or completeness checks. The send boundary is deterministic." },
  { number: "03", title: "Verification", copy: "After Gmail sends, InboxBridge reconciles the message and guards against blind retries or duplicate delivery." },
];

function SafetyPage() {
  return (
    <Page className="inner-page safety-page">
      <PageHeader title={<>AI proposes.<br /><em>Systems decide.</em></>} intro="The model can suggest language. It cannot decide who receives it, whether it is complete, or whether it has been sent." />
      <div className="safety-intro page-shell"><div className="safety-statement"><span className="safety-quote">“</span><p>InboxBridge treats a draft as a proposal, not an action.</p></div><div className="safety-state"><span>SAFE SEND STATE</span><strong>Awaiting explicit confirmation</strong><small>AI output is visible. Gmail is untouched.</small></div></div>
      <section className="principles-section page-shell">
        <div className="principles-list">{safetyPrinciples.map((principle) => <article className="principle-row" key={principle.number}><span className="principle-number">{principle.number}</span><h2>{principle.title}</h2><p>{principle.copy}</p></article>)}</div>
      </section>
      <section className="send-boundary page-shell">
        <div className="section-heading"><h2>Every handoff has a visible state.</h2></div>
        <div className="boundary-flow"><div><span>01</span><strong>AI draft</strong><small>proposal</small></div><i aria-hidden="true" /><div className="boundary-active"><span>02</span><strong>Confirm</strong><small>human decision</small></div><i aria-hidden="true" /><div><span>03</span><strong>Gmail send</strong><small>verified result</small></div></div>
      </section>
    </Page>
  );
}

function ArchitecturePage() {
  return (
    <Page className="inner-page architecture-page">
      <PageHeader title={<>A small core,<br /><em>clear boundaries.</em></>} intro="Telegram is the interface. Gmail remains the source of truth. InboxBridge coordinates the decisions between them." />
      <section className="system-section page-shell">
        <div className="system-diagram">
          <div className="system-node system-interface"><span className="mono-label">INTERFACE</span><strong>Telegram</strong><small>Bot API</small></div>
          <div className="system-link" aria-hidden="true"><span>INPUT</span></div>
          <div className="system-core"><span className="mono-label">CORE ENGINE</span><strong>InboxBridge</strong><p>Intent routing, structured LLM output, trusted state, and verified delivery.</p><div className="core-parts"><span>Intent</span><span>LLM</span><span>Docs</span><span>State</span><span>Delivery</span></div></div>
          <div className="system-link" aria-hidden="true"><span>OUTPUT</span></div>
          <div className="system-node system-external"><span className="mono-label">SOURCE OF TRUTH</span><strong>Gmail</strong><small>OAuth 2.0 / API</small></div>
        </div>
      </section>
      <div className="architecture-notes page-shell"><div><h2>Built to stay small.</h2><p>Docker and Linux VPS deployment keep the runtime portable. SQLite stores identifiers and statuses, not email bodies or attachment content.</p></div><div><h2>Designed to reconcile.</h2><p>Retries and restarts return to trusted state. An uncertain send is checked against Gmail before anything can be sent again.</p></div></div>
    </Page>
  );
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-shell page-shell">
        <div className="footer-brand"><LogoMark /><p>Conversational Gmail via Telegram.</p></div>
        <div className="footer-nav">{navItems.slice(1).map((item) => <RouteLink key={item.to} to={item.to}>{item.label}</RouteLink>)}</div>
        <div className="footer-meta"><a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub <ExternalArrow /></a><small>© 2026 Daniel Arroyo</small></div>
      </div>
    </footer>
  );
}

const pageMeta: Record<string, { title: string; description: string }> = {
  "/": { title: "InboxBridge | Conversational Gmail via Telegram", description: "A multilingual Gmail assistant for Telegram with attachment-aware context, safe draft confirmation and verified delivery." },
  "/how-it-works": { title: "How it works | InboxBridge", description: "See how InboxBridge moves from incoming Gmail mail to a confirmed, verified delivery." },
  "/capabilities": { title: "Capabilities | InboxBridge", description: "Explore InboxBridge capabilities for email context, drafting, translation, and safe delivery." },
  "/safety": { title: "Safety | InboxBridge", description: "Understand the trusted state, no bypass, and verification principles behind InboxBridge." },
  "/architecture": { title: "Architecture | InboxBridge", description: "A clear view of the InboxBridge Telegram, core engine, and Gmail architecture." },
};

function PageMeta() {
  const { pathname } = useRoute();
  useEffect(() => {
    const meta = pageMeta[pathname] ?? pageMeta["/"];
    document.title = meta.title;
    document.querySelector('meta[name="description"]')?.setAttribute("content", meta.description);
  }, [pathname]);
  return null;
}

function App() {
  return (
    <AppRouter>
      <PageMeta />
      <ScrollToTop />
      <ScrollRevealObserver />
      <Header />
      <PageTransition />
      <Footer />
    </AppRouter>
  );
}

const routeComponents = {
  "/": HomePage,
  "/how-it-works": HowItWorksPage,
  "/capabilities": CapabilitiesPage,
  "/safety": SafetyPage,
  "/architecture": ArchitecturePage,
};

function RouteView() {
  const { pathname, navigate } = useRoute();
  const knownPage = routeComponents[pathname as keyof typeof routeComponents];
  const PageComponent = knownPage ?? HomePage;

  useEffect(() => {
    if (!knownPage) navigate("/");
  }, [knownPage, navigate]);

  return <PageComponent />;
}

createRoot(document.getElementById("root")!).render(<App />);
