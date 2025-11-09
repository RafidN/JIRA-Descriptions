(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("ticketForm");
  const generateBtn = $("generateBtn");
  const createBtn = $("createBtn");
  const loading = $("loading");
  const loadingPhase = $("loadingPhase");
  const errorBox = $("error");
  const desc = $("description");

  const issueView = $("issue-view");
  const createCard = $("create-issue-card");
  const issueTitle = $("issueTitle");
  const issueTypeBadge = $("issueTypeBadge");
  const priorityText = $("priorityText");
  const assigneeText = $("assigneeText");
  const rendered = $("rendered");
  const debug = $("debug");

  let lastMarkdown = "";

  function setLoading(on, phaseText = "working") {
    loading.classList.toggle("hidden", !on);
    loadingPhase.textContent = phaseText;
    generateBtn.disabled = on;
    // Disable all except the Create button so the swap can still happen if needed
    form.querySelectorAll("input, select, textarea, button").forEach(el => {
      if (el !== generateBtn && el !== createBtn) el.disabled = on;
    });
  }

  // --- Minimal Markdown fallback (headings, code fences, lists, bold/italic, links) ---
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function basicMarkdown(md) {
    // Code fences
    md = md.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    // Inline code
    md = md.replace(/`([^`]+)`/g, (_, code) => `<code>${escapeHtml(code)}</code>`);
    // Headings
    md = md.replace(/^###### (.*)$/gm, "<h6>$1</h6>");
    md = md.replace(/^##### (.*)$/gm, "<h5>$1</h5>");
    md = md.replace(/^#### (.*)$/gm, "<h4>$1</h4>");
    md = md.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    md = md.replace(/^## (.*)$/gm, "<h2>$1</h2>");
    md = md.replace(/^# (.*)$/gm, "<h1>$1</h1>");
    // Bold / italic
    md = md.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    md = md.replace(/_([^_]+)_/g, "<em>$1</em>");
    // Links [text](url)
    md = md.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // Checkboxes
    md = md.replace(/^- \[ \] (.*)$/gm, '<li><input type="checkbox" disabled> $1</li>');
    md = md.replace(/^- \[x\] (.*)$/gmi, '<li><input type="checkbox" checked disabled> $1</li>');
    // Unordered list items
    md = md.replace(/^(?:- |\* )(.*)$/gm, "<li>$1</li>");
    // Wrap consecutive <li> into <ul>
    md = md.replace(/(<li>[\s\S]*?<\/li>)(\s*(?=<li>))/g, "$1");
    md = md.replace(/(?:\s*<li>[\s\S]*?<\/li>)+/g, (block) => `<ul>${block}</ul>`);
    // Paragraphs (very naive)
    md = md.replace(/(?:^|\n)([^<\n][^\n]*)(?=\n|$)/g, (m, line) => {
      const trimmed = line.trim();
      if (!trimmed) return "";
      if (/^<(h\d|ul|li|pre|blockquote|p|code)/i.test(trimmed)) return m;
      return `\n<p>${trimmed}</p>`;
    });
    return md;
  }

  function renderMarkdown(md) {
    try {
      if (window.marked) {
        // Support both modern and older builds
        if (typeof window.marked.setOptions === "function") {
          window.marked.setOptions({ gfm: true, breaks: true });
        }
        const html = (typeof window.marked.parse === "function")
          ? window.marked.parse(md)
          : window.marked(md);
        rendered.innerHTML = html;
        console.log("[markdown] rendered with marked");
      } else {
        rendered.innerHTML = basicMarkdown(md);
        console.warn("[markdown] using basic fallback renderer");
      }
    } catch (e) {
      console.error("Markdown render error:", e);
      rendered.innerHTML = basicMarkdown(md); // safe fallback
    }
  }

  function showIssueView() {
    issueTitle.textContent = $("summary").value || "(no summary)";
    issueTypeBadge.textContent = $("issueType").value;
    priorityText.textContent = $("priority").value;
    assigneeText.textContent = $("assignee").value || "Unassigned";

    const md = desc.value || lastMarkdown || "_No description provided._";
    renderMarkdown(md);

    createCard.classList.add("hidden");
    issueView.classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function generateActionItems() {
    errorBox.classList.add("hidden");
    const repo_url = $("repoUrl").value.trim();
    const jira_title = $("summary").value.trim();
    const max_context_chunks = parseInt(($("maxK").value || "12"), 10);
    const chunk_size = parseInt(($("chunkSize").value || "1200"), 10);
    const overlap = parseInt(($("overlap").value || "150"), 10);

    if (!repo_url || !jira_title) {
      errorBox.textContent = "Please enter both a Summary and a GitHub Repo URL.";
      errorBox.classList.remove("hidden");
      return;
    }

    try {
      setLoading(true, "cloning / indexing");
      const payload = { repo_url, jira_title, max_context_chunks, chunk_size, overlap };

      const t0 = performance.now();
      const resp = await fetch("/action-items", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        setLoading(false);
        const text = await resp.text().catch(() => "");
        errorBox.textContent = text || `Request failed with ${resp.status}`;
        errorBox.classList.remove("hidden");
        return;
      }

      setLoading(true, "rendering plan");
      const data = await resp.json();
      lastMarkdown = data.action_items_markdown || "";
      desc.value = lastMarkdown;

      debug.textContent = JSON.stringify(
        { used_files: data.used_files, used_chunks: data.used_chunks, took_ms: Math.round(performance.now() - t0) },
        null, 2
      );

      setLoading(false, "done");
    } catch (err) {
      setLoading(false);
      errorBox.textContent = `Error: ${err}`;
      errorBox.classList.remove("hidden");
    }
  }

  generateBtn.addEventListener("click", generateActionItems);

  createBtn.addEventListener("click", (e) => {
    e.preventDefault();
    showIssueView();
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    showIssueView();
  });

  $("backBtn").addEventListener("click", () => {
    issueView.classList.add("hidden");
    createCard.classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  $("cancelBtn").addEventListener("click", () => {
    form.reset();
    desc.value = "";
    lastMarkdown = "";
    errorBox.classList.add("hidden");
  });
})();
