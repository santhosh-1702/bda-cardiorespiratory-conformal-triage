/* ==========================================================================
   Cardiorespiratory Triage with PPG & Conformal Prediction
   Interactive Client Script
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initReadingProgress();
  initScrollspy();
  initMobileMenu();
  initCopyButtons();
  initTabs();
  initSearch();
  initLightbox();
  initBackToTop();
  initMathJaxOrKaTeX();
  initMermaid();
});

/* --------------------------------------------------------------------------
   Theme Switcher (Dark / Light Mode)
   -------------------------------------------------------------------------- */
function initTheme() {
  const themeToggle = document.getElementById('theme-toggle');
  const storedTheme = localStorage.getItem('bda_docs_theme');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  const initialTheme = storedTheme || (systemPrefersDark ? 'dark' : 'dark'); // Default to dark
  setTheme(initialTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
      localStorage.setItem('bda_docs_theme', newTheme);
      
      // Re-render mermaid with matching theme if mermaid exists
      if (window.mermaid) {
        initMermaid();
      }
    });
  }
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    const sunIcon = themeToggle.querySelector('.icon-sun');
    const moonIcon = themeToggle.querySelector('.icon-moon');
    if (sunIcon && moonIcon) {
      if (theme === 'light') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
      } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
      }
    }
  }
}

/* --------------------------------------------------------------------------
   Reading Progress Bar
   -------------------------------------------------------------------------- */
function initReadingProgress() {
  const progressBar = document.getElementById('reading-progress');
  if (!progressBar) return;

  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
    progressBar.style.width = `${progress}%`;
  }, { passive: true });
}

/* --------------------------------------------------------------------------
   Scrollspy & Active Navigation Link
   -------------------------------------------------------------------------- */
function initScrollspy() {
  const tocLinks = document.querySelectorAll('.toc-link');
  const sections = document.querySelectorAll('section[id]');
  if (!tocLinks.length || !sections.length) return;

  const observerOptions = {
    root: null,
    rootMargin: '-10% 0px -70% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        tocLinks.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));
}

/* --------------------------------------------------------------------------
   Mobile Drawer Toggle
   -------------------------------------------------------------------------- */
function initMobileMenu() {
  const menuBtn = document.getElementById('mobile-menu-btn');
  const sidebar = document.getElementById('sidebar');
  if (!menuBtn || !sidebar) return;

  menuBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    sidebar.classList.toggle('mobile-open');
  });

  // Close when clicking a TOC link on mobile
  const links = sidebar.querySelectorAll('.toc-link');
  links.forEach(link => {
    link.addEventListener('click', () => {
      sidebar.classList.remove('mobile-open');
    });
  });

  // Close when clicking outside
  document.addEventListener('click', (e) => {
    if (!sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
      sidebar.classList.remove('mobile-open');
    }
  });
}

/* --------------------------------------------------------------------------
   Code Copy-to-Clipboard
   -------------------------------------------------------------------------- */
function initCopyButtons() {
  const copyButtons = document.querySelectorAll('.copy-btn');
  copyButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const codeBlock = btn.closest('.code-block-container')?.querySelector('code, pre');
      if (!codeBlock) return;

      const codeText = codeBlock.innerText;
      try {
        await navigator.clipboard.writeText(codeText);
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg> Copied!
        `;
        btn.style.color = '#10b981';
        btn.style.borderColor = '#10b981';

        setTimeout(() => {
          btn.innerHTML = originalHtml;
          btn.style.color = '';
          btn.style.borderColor = '';
        }, 2000);
      } catch (err) {
        console.error('Failed to copy text: ', err);
      }
    });
  });
}

/* --------------------------------------------------------------------------
   Tab Navigation Components
   -------------------------------------------------------------------------- */
function initTabs() {
  const tabContainers = document.querySelectorAll('.tab-container');
  tabContainers.forEach(container => {
    const btns = container.querySelectorAll('.tab-btn');
    const panes = container.querySelectorAll('.tab-pane');

    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');

        btns.forEach(b => b.classList.remove('active'));
        panes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const activePane = container.querySelector(`.tab-pane[data-tab="${targetTab}"]`);
        if (activePane) {
          activePane.classList.add('active');
        }
      });
    });
  });
}

/* --------------------------------------------------------------------------
   TOC Quick Filter / Search
   -------------------------------------------------------------------------- */
function initSearch() {
  const searchInput = document.getElementById('sidebar-search');
  const tocLinks = document.querySelectorAll('.toc-link');
  if (!searchInput || !tocLinks.length) return;

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    tocLinks.forEach(link => {
      const text = link.textContent.toLowerCase();
      if (!query || text.includes(query)) {
        link.style.display = 'flex';
      } else {
        link.style.display = 'none';
      }
    });
  });
}

/* --------------------------------------------------------------------------
   High-Res Lightbox Modal
   -------------------------------------------------------------------------- */
function initLightbox() {
  const trigger = document.getElementById('lightbox-trigger');
  const modal = document.getElementById('lightbox-modal');
  const modalImg = document.getElementById('lightbox-img');
  const closeBtn = document.getElementById('lightbox-close');

  if (!trigger || !modal || !modalImg) return;

  trigger.addEventListener('click', () => {
    const srcImg = trigger.querySelector('img');
    if (srcImg) {
      modalImg.src = srcImg.src;
      modal.classList.add('active');
      document.body.style.overflow = 'hidden';
    }
  });

  const closeModal = () => {
    modal.classList.remove('active');
    document.body.style.overflow = '';
  };

  if (closeBtn) closeBtn.addEventListener('click', closeModal);

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeModal();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });
}

/* --------------------------------------------------------------------------
   Back to Top Button
   -------------------------------------------------------------------------- */
function initBackToTop() {
  const btn = document.getElementById('back-to-top');
  if (!btn) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 400) {
      btn.classList.add('visible');
    } else {
      btn.classList.remove('visible');
    }
  }, { passive: true });

  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

/* --------------------------------------------------------------------------
   LaTeX Math Rendering via KaTeX
   -------------------------------------------------------------------------- */
function initMathJaxOrKaTeX() {
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false
      });
    } catch (err) {
      console.warn('KaTeX rendering error:', err);
    }
  }
}

/* --------------------------------------------------------------------------
   Mermaid Flowchart Rendering
   -------------------------------------------------------------------------- */
function initMermaid() {
  if (window.mermaid) {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    try {
      window.mermaid.initialize({
        startOnLoad: true,
        theme: currentTheme === 'dark' ? 'dark' : 'neutral',
        themeVariables: {
          darkMode: currentTheme === 'dark',
          primaryColor: currentTheme === 'dark' ? '#1e293b' : '#f1f5f9',
          primaryTextColor: currentTheme === 'dark' ? '#f8fafc' : '#0f172a',
          primaryBorderColor: '#06b6d4',
          lineColor: '#06b6d4',
          secondaryColor: '#0f172a',
          tertiaryColor: '#1e293b',
          edgeLabelBackground: currentTheme === 'dark' ? '#0f172a' : '#ffffff'
        },
        flowchart: {
          curve: 'basis',
          htmlLabels: true
        }
      });
      window.mermaid.run();
    } catch (err) {
      console.warn('Mermaid rendering error:', err);
    }
  }
}
