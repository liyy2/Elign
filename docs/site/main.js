(() => {
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const revealElements = [...document.querySelectorAll(".reveal")];
  if (!prefersReducedMotion && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.18 }
    );

    revealElements.forEach((el) => {
      if (!el.classList.contains("in-view")) {
        observer.observe(el);
      }
    });
  } else {
    revealElements.forEach((el) => el.classList.add("in-view"));
  }

  const steps = [...document.querySelectorAll(".concept-step")];
  let stepIndex = 0;
  let stepTimer = null;

  const setActiveStep = (idx) => {
    steps.forEach((el, i) => {
      el.classList.toggle("is-active", i === idx);
    });
  };

  if (steps.length > 0) {
    setActiveStep(0);
    if (!prefersReducedMotion) {
      stepTimer = setInterval(() => {
        stepIndex = (stepIndex + 1) % steps.length;
        setActiveStep(stepIndex);
      }, 2900);

      steps.forEach((stepEl, idx) => {
        stepEl.addEventListener("mouseenter", () => {
          stepIndex = idx;
          setActiveStep(stepIndex);
        });
      });
    }
  }

  const fedSequence = document.getElementById("fedSequence");
  const stageButtons = [...document.querySelectorAll(".stage-chip")];
  const sequenceCaption = document.getElementById("sequenceCaption");
  const stageCaptions = [
    "1) A node follows the shared trajectory to the branch point.",
    "2) The shared prefix branches into grouped rollout trajectories.",
    "3) Terminal rollouts are sent to the preference model (MLFF).",
    "4) Rewards are computed and used for FED-GRPO policy updates."
  ];

  let currentStage = 0;
  let stageTimer = null;
  let tokenStops = [];

  const getTokenAndPath = (tokenId, pathId) => {
    if (!fedSequence) {
      return null;
    }
    const token = fedSequence.querySelector(`#${tokenId}`);
    const path = fedSequence.querySelector(`#${pathId}`);
    if (!token || !path) {
      return null;
    }
    return { token, path };
  };

  const hideAllTokens = () => {
    if (!fedSequence) {
      return;
    }
    [...fedSequence.querySelectorAll(".token")].forEach((token) => {
      token.style.opacity = "0";
    });
  };

  const placeTokenOnPath = (tokenId, pathId, progress) => {
    const pair = getTokenAndPath(tokenId, pathId);
    if (!pair) {
      return;
    }
    const length = pair.path.getTotalLength();
    const point = pair.path.getPointAtLength(length * progress);
    pair.token.setAttribute("cx", point.x.toFixed(2));
    pair.token.setAttribute("cy", point.y.toFixed(2));
    pair.token.style.opacity = "1";
  };

  const animateTokenOnPath = (tokenId, pathId, durationMs, options = {}) => {
    const pair = getTokenAndPath(tokenId, pathId);
    if (!pair) {
      return () => {};
    }

    const { loop = true, delayMs = 0 } = options;
    const totalLength = pair.path.getTotalLength();
    pair.token.style.opacity = "1";

    let rafId = 0;
    let startTs = 0;
    let stopped = false;

    const tickToken = (ts) => {
      if (stopped) {
        return;
      }
      if (startTs === 0) {
        startTs = ts + delayMs;
      }
      if (ts < startTs) {
        rafId = window.requestAnimationFrame(tickToken);
        return;
      }

      const elapsed = ts - startTs;
      const rawProgress = elapsed / durationMs;
      const progress = loop ? rawProgress % 1 : Math.min(rawProgress, 1);
      const point = pair.path.getPointAtLength(totalLength * progress);

      pair.token.setAttribute("cx", point.x.toFixed(2));
      pair.token.setAttribute("cy", point.y.toFixed(2));

      if (!loop && rawProgress >= 1) {
        return;
      }
      rafId = window.requestAnimationFrame(tickToken);
    };

    rafId = window.requestAnimationFrame(tickToken);
    return () => {
      stopped = true;
      window.cancelAnimationFrame(rafId);
      pair.token.style.opacity = "0";
    };
  };

  const stopTokenAnimations = () => {
    tokenStops.forEach((stopFn) => stopFn());
    tokenStops = [];
    hideAllTokens();
  };

  const runStageAnimation = (stage) => {
    stopTokenAnimations();
    if (!fedSequence) {
      return;
    }

    if (prefersReducedMotion) {
      if (stage === 0) {
        placeTokenOnPath("token-main", "edge-main", 1);
      }
      if (stage === 1) {
        placeTokenOnPath("token-b1", "edge-b1", 1);
        placeTokenOnPath("token-b2", "edge-b2", 1);
        placeTokenOnPath("token-b3", "edge-b3", 1);
      }
      if (stage === 2) {
        placeTokenOnPath("token-p1", "edge-p1", 1);
        placeTokenOnPath("token-p2", "edge-p2", 1);
        placeTokenOnPath("token-p3", "edge-p3", 1);
      }
      if (stage === 3) {
        placeTokenOnPath("token-r", "edge-r", 1);
        placeTokenOnPath("token-u", "edge-u", 1);
      }
      return;
    }

    if (stage === 0) {
      tokenStops.push(animateTokenOnPath("token-main", "edge-main", 1700));
    }
    if (stage === 1) {
      tokenStops.push(animateTokenOnPath("token-b1", "edge-b1", 1500, { delayMs: 0 }));
      tokenStops.push(animateTokenOnPath("token-b2", "edge-b2", 1500, { delayMs: 230 }));
      tokenStops.push(animateTokenOnPath("token-b3", "edge-b3", 1500, { delayMs: 460 }));
    }
    if (stage === 2) {
      tokenStops.push(animateTokenOnPath("token-p1", "edge-p1", 1500, { delayMs: 0 }));
      tokenStops.push(animateTokenOnPath("token-p2", "edge-p2", 1500, { delayMs: 220 }));
      tokenStops.push(animateTokenOnPath("token-p3", "edge-p3", 1500, { delayMs: 440 }));
    }
    if (stage === 3) {
      tokenStops.push(animateTokenOnPath("token-r", "edge-r", 1250, { delayMs: 0 }));
      tokenStops.push(animateTokenOnPath("token-u", "edge-u", 1250, { delayMs: 320 }));
    }
  };

  const setStage = (stage, manual = false) => {
    currentStage = stage;
    if (fedSequence) {
      fedSequence.dataset.stage = String(stage);
    }
    stageButtons.forEach((btn) => {
      btn.classList.toggle("is-active", Number.parseInt(btn.dataset.stage || "0", 10) === stage);
    });
    if (sequenceCaption) {
      sequenceCaption.textContent = stageCaptions[stage] || stageCaptions[0];
    }
    runStageAnimation(stage);

    if (manual && !prefersReducedMotion) {
      window.clearInterval(stageTimer);
      stageTimer = window.setInterval(() => {
        setStage((currentStage + 1) % 4);
      }, 3600);
    }
  };

  if (fedSequence) {
    stageButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number.parseInt(btn.dataset.stage || "0", 10);
        setStage(idx, true);
      });
    });

    setStage(0);
    if (!prefersReducedMotion) {
      stageTimer = window.setInterval(() => {
        setStage((currentStage + 1) % 4);
      }, 3600);
    }
  }

  const canvas = document.getElementById("hero-canvas");
  const ctx = canvas?.getContext("2d", { alpha: true });

  if (canvas && ctx && !prefersReducedMotion) {
    const particles = [];
    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const palette = ["rgba(15,118,110,0.52)", "rgba(204,140,74,0.42)", "rgba(46,145,131,0.38)"];

    const resetParticles = () => {
      particles.length = 0;
      const count = Math.max(34, Math.min(95, Math.floor((width * height) / 27000)));
      for (let i = 0; i < count; i += 1) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.42,
          vy: (Math.random() - 0.5) * 0.42,
          r: Math.random() * 1.8 + 0.6,
          c: palette[i % palette.length]
        });
      }
    };

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = Math.max(window.innerHeight * 0.95, 620);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      resetParticles();
    };

    let rafId = 0;
    const tick = () => {
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < particles.length; i += 1) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < -10 || p.x > width + 10) {
          p.vx *= -1;
        }
        if (p.y < -10 || p.y > height + 10) {
          p.vy *= -1;
        }

        ctx.beginPath();
        ctx.fillStyle = p.c;
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }

      for (let i = 0; i < particles.length; i += 1) {
        for (let j = i + 1; j < particles.length; j += 1) {
          const a = particles[i];
          const b = particles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.hypot(dx, dy);
          if (dist < 100) {
            const alpha = (1 - dist / 100) * 0.12;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(19,34,41,${alpha})`;
            ctx.lineWidth = 0.7;
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      rafId = window.requestAnimationFrame(tick);
    };

    resize();
    tick();

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      window.cancelAnimationFrame(rafId);
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        resize();
        tick();
      }, 110);
    });
  }

  window.addEventListener("beforeunload", () => {
    if (stepTimer) {
      window.clearInterval(stepTimer);
    }
    if (stageTimer) {
      window.clearInterval(stageTimer);
    }
    stopTokenAnimations();
  });
})();
