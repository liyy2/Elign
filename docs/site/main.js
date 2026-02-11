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

  const forceWeight = document.getElementById("forceWeight");
  const energyWeight = document.getElementById("energyWeight");
  const groupSize = document.getElementById("groupSize");

  const forceOut = document.getElementById("forceOut");
  const energyOut = document.getElementById("energyOut");
  const groupOut = document.getElementById("groupOut");

  const forceBar = document.getElementById("forceBar");
  const energyBar = document.getElementById("energyBar");
  const mixFormula = document.getElementById("mixFormula");
  const rolloutStrip = document.getElementById("rolloutStrip");
  const rolloutCaption = document.getElementById("rolloutCaption");

  const renderRolloutStrip = (k) => {
    const shown = Math.min(12, k);
    rolloutStrip.innerHTML = "";
    for (let i = 0; i < shown; i += 1) {
      const chip = document.createElement("span");
      chip.className = "rollout-chip";
      chip.style.animationDelay = `${(i % 6) * -0.15}s`;
      rolloutStrip.appendChild(chip);
    }
    rolloutCaption.textContent = `Showing ${shown} sampled rollouts from K=${k}.`;
  };

  const updateMixer = () => {
    const wf = Number.parseFloat(forceWeight.value);
    const we = Number.parseFloat(energyWeight.value);
    const k = Number.parseInt(groupSize.value, 10);

    forceOut.value = wf.toFixed(2);
    energyOut.value = we.toFixed(2);
    groupOut.value = String(k);

    const total = Math.max(wf + we, 1e-6);
    const forcePct = Math.max(12, (wf / total) * 100);
    const energyPct = Math.max(12, (we / total) * 100);

    forceBar.style.height = `${forcePct}%`;
    energyBar.style.height = `${energyPct}%`;
    mixFormula.textContent = `A = ${wf.toFixed(2)} · z(F) + ${we.toFixed(2)} · z(E)`;

    renderRolloutStrip(k);
  };

  if (forceWeight && energyWeight && groupSize) {
    [forceWeight, energyWeight, groupSize].forEach((el) => {
      el.addEventListener("input", updateMixer);
    });
    updateMixer();
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
  });
})();
