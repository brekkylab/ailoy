---
layout: home

hero:
  name: Ailoy
  text: 
  tagline: AI agent builder with a VM at its heart.
  actions:
    - theme: brand
      text: Guide
      link: /guide/quick-start
    - theme: alt
      text: View on GitHub
      link: https://github.com/brekkylab/ailoy

---

[Full example lists](https://github.com/brekkylab/ailoy#what-can-an-agent-do)

<div class="demo-grid">
  <figure><img src="./images/ailoy-cad.gif" alt="CAD modeling"><figcaption>CAD modeling</figcaption></figure>
  <figure><img src="./images/ailoy-sam3-video.gif" alt="Image editing"><figcaption>Image editing</figcaption></figure>
  <figure><img src="./images/ailoy-openttd.gif" alt="Gameplay"><figcaption>Gameplay</figcaption></figure>
  <figure><img src="./images/ailoy-laya.gif" alt="Laya(JEV)"><figcaption>Laya(JEV)</figcaption></figure>
</div>

<style>
.demo-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}
.demo-grid figure { margin: 0; }
.demo-grid img { width: 100%; border-radius: 8px; }
.demo-grid figcaption { text-align: center; font-size: 14px; color: var(--vp-c-text-2); margin-top: 4px; }
@media (max-width: 640px) { .demo-grid { grid-template-columns: 1fr; } }
</style>
