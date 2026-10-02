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

<div class="demo-grid">
  <figure><img src="./images/ailoy-cad.gif" alt="CAD demo"></figure>
  <figure><img src="./images/ailoy-sam3-video.gif" alt="SAM3 demo"></figure>
  <figure><img src="./images/ailoy-openttd.gif" alt="Gameplay demo"></figure>
  <figure><img src="./images/ailoy-laya.gif" alt="Laya demo"></figure>
</div>

<style>
.demo-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}
.demo-grid figure { margin: 0; }
.demo-grid img { width: 100%; border-radius: 8px; }
@media (max-width: 640px) { .demo-grid { grid-template-columns: 1fr; } }
</style>
