# fastpitch-per-phoneme

Companion notebook for **[Generate Slow, Don't Slow the Generation](https://ramwise.dev/blog/generate-slow-dont-slow-the-generation/)**.

Slowing speech to *teach* it means slowing each sound differently — hold the
vowels, keep the stops crisp. That needs a model that exposes a **per-phoneme
duration** knob. [FastPitch](https://arxiv.org/abs/2006.06873) does: its
`generate_spectrogram` takes a `pace` value *per token*. This notebook shows the
approach — a uniform slowdown (one number) versus a class-aware one (hold
vowels, keep stops crisp) — and why it beats the single global `length_scale`
scalar that the browser (Piper) version is stuck with.

- [`per_phoneme_slowdown.ipynb`](per_phoneme_slowdown.ipynb) — load pretrained
  FastPitch + HiFi-GAN, build a per-token `pace` tensor, and hear uniform vs.
  class-aware slowdown.

**Not zero-dependency.** Unlike the other examples in this repo, this one needs
a **GPU** and NVIDIA NeMo, so run it on Google Colab (*Runtime → T4 GPU*). It's a
teaching notebook meant to be run there, not in CI. The exact ear-tuned per-class
scales and loudness-leveling that the shipped app uses are left out on purpose —
what's here is the general, teachable mechanism.
