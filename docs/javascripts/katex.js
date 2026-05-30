// Render KaTeX math after each (instant-navigation) page load.
// Paired with the `pymdownx.arithmatex` (generic) extension, which wraps math in
// \( … \) (inline) and \[ … \] (display) delimiters.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
  });
});
