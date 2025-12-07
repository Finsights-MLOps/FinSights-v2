## Streamlit-specific trap:

The key problem: st.markdown('<div class="feature-card">') does not become the parent of the st.container(border=True) that comes after it. Streamlit builds its own layout tree; each st.* call is wrapped in its own block <div>.

```
<div class="stMarkdown">   <!-- holds <div class="feature-card"> -->
  <div class="feature-card"></div>
</div>

<div data-testid="stVerticalBlockBorderWrapper">  <!-- st.container -->
  ...
</div>
```

- selector: never matches anything.
- the six cards are “dead” even though the CSS looks right.   
  

```
<div data-testid="column">
  <div data-testid="stVerticalBlockBorderWrapper" class="outer">
    ... icon ...
    <div data-testid="stVerticalBlockBorderWrapper" class="inner">
      ... h4 id="context-aware-q-a" ...
    </div>
    ... icon 2 ...
    <div data-testid="stVerticalBlockBorderWrapper" class="inner">
      ... h4 id="citation-first-answers" ...
    </div>
  </div>
</div>
```

