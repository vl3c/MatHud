# Renderer Rendering Modes And Benchmarks

MatHud renderers now support two execution strategies:

- `legacy` — emits primitives immediately using the historical helper pipeline.
- `optimized` — generates batched plans, reuses pooled DOM primitives, and minimizes Canvas2D state churn.

Both strategies produce identical geometry; the optimized path focuses on reducing DOM operations and context mutations.

## Selecting A Strategy

The active renderer inspects `window.MatHudRendererStrategy` (or the `mathud.renderer.strategy` key in `localStorage`) during construction. Valid values are `legacy` and `optimized`.

```python
# Switch at runtime before creating the canvas
from browser import window
window.MatHudRendererStrategy = "optimized"

# Persist between sessions
window.localStorage.setItem("mathud.renderer.strategy", "optimized")
```

Absent a preference, renderers default to the legacy strategy to preserve existing behaviour.

## Benchmark Procedure

Use `run_renderer_performance` from `static/client/client_tests/renderer_performance_tests.py` to gather metrics. The helper accepts the workload description, iteration count, and render mode:

```python
from renderer_performance_tests import run_renderer_performance

# Measure legacy path
legacy = run_renderer_performance(iterations=2, render_mode="legacy")

# Measure optimized path
optimized = run_renderer_performance(iterations=2, render_mode="optimized")
```

Each call returns timing summaries for draw, pan, and zoom operations together with DOM node counts. Metrics are also logged to the browser console for quick inspection.

## Maintaining Baselines

1. Ensure the Brython test harness is loaded (for example, issue `run tests` in the client console).
2. Invoke `run_renderer_performance` for both strategies with the shared baseline scene.
3. Record the results in your performance notes or analytics dashboard.

The unit test `TestRendererPerformance.test_optimized_renderer_not_slower` confirms that the optimized mode remains within 15% of the legacy timings. Update your benchmark logs after capturing new measurements so the tables reflect the latest verified data.


