"""
Automated validation report generation.

Collects results from all cohort validators, head-to-head comparisons,
and reversion analysis into a structured HTML report.
"""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationReport:
    """Generate a comprehensive validation report.

    Accumulates results from individual validators and produces
    an HTML report with tables, plots, and interpretation.

    Parameters
    ----------
    model_name : str
        Name of the model being validated.
    output_dir : str or Path
        Directory for saving report and assets.
    """

    def __init__(self, model_name="softHRD_v2", output_dir="validation_results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cohort_results = []
        self.head_to_head_result = None
        self.reversion_result = None
        self.figures = []  # (title, fig) pairs
        self.notes = []

    def add_cohort_result(self, cohort_name, metrics_dict, plots=None):
        """Add validation results for one cohort.

        Parameters
        ----------
        cohort_name : str
        metrics_dict : dict
            Output from BaseCohortValidator.validate().
        plots : list of matplotlib.figure.Figure or None
        """
        self.cohort_results.append({
            "cohort": cohort_name,
            "metrics": metrics_dict,
        })
        if plots:
            for i, fig in enumerate(plots):
                self.figures.append((f"{cohort_name} — Plot {i+1}", fig))

    def add_head_to_head(self, h2h_result, h2h_obj=None):
        """Add head-to-head comparison results.

        Parameters
        ----------
        h2h_result : pd.DataFrame
            Output from HeadToHead.run_all().
        h2h_obj : HeadToHead or None
            If provided, generates comparison plots automatically.
        """
        self.head_to_head_result = h2h_result
        if h2h_obj is not None:
            try:
                self.figures.append(
                    ("ROC Comparison", h2h_obj.plot_roc_comparison())
                )
            except Exception as e:
                logger.warning("ROC plot failed: %s", e)
            try:
                self.figures.append(
                    ("Agreement Heatmap", h2h_obj.plot_agreement_heatmap())
                )
            except Exception as e:
                logger.warning("Agreement plot failed: %s", e)

    def add_reversion_result(self, reversion_result, plots=None):
        """Add reversion analysis results."""
        self.reversion_result = reversion_result
        if plots:
            for fig in plots:
                self.figures.append(("Reversion Analysis", fig))

    def add_note(self, text):
        """Add a free-text note to the report."""
        self.notes.append(text)

    def generate_report(self, output_path=None):
        """Create the full validation report as HTML.

        Parameters
        ----------
        output_path : str or Path or None
            If None, saves to output_dir/validation_report.html.

        Returns
        -------
        Path
            Path to the generated report file.
        """
        if output_path is None:
            output_path = self.output_dir / "validation_report.html"
        output_path = Path(output_path)

        html_parts = [self._html_header()]

        # Summary
        html_parts.append(self._section_summary())

        # Per-cohort results
        for cr in self.cohort_results:
            html_parts.append(self._section_cohort(cr))

        # Head-to-head
        if self.head_to_head_result is not None:
            html_parts.append(self._section_head_to_head())

        # Reversion analysis
        if self.reversion_result is not None:
            html_parts.append(self._section_reversion())

        # Figures
        if self.figures:
            html_parts.append(self._section_figures())

        # Notes
        if self.notes:
            html_parts.append(self._section_notes())

        html_parts.append(self._html_footer())

        html = "\n".join(html_parts)
        output_path.write_text(html)
        logger.info("Report written to %s", output_path)

        # Also save raw metrics as JSON
        self._save_metrics_json()

        return output_path

    def generate_summary_table(self) -> pd.DataFrame:
        """Create a summary DataFrame of AUC across all cohorts.

        Returns
        -------
        pd.DataFrame
            Columns: Cohort, N, AUC, Sensitivity, Specificity, P-value.
        """
        rows = []
        for cr in self.cohort_results:
            m = cr["metrics"]
            bm = m.get("binary_metrics", {})
            dm = m.get("drug_response_metrics", {})
            sm = m.get("survival_metrics", {})

            row = {
                "Cohort": cr["cohort"],
                "N": m.get("n_samples", ""),
                "AUC": bm.get("auc", dm.get("auc", "")),
                "Sensitivity": bm.get("sensitivity", ""),
                "Specificity": bm.get("specificity", ""),
                "MCC": bm.get("mcc", ""),
                "P-value": dm.get("ttest_p", sm.get("logrank_p", "")),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # HTML generation helpers
    # ------------------------------------------------------------------

    def _html_header(self):
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Validation Report: {self.model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1000px; margin: 40px auto; padding: 0 20px;
         color: #333; line-height: 1.6; }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }}
  h2 {{ color: #34495e; margin-top: 2em; }}
  h3 {{ color: #7f8c8d; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .metric-good {{ color: #27ae60; font-weight: bold; }}
  .metric-bad {{ color: #e74c3c; }}
  .note {{ background: #fef9e7; border-left: 4px solid #f39c12;
           padding: 10px 15px; margin: 1em 0; }}
  .interpretation {{ background: #eaf2f8; border-left: 4px solid #3498db;
                     padding: 10px 15px; margin: 1em 0; }}
  img {{ max-width: 100%; height: auto; margin: 1em 0; }}
  .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Validation Report: {self.model_name}</h1>
<p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
"""

    def _html_footer(self):
        return """
</body>
</html>"""

    def _section_summary(self):
        summary_df = self.generate_summary_table()
        return f"""
<h2>Summary</h2>
<p>{len(self.cohort_results)} cohort(s) validated,
{len(self.figures)} figure(s) generated.</p>
{self._df_to_html(summary_df)}
"""

    def _section_cohort(self, cr):
        cohort = cr["cohort"]
        m = cr["metrics"]
        parts = [f"<h2>Cohort: {cohort}</h2>"]
        parts.append(f"<p>N = {m.get('n_samples', '?')}, "
                     f"Endpoint: {m.get('endpoint', '?')}</p>")

        # Binary metrics table
        bm = m.get("binary_metrics", {})
        if bm:
            parts.append("<h3>Binary Classification Metrics</h3>")
            parts.append(self._metrics_table(bm))

        # Drug response
        dm = m.get("drug_response_metrics", {})
        if dm:
            parts.append("<h3>Drug Response Metrics</h3>")
            parts.append(self._metrics_table(dm))

        # Survival
        sm = m.get("survival_metrics", {})
        if sm:
            parts.append("<h3>Survival Metrics</h3>")
            parts.append(self._metrics_table(sm))

        # PARPi7 correlation
        pc = m.get("parpi7_correlation", {})
        if pc:
            parts.append(f"<h3>PARPi7 Correlation</h3>")
            parts.append(f"<p>Pearson r = {pc.get('pearson_r', '?'):.3f}, "
                         f"p = {pc.get('p_value', '?'):.2e}</p>")

        # Per cancer type
        pct = m.get("per_cancer_type", {})
        if pct:
            parts.append("<h3>Per Cancer Type</h3>")
            rows = []
            for ct, ct_m in pct.items():
                rows.append({
                    "Cancer Type": ct,
                    "N": ct_m.get("n", ""),
                    "AUC": f"{ct_m.get('auc', float('nan')):.3f}",
                    "Sensitivity": f"{ct_m.get('sensitivity', float('nan')):.3f}",
                    "Specificity": f"{ct_m.get('specificity', float('nan')):.3f}",
                })
            parts.append(self._df_to_html(pd.DataFrame(rows)))

        return "\n".join(parts)

    def _section_head_to_head(self):
        parts = ["<h2>Head-to-Head Signature Comparison</h2>"]
        if isinstance(self.head_to_head_result, pd.DataFrame):
            parts.append(self._df_to_html(self.head_to_head_result.round(3)))
        return "\n".join(parts)

    def _section_reversion(self):
        parts = ["<h2>Reversion Analysis</h2>"]
        if isinstance(self.reversion_result, dict):
            interp = self.reversion_result.get("interpretation", "")
            if interp:
                parts.append(f'<div class="interpretation"><pre>{interp}</pre></div>')

            tests = self.reversion_result.get("tests", {})
            if tests:
                parts.append("<h3>Statistical Tests</h3>")
                parts.append(self._metrics_table(
                    {k: v for k, v in tests.items() if isinstance(v, dict)}
                ))
        return "\n".join(parts)

    def _section_figures(self):
        parts = ["<h2>Figures</h2>"]
        for title, fig in self.figures:
            parts.append(f"<h3>{title}</h3>")
            parts.append(self._fig_to_html(fig))
        return "\n".join(parts)

    def _section_notes(self):
        parts = ["<h2>Notes</h2>"]
        for note in self.notes:
            parts.append(f'<div class="note">{note}</div>')
        return "\n".join(parts)

    @staticmethod
    def _metrics_table(metrics_dict):
        """Render a dict of metrics as an HTML table."""
        rows = []
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                # Nested dict (e.g. test results): flatten
                for k2, v2 in v.items():
                    rows.append(f"<tr><td>{k}.{k2}</td><td>{_fmt(v2)}</td></tr>")
            elif isinstance(v, (list, pd.Series, np.ndarray)):
                continue  # Skip array values
            else:
                rows.append(f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>")
        return "<table><tr><th>Metric</th><th>Value</th></tr>" + "".join(rows) + "</table>"

    @staticmethod
    def _df_to_html(df):
        """Convert DataFrame to styled HTML table."""
        return df.to_html(index=False, classes="", border=0, na_rep="—",
                          float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))

    @staticmethod
    def _fig_to_html(fig):
        """Embed matplotlib figure as base64 PNG in HTML."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f'<img src="data:image/png;base64,{b64}" />'

    def _save_metrics_json(self):
        """Save all numeric metrics as JSON for programmatic access."""
        data = {
            "model": self.model_name,
            "generated": datetime.now().isoformat(),
            "cohorts": [],
        }
        for cr in self.cohort_results:
            entry = {"cohort": cr["cohort"]}
            for k, v in cr["metrics"].items():
                if isinstance(v, dict):
                    entry[k] = {
                        mk: mv for mk, mv in v.items()
                        if isinstance(mv, (int, float, str, bool))
                    }
                elif isinstance(v, (int, float, str)):
                    entry[k] = v
            data["cohorts"].append(entry)

        if self.head_to_head_result is not None:
            data["head_to_head"] = self.head_to_head_result.to_dict(orient="records")

        json_path = self.output_dir / "validation_metrics.json"
        json_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Metrics JSON saved to %s", json_path)


def _fmt(v):
    """Format a metric value for display."""
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)
