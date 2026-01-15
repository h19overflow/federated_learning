# Module 5: Report Generation

**Agent Assignment:** backend-logic-architect
**Priority:** P2 (Start after core modules work)
**Dependencies:** Module 1 (database), matplotlib/reportlab
**Estimated Effort:** 1-2 days

---

## Purpose

Generate comprehensive PDF research reports summarizing the autonomous research session with visualizations, insights, and recommendations.

---

## File Structure

```
federated_pneumonia_detection/src/control/agentic_systems/research_assistant/report_generation/
├── __init__.py
├── research_report.py          # Main report generator
├── visualizations.py           # Charts and graphs
├── insights_analyzer.py        # Extract insights from experiments
└── templates/
    └── report_template.html    # HTML template for PDF
```

---

## Report Generator (research_report.py)

```python
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from typing import List, Dict, Any
import io

from ..knowledge_base.database import ExperimentDatabase
from ..knowledge_base.schemas import ExperimentRun, ResearchSession
from .visualizations import ResearchVisualizations
from .insights_analyzer import InsightsAnalyzer

class ResearchReportGenerator:
    """
    Generate comprehensive PDF research reports

    Report structure:
    1. Executive Summary
    2. Research Configuration
    3. Experiment Timeline
    4. Performance Analysis (centralized vs federated)
    5. Hyperparameter Impact Analysis
    6. Key Insights and Recommendations
    7. Appendix (all experiments table)
    """

    def __init__(self, db: ExperimentDatabase):
        self.db = db
        self.visualizations = ResearchVisualizations()
        self.insights_analyzer = InsightsAnalyzer()
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30
        )
        self.section_style = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3b82f6'),
            spaceAfter=12
        )

    def generate_report(self, session_id: int, output_path: str):
        """
        Generate full research report PDF

        Args:
            session_id: Research session to report on
            output_path: Where to save PDF
        """

        # Fetch data
        session = self.db.get_session_by_id(session_id)
        experiments = self.db.get_all_experiments(session_id)
        summary = self.db.get_session_summary(session_id)

        # Create PDF
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # 1. Title Page
        story.extend(self._build_title_page(session, summary))
        story.append(PageBreak())

        # 2. Executive Summary
        story.extend(self._build_executive_summary(session, summary, experiments))
        story.append(Spacer(1, 0.3 * inch))

        # 3. Research Configuration
        story.extend(self._build_configuration_section(session))
        story.append(Spacer(1, 0.3 * inch))

        # 4. Experiment Timeline
        story.extend(self._build_timeline_section(experiments))
        story.append(PageBreak())

        # 5. Performance Analysis
        story.extend(self._build_performance_analysis(experiments))
        story.append(PageBreak())

        # 6. Hyperparameter Impact
        story.extend(self._build_hyperparameter_analysis(experiments))
        story.append(PageBreak())

        # 7. Key Insights
        story.extend(self._build_insights_section(experiments))
        story.append(PageBreak())

        # 8. Appendix: Full Experiment Table
        story.extend(self._build_appendix(experiments))

        # Build PDF
        doc.build(story)
        print(f"📄 Report saved to: {output_path}")

    def _build_title_page(self, session: ResearchSession, summary: dict) -> List:
        """Title page with project info"""
        story = []

        # Title
        story.append(Paragraph(
            "Autonomous Research Assistant Report",
            self.title_style
        ))
        story.append(Spacer(1, 0.5 * inch))

        # Session info
        info_text = f"""
        <b>Session ID:</b> {session.id}<br/>
        <b>Start Time:</b> {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>End Time:</b> {session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'In Progress'}<br/>
        <b>Total Experiments:</b> {summary['total_experiments']}<br/>
        <b>Stopping Reason:</b> {session.stopping_reason or 'N/A'}
        """
        story.append(Paragraph(info_text, self.styles['Normal']))
        story.append(Spacer(1, 0.5 * inch))

        # Best results
        best_text = f"""
        <b>Best Results:</b><br/>
        Centralized Recall: {summary['best_centralized_recall']:.3f}<br/>
        Federated Recall: {summary['best_federated_recall']:.3f}
        """
        story.append(Paragraph(best_text, self.styles['Normal']))

        return story

    def _build_executive_summary(
        self,
        session: ResearchSession,
        summary: dict,
        experiments: List[ExperimentRun]
    ) -> List:
        """Executive summary with key findings"""
        story = []

        story.append(Paragraph("Executive Summary", self.section_style))

        # Generate insights
        insights = self.insights_analyzer.generate_executive_summary(experiments, summary)

        summary_text = f"""
        This research session explored the hyperparameter space for pneumonia detection
        using both centralized and federated learning paradigms. The autonomous agent
        conducted {summary['total_experiments']} experiments over
        {self._format_duration(session.start_time, session.end_time)}.<br/><br/>

        <b>Key Achievements:</b><br/>
        • Best centralized recall: {summary['best_centralized_recall']:.3f}<br/>
        • Best federated recall: {summary['best_federated_recall']:.3f}<br/>
        • Average training time: {summary['avg_training_time']:.1f} seconds<br/>
        • Failed experiments: {summary.get('failed_experiments', 0)}<br/><br/>

        <b>Primary Findings:</b><br/>
        {self._format_insights_as_bullets(insights)}
        """

        story.append(Paragraph(summary_text, self.styles['Normal']))

        return story

    def _build_performance_analysis(self, experiments: List[ExperimentRun]) -> List:
        """Performance comparison with visualizations"""
        story = []

        story.append(Paragraph("Performance Analysis", self.section_style))

        # Generate visualizations
        recall_chart = self.visualizations.generate_recall_over_time(experiments)
        paradigm_comparison = self.visualizations.generate_paradigm_comparison(experiments)

        # Add charts
        story.append(Image(recall_chart, width=5*inch, height=3*inch))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Image(paradigm_comparison, width=5*inch, height=3*inch))

        return story

    def _build_hyperparameter_analysis(self, experiments: List[ExperimentRun]) -> List:
        """Hyperparameter impact analysis"""
        story = []

        story.append(Paragraph("Hyperparameter Impact Analysis", self.section_style))

        # Generate heatmap
        heatmap = self.visualizations.generate_hyperparameter_heatmap(experiments)
        story.append(Image(heatmap, width=6*inch, height=4*inch))

        # Add analysis text
        hp_insights = self.insights_analyzer.analyze_hyperparameter_impact(experiments)
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(hp_insights, self.styles['Normal']))

        return story

    def _build_insights_section(self, experiments: List[ExperimentRun]) -> List:
        """Key insights and recommendations"""
        story = []

        story.append(Paragraph("Key Insights & Recommendations", self.section_style))

        insights = self.insights_analyzer.generate_detailed_insights(experiments)

        for category, findings in insights.items():
            story.append(Paragraph(f"<b>{category}</b>", self.styles['Heading3']))
            for finding in findings:
                story.append(Paragraph(f"• {finding}", self.styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

        return story

    def _build_appendix(self, experiments: List[ExperimentRun]) -> List:
        """Full experiment table"""
        story = []

        story.append(Paragraph("Appendix: All Experiments", self.section_style))

        # Build table data
        table_data = [['#', 'Paradigm', 'LR', 'Batch', 'Dropout', 'Recall', 'Accuracy', 'Status']]

        for exp in experiments:
            table_data.append([
                str(exp.experiment_number),
                exp.paradigm,
                str(exp.hyperparameters.get('learning_rate', '-')),
                str(exp.hyperparameters.get('batch_size', '-')),
                str(exp.hyperparameters.get('dropout_rate', '-')),
                f"{exp.metrics.get('recall', 0):.3f}" if exp.metrics else '-',
                f"{exp.metrics.get('accuracy', 0):.3f}" if exp.metrics else '-',
                exp.status
            ])

        # Create table
        table = Table(table_data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])

        story.append(table)

        return story

    def _format_duration(self, start, end) -> str:
        """Format duration as human-readable string"""
        if not end:
            return "In Progress"
        duration = (end - start).total_seconds()
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        return f"{hours}h {minutes}m"

    def _format_insights_as_bullets(self, insights: List[str]) -> str:
        """Format insights list as HTML bullets"""
        return "<br/>".join([f"• {insight}" for insight in insights])
```

---

## Visualizations (visualizations.py)

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import seaborn as sns
import numpy as np
import io
from typing import List
from ..knowledge_base.schemas import ExperimentRun

class ResearchVisualizations:
    """Generate charts and graphs for research reports"""

    def __init__(self):
        sns.set_style("whitegrid")
        self.colors = {
            'centralized': '#3b82f6',
            'federated': '#10b981'
        }

    def generate_recall_over_time(self, experiments: List[ExperimentRun]) -> io.BytesIO:
        """
        Line chart: Recall over experiment number

        Shows both centralized and federated on same plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separate by paradigm
        cent_exps = [e for e in experiments if e.paradigm == 'centralized' and e.metrics]
        fed_exps = [e for e in experiments if e.paradigm == 'federated' and e.metrics]

        # Plot centralized
        cent_x = [e.experiment_number for e in cent_exps]
        cent_y = [e.metrics['recall'] for e in cent_exps]
        ax.plot(cent_x, cent_y, 'o-', color=self.colors['centralized'], label='Centralized', linewidth=2)

        # Plot federated
        fed_x = [e.experiment_number for e in fed_exps]
        fed_y = [e.metrics['recall'] for e in fed_exps]
        ax.plot(fed_x, fed_y, 's-', color=self.colors['federated'], label='Federated', linewidth=2)

        ax.set_xlabel('Experiment Number', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('Recall Over Experiment Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def generate_paradigm_comparison(self, experiments: List[ExperimentRun]) -> io.BytesIO:
        """
        Box plot: Recall distribution for centralized vs federated
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        cent_recalls = [e.metrics['recall'] for e in experiments if e.paradigm == 'centralized' and e.metrics]
        fed_recalls = [e.metrics['recall'] for e in experiments if e.paradigm == 'federated' and e.metrics]

        data = [cent_recalls, fed_recalls]
        labels = ['Centralized', 'Federated']

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['centralized'])
        bp['boxes'][1].set_facecolor(self.colors['federated'])

        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('Recall Distribution: Centralized vs Federated', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def generate_hyperparameter_heatmap(self, experiments: List[ExperimentRun]) -> io.BytesIO:
        """
        Heatmap: Learning rate vs batch size with recall as color
        """
        # Extract data
        lrs = []
        batches = []
        recalls = []

        for exp in experiments:
            if exp.metrics and 'recall' in exp.metrics:
                lrs.append(exp.hyperparameters.get('learning_rate', 0))
                batches.append(exp.hyperparameters.get('batch_size', 0))
                recalls.append(exp.metrics['recall'])

        # Create heatmap (simplified - in production use proper binning)
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(lrs, batches, c=recalls, s=200, cmap='RdYlGn', vmin=0.85, vmax=0.95, edgecolors='black')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Recall', fontsize=12)

        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Batch Size', fontsize=12)
        ax.set_title('Hyperparameter Space Exploration', fontsize=14, fontweight='bold')
        ax.set_xscale('log')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf
```

---

## Insights Analyzer (insights_analyzer.py)

```python
from typing import List, Dict
from ..knowledge_base.schemas import ExperimentRun
import numpy as np

class InsightsAnalyzer:
    """Extract insights from experiment results"""

    def generate_executive_summary(
        self,
        experiments: List[ExperimentRun],
        summary: dict
    ) -> List[str]:
        """Generate 3-5 key insights for executive summary"""

        insights = []

        # Paradigm comparison
        if summary['best_centralized_recall'] > summary['best_federated_recall']:
            diff = (summary['best_centralized_recall'] - summary['best_federated_recall']) * 100
            insights.append(
                f"Centralized learning outperformed federated by {diff:.1f}% in best recall"
            )
        else:
            insights.append(
                "Federated learning achieved competitive performance with centralized"
            )

        # Learning rate insight
        lr_insight = self._analyze_learning_rate_trend(experiments)
        if lr_insight:
            insights.append(lr_insight)

        # Convergence insight
        if summary['total_experiments'] >= 20:
            insights.append(
                f"Agent efficiently explored {summary['total_experiments']} configurations, "
                "achieving convergence through intelligent Bayesian-inspired search"
            )

        return insights

    def analyze_hyperparameter_impact(self, experiments: List[ExperimentRun]) -> str:
        """Analyze which hyperparameters had the biggest impact"""

        # Calculate correlation between each hyperparameter and recall
        hp_impact = {}

        for hp_name in ['learning_rate', 'batch_size', 'dropout_rate']:
            values = []
            recalls = []

            for exp in experiments:
                if exp.metrics and hp_name in exp.hyperparameters:
                    values.append(exp.hyperparameters[hp_name])
                    recalls.append(exp.metrics['recall'])

            if len(values) > 3:
                corr = np.corrcoef(values, recalls)[0, 1]
                hp_impact[hp_name] = abs(corr)

        # Sort by impact
        sorted_hp = sorted(hp_impact.items(), key=lambda x: x[1], reverse=True)

        analysis = "<b>Hyperparameter Impact Ranking:</b><br/><br/>"
        for hp, impact in sorted_hp:
            analysis += f"• <b>{hp}</b>: {impact:.2f} correlation with recall<br/>"

        return analysis

    def generate_detailed_insights(self, experiments: List[ExperimentRun]) -> Dict[str, List[str]]:
        """Generate categorized insights"""

        insights = {
            "Performance Trends": [],
            "Hyperparameter Findings": [],
            "Paradigm Comparison": [],
            "Recommendations": []
        }

        # Performance trends
        completed = [e for e in experiments if e.status == 'completed' and e.metrics]
        if completed:
            avg_recall = np.mean([e.metrics['recall'] for e in completed])
            insights["Performance Trends"].append(
                f"Average recall across all experiments: {avg_recall:.3f}"
            )

        # Hyperparameter findings
        lr_finding = self._analyze_learning_rate_trend(experiments)
        if lr_finding:
            insights["Hyperparameter Findings"].append(lr_finding)

        # Recommendations
        best_exp = max(completed, key=lambda e: e.metrics['recall'])
        insights["Recommendations"].append(
            f"For production deployment, use: LR={best_exp.hyperparameters.get('learning_rate')}, "
            f"Batch={best_exp.hyperparameters.get('batch_size')}, "
            f"Dropout={best_exp.hyperparameters.get('dropout_rate')}"
        )

        return insights

    def _analyze_learning_rate_trend(self, experiments: List[ExperimentRun]) -> str:
        """Analyze learning rate impact"""
        # Group by LR and calculate avg recall
        lr_groups = {}

        for exp in experiments:
            if exp.metrics and 'learning_rate' in exp.hyperparameters:
                lr = exp.hyperparameters['learning_rate']
                if lr not in lr_groups:
                    lr_groups[lr] = []
                lr_groups[lr].append(exp.metrics['recall'])

        if len(lr_groups) < 2:
            return None

        # Find best LR
        lr_avgs = {lr: np.mean(recalls) for lr, recalls in lr_groups.items()}
        best_lr = max(lr_avgs, key=lr_avgs.get)

        return f"Learning rate {best_lr} achieved the highest average recall ({lr_avgs[best_lr]:.3f})"
```

---

## Acceptance Criteria

- ✅ PDF report generated successfully
- ✅ Report includes all required sections
- ✅ Visualizations render correctly
- ✅ Insights are data-driven and accurate
- ✅ Report is professionally formatted
- ✅ Can be generated for any completed session
- ✅ Handles edge cases (few experiments, all failed, etc.)

---

**Status:** Ready for Implementation
**Blocked By:** Module 1 (database)
**Blocks:** None (optional enhancement)
