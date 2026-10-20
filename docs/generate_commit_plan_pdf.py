"""
Script to generate the comprehensive Git Commit Roadmap & Execution Plan in PDF format.
Uses fpdf2 with modern styling, color accents, structured tables, and page numbers.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from fpdf import FPDF


class ModernPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(110, 120, 135)
            self.cell(0, 7, "EmoBase -- Git Commit Roadmap & Strategic Execution Plan", border=0, align="L")
            self.set_font("Helvetica", "", 8)
            self.cell(0, 7, f"Page {self.page_no()}", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(220, 226, 235)
            self.line(16, 12, 194, 12)
            self.ln(3)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(220, 226, 235)
        self.line(16, self.get_y(), 194, self.get_y())
        self.set_font("Helvetica", "", 8)
        self.set_text_color(130, 140, 155)
        self.cell(0, 8, "Confidential & Proprietary -- EmoBase Multimodal Project Plan", align="L")
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", align="R")

    def section_banner(self, title, subtitle=None):
        self.ln(3)
        self.set_fill_color(30, 41, 59)  # Deep slate navy
        self.rect(self.get_x(), self.get_y(), 178, 9, style="F")
        self.set_xy(self.get_x() + 4, self.get_y() + 1.2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, title.upper(), align="L")
        self.ln(9)
        if subtitle:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(71, 85, 105)
            self.cell(0, 5, subtitle)
            self.ln(6)
        self.set_text_color(30, 41, 59)

    def sub_header(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(14, 116, 144)  # Cyan/Teal accent
        self.cell(0, 6, text, align="L")
        self.ln(6)
        self.set_text_color(30, 41, 59)

    def info_card(self, title, items):
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(203, 213, 225)
        start_y = self.get_y()
        self.rect(16, start_y, 178, 7 + (len(items) * 5.2), style="DF")
        self.set_xy(20, start_y + 1.5)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(30, 41, 59)
        self.cell(0, 5, title)
        self.ln(5.5)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(51, 65, 85)
        for item in items:
            self.set_x(22)
            self.cell(0, 4.8, f"-  {item}")
            self.ln(4.8)
        self.ln(3)

    def commit_table_row(self, day, commit_type, message, files, is_header=False):
        col_w = [18, 26, 74, 60]
        if is_header:
            self.set_fill_color(226, 232, 240)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(15, 23, 42)
            h = 6
        else:
            self.set_fill_color(255, 255, 255)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(30, 41, 59)
            h = 6

        self.cell(col_w[0], h, day, border="B", fill=is_header)
        self.set_font("Helvetica", "B" if is_header else "I", 7.5)
        self.cell(col_w[1], h, commit_type, border="B", fill=is_header)
        self.set_font("Helvetica", "B" if is_header else "", 7.5)
        self.cell(col_w[2], h, message, border="B", fill=is_header)
        self.set_font("Helvetica", "" if is_header else "I", 7)
        self.cell(col_w[3], h, files, border="B", fill=is_header)
        self.ln(h)


def build_plan_pdf(output_path: str):
    pdf = ModernPDF()
    pdf.add_page()

    # Document Header / Hero
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(16, 16, 178, 28, style="F")

    pdf.set_xy(22, 19)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "EmoBase -- Git Commit & Execution Roadmap", align="L")
    pdf.ln(7.5)

    pdf.set_xy(22, 27)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, "Strategic Engineering Plan: Baseline to Production (Starting Today)")
    pdf.ln(5)

    pdf.set_xy(22, 34)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 5, "Target Repository: Multimodal Emotion Recognition (Text + Audio + Face + Context Memory)")
    pdf.ln(13)

    # 1. Executive Summary & Baseline
    pdf.section_banner("1. Current Baseline & Architecture State (Completed Today)")

    pdf.info_card(
        "Current Repository Health Status",
        [
            "Repository Cleaned: Removed 2.73 GB of Google Drive temporary cache (.tmp.driveupload).",
            "Jupyter Notebooks Replaced: All working notebooks converted to modular Python modules in src/.",
            "Git Database Optimized: Total Git repository footprint is under 2.0 MB (ultra-fast clone/push).",
            "Initial Commits Done: 6 atomic commits established covering setup, datasets, extractors, models & docs.",
            "Large Data Safeguards: .gitignore strictly configured against >100MB weights, avoiding GitHub push blocks.",
        ],
    )

    pdf.ln(2)

    # 2. Multi-Day Commit Strategy
    pdf.section_banner("2. Step-by-Step Commit Roadmap From Today Onwards", "Chronological plan divided into 5 strategic phases")

    # Table Header
    pdf.commit_table_row("Day", "Type", "Standardized Commit Message", "Target Files / Scope", is_header=True)

    # Phase 1: Remote Publishing & Data Hosting
    pdf.commit_table_row("Day 1 (Today)", "chore", "publish main branch to GitHub remote", "git push origin main")
    pdf.commit_table_row("Day 1", "docs", "add remote setup instructions and badges", "README.md")
    pdf.commit_table_row("Day 1", "feat(assets)", "add automated Hugging Face weight puller", "scripts/download_assets.py")

    # Phase 2: Testing & CI/CD
    pdf.commit_table_row("Day 2", "test", "add unit tests for cross-attention & GRU memory", "tests/test_models.py")
    pdf.commit_table_row("Day 2", "test", "add synthetic test fixtures for audio & face shapes", "tests/conftest.py")
    pdf.commit_table_row("Day 3", "ci", "create GitHub Actions CI workflow for test matrix", ".github/workflows/ci.yml")
    pdf.commit_table_row("Day 3", "style", "enforce ruff/black code formatting and linting", "pyproject.toml")

    # Phase 3: Experiment Tracking & Ablation Studies
    pdf.commit_table_row("Day 4", "feat(wandb)", "integrate Weights & Biases experiment logging", "src/fusion_layer2.py")
    pdf.commit_table_row("Day 4", "feat(train)", "add early stopping and cosine annealing scheduler", "src/fusion_layer2.py")
    pdf.commit_table_row("Day 5", "feat(ablation)", "implement unimodal ablation benchmark script", "scripts/run_ablations.py")
    pdf.commit_table_row("Day 5", "docs", "record unimodal vs multimodal ablation metrics", "results/results.txt")

    # Phase 4: Real-Time Multimodal Inference & Web Demo
    pdf.commit_table_row("Day 6", "feat(infer)", "add end-to-end multimodal inference pipeline", "src/infer.py")
    pdf.commit_table_row("Day 7", "feat(demo)", "build interactive Gradio web application", "app.py")
    pdf.commit_table_row("Day 7", "feat(ui)", "add radar chart visualization for 7 emotions", "app.py, requirements.txt")

    # Phase 5: Containerization & v1.0.0 Release
    pdf.commit_table_row("Day 8", "build(docker)", "add multi-stage Dockerfile and docker-compose", "Dockerfile, .dockerignore")
    pdf.commit_table_row("Day 9", "docs", "complete comprehensive API docs and video guide", "docs/API.md, README.md")
    pdf.commit_table_row("Day 10", "release", "tag v1.0.0 release and attach binary checkpoints", "CHANGELOG.md, Git Tag v1.0.0")

    pdf.ln(5)

    # 3. Strategy for Managing Large Data (>100MB)
    pdf.section_banner("3. Technical Strategy: Large Data & Model Weights Handling")

    pdf.info_card(
        "Standard Machine Learning Big Data Protocol",
        [
            "The 100MB GitHub Boundary: Any file >100MB is automatically rejected by GitHub servers.",
            "Decoupled Storage: Keep code in Git (<5MB) and host heavy checkpoints (360-500MB) externally.",
            "Option A (Hugging Face Hub): Free, limitless model hosting; clone weights via snapshot_download().",
            "Option B (GitHub Releases): Up to 2.0 GB per asset; upload model.safetensors directly to v1.0.0 release tag.",
            "Option C (Git LFS): git lfs track '*.safetensors' (note: GitHub free tier limited to 2GB total LFS bandwidth).",
            "Automated Onboarding: Anyone cloning your repo runs 'python scripts/download_assets.py' to get weights.",
        ],
    )

    pdf.ln(4)

    # 4. Conventional Commits Standard Guide
    pdf.section_banner("4. Conventional Commits Standardization & Quality Checklist")

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    rules = (
        "1. feat: A new feature or capability (e.g., feat(models): implement cross attention block)\n"
        "2. fix: A bug fix (e.g., fix(audio): resolve sample rate mismatch in librosa loader)\n"
        "3. refactor: Code restructuring without changing functional behavior (e.g., refactor: convert ipynb to py)\n"
        "4. docs: Documentation updates only (e.g., docs: update benchmark metrics in README)\n"
        "5. test: Adding or correcting tests (e.g., test: add test case for tensor batch alignment)\n"
        "6. chore: Tooling, dependency, or configuration updates (e.g., chore: bump transformers dependency)\n"
        "7. ci: Continuous integration workflows (e.g., ci: configure GitHub Actions test runner)"
    )
    pdf.multi_cell(178, 5, rules)

    pdf.ln(3)

    pdf.info_card(
        "Immediate Action Checklist For Today",
        [
            "Step 1: Create an empty repository on GitHub named 'EmoBase' (do not initialize with README).",
            "Step 2: Link your remote: git remote add origin https://github.com/Vaibhav-801/EmoBase.git",
            "Step 3: Push all 6 clean commits: git branch -M main && git push -u origin main",
            "Step 4: Verify your clean repository is live with the full README and documentation.",
        ],
    )

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    print(f"Plan PDF successfully created at: {output_path}")


if __name__ == "__main__":
    out_file = r"C:\EmoBase\docs\EmoBase_Git_Commit_Roadmap_and_Execution_Plan.pdf"
    build_plan_pdf(out_file)
