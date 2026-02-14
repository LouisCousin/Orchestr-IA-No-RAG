"""Génération du document DOCX final avec charte graphique configurable."""

import logging
from pathlib import Path
from typing import Optional

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT

from src.core.plan_parser import NormalizedPlan, PlanSection
from src.utils.file_utils import ensure_dir

logger = logging.getLogger("orchestria")


DEFAULT_STYLING = {
    "primary_color": "#F0C441",
    "secondary_color": "#4E4E50",
    "font_title": "Calibri",
    "font_body": "Calibri",
    "font_size_title": 16,
    "font_size_body": 11,
    "margin_top_cm": 2.5,
    "margin_bottom_cm": 2.5,
    "margin_left_cm": 2.5,
    "margin_right_cm": 2.5,
    "logo_path": None,
}


def hex_to_rgb(hex_color: str) -> RGBColor:
    """Convertit une couleur hex en RGBColor."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return RGBColor(r, g, b)


class ExportEngine:
    """Génère le document DOCX final."""

    def __init__(self, styling: Optional[dict] = None):
        self.styling = {**DEFAULT_STYLING, **(styling or {})}

    def export_docx(
        self,
        plan: NormalizedPlan,
        generated_sections: dict,
        output_path: Path,
        project_name: str = "",
    ) -> Path:
        """Génère le document DOCX complet."""
        ensure_dir(output_path.parent)
        doc = Document()

        self._setup_styles(doc)
        self._setup_margins(doc)
        self._add_cover_page(doc, plan, project_name)
        self._add_table_of_contents(doc)
        self._add_sections(doc, plan, generated_sections)

        doc.save(str(output_path))
        logger.info(f"Document DOCX exporté : {output_path}")
        return output_path

    def _setup_margins(self, doc: Document) -> None:
        """Configure les marges du document."""
        for section in doc.sections:
            section.top_margin = Cm(self.styling["margin_top_cm"])
            section.bottom_margin = Cm(self.styling["margin_bottom_cm"])
            section.left_margin = Cm(self.styling["margin_left_cm"])
            section.right_margin = Cm(self.styling["margin_right_cm"])

    def _setup_styles(self, doc: Document) -> None:
        """Configure les styles du document."""
        style = doc.styles["Normal"]
        font = style.font
        font.name = self.styling["font_body"]
        font.size = Pt(self.styling["font_size_body"])
        font.color.rgb = hex_to_rgb(self.styling["secondary_color"])

        # Configurer les styles de titre
        for level in range(1, 4):
            style_name = f"Heading {level}"
            if style_name in doc.styles:
                heading_style = doc.styles[style_name]
                heading_font = heading_style.font
                heading_font.name = self.styling["font_title"]
                heading_font.color.rgb = hex_to_rgb(self.styling["secondary_color"])

                if level == 1:
                    heading_font.size = Pt(self.styling["font_size_title"])
                    heading_font.bold = True
                elif level == 2:
                    heading_font.size = Pt(self.styling["font_size_title"] - 2)
                    heading_font.bold = True
                elif level == 3:
                    heading_font.size = Pt(self.styling["font_size_title"] - 4)

    def _add_cover_page(self, doc: Document, plan: NormalizedPlan, project_name: str) -> None:
        """Ajoute une page de couverture."""
        # Espace vertical
        for _ in range(6):
            doc.add_paragraph("")

        # Logo (si configuré)
        logo_path = self.styling.get("logo_path")
        if logo_path and Path(logo_path).exists():
            try:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(str(logo_path), width=Inches(2))
            except Exception:
                pass

        # Titre du document
        title = plan.title or project_name or "Document"
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title)
        run.font.size = Pt(28)
        run.font.bold = True
        run.font.color.rgb = hex_to_rgb(self.styling["primary_color"])
        run.font.name = self.styling["font_title"]

        # Sous-titre / objectif
        if plan.objective:
            doc.add_paragraph("")
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(plan.objective)
            run.font.size = Pt(14)
            run.font.color.rgb = hex_to_rgb(self.styling["secondary_color"])
            run.font.name = self.styling["font_body"]

        # Ligne décorative
        doc.add_paragraph("")
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("─" * 40)
        run.font.color.rgb = hex_to_rgb(self.styling["primary_color"])

        # Généré par Orchestr'IA
        doc.add_paragraph("")
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("Généré par Orchestr'IA")
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.color.rgb = hex_to_rgb(self.styling["secondary_color"])

        doc.add_page_break()

    def _add_table_of_contents(self, doc: Document) -> None:
        """Ajoute une table des matières."""
        doc.add_heading("Table des matières", level=1)
        doc.add_paragraph(
            "La table des matières sera mise à jour automatiquement "
            "lors de l'ouverture dans Microsoft Word (Ctrl+A puis F9)."
        )

        # Champ TOC Word
        from docx.oxml.ns import qn
        paragraph = doc.add_paragraph()
        run = paragraph.add_run()
        fldChar = run._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "begin"})
        run._r.append(fldChar)

        run2 = paragraph.add_run()
        instrText = run2._r.makeelement(qn("w:instrText"), {})
        instrText.text = ' TOC \\o "1-3" \\h \\z \\u '
        run2._r.append(instrText)

        run3 = paragraph.add_run()
        fldChar2 = run3._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "end"})
        run3._r.append(fldChar2)

        doc.add_page_break()

    def _add_sections(self, doc: Document, plan: NormalizedPlan, generated_sections: dict) -> None:
        """Ajoute les sections générées au document."""
        for section in plan.sections:
            heading_level = min(section.level, 3)  # DOCX supporte 1-9 mais on limite à 3
            doc.add_heading(f"{section.id} {section.title}", level=heading_level)

            content = generated_sections.get(section.id, "")
            if content:
                # Découper en paragraphes et les ajouter
                paragraphs = content.split("\n\n")
                for para_text in paragraphs:
                    para_text = para_text.strip()
                    if not para_text:
                        continue

                    # Détecter les sous-titres dans le contenu généré
                    if para_text.startswith("**") and para_text.endswith("**"):
                        p = doc.add_paragraph()
                        run = p.add_run(para_text.strip("*").strip())
                        run.bold = True
                    elif para_text.startswith("- ") or para_text.startswith("• "):
                        # Listes à puces
                        for line in para_text.split("\n"):
                            line = line.strip().lstrip("-•").strip()
                            if line:
                                doc.add_paragraph(line, style="List Bullet")
                    else:
                        doc.add_paragraph(para_text)
            else:
                p = doc.add_paragraph("[Section non générée]")
                p.runs[0].font.italic = True
                p.runs[0].font.color.rgb = RGBColor(180, 180, 180)

    def export_metadata_excel(
        self,
        plan: NormalizedPlan,
        generated_sections: dict,
        cost_report: dict,
        output_path: Path,
    ) -> Path:
        """Exporte les métadonnées du projet en Excel (Phase 2 complet, base en Phase 1)."""
        import pandas as pd

        ensure_dir(output_path.parent)

        # Feuille : Sections
        sections_data = []
        for s in plan.sections:
            sections_data.append({
                "ID": s.id,
                "Titre": s.title,
                "Niveau": s.level,
                "Budget pages": s.page_budget or "",
                "Statut": s.status,
                "Longueur (car.)": len(generated_sections.get(s.id, "")),
            })

        # Feuille : Coûts
        costs_data = []
        for entry in cost_report.get("entries", []):
            costs_data.append({
                "Section": entry.get("section_id", ""),
                "Modèle": entry.get("model", ""),
                "Tokens input": entry.get("input_tokens", 0),
                "Tokens output": entry.get("output_tokens", 0),
                "Coût (USD)": entry.get("cost_usd", 0),
                "Type": entry.get("task_type", ""),
            })

        with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
            pd.DataFrame(sections_data).to_excel(writer, sheet_name="Sections", index=False)
            if costs_data:
                pd.DataFrame(costs_data).to_excel(writer, sheet_name="Coûts", index=False)

        logger.info(f"Métadonnées exportées : {output_path}")
        return output_path
