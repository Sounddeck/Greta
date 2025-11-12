"""
File Generation Service

Provides capabilities for the Master Agent to generate various file formats:
- PDF reports
- Excel/CSV spreadsheets
- Graphics and visualizations
- Other structured file formats
"""

import logging
import io
import os
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Excel/CSV generation
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
import csv

# Graphics generation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Image processing
from PIL import Image

logger = logging.getLogger(__name__)


class FileGenerationService:
    """Service for generating various file formats"""
    
    def __init__(self):
        self.output_dir = Path("generated_files")
        self.output_dir.mkdir(exist_ok=True)
        logger.info("✅ File Generation Service initialized")
    
    # ==================== PDF GENERATION ====================
    
    def generate_pdf_report(
        self,
        title: str,
        content: List[Dict[str, Any]],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a PDF report
        
        Args:
            title: Report title
            content: List of content sections, each with 'type' and 'data'
                    Types: 'heading', 'paragraph', 'table', 'spacer', 'page_break'
            filename: Optional custom filename
            metadata: Optional metadata (author, subject, etc.)
        
        Returns:
            Dictionary with file path, base64 data, and metadata
        """
        try:
            # Generate filename
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}.pdf"
            
            filepath = self.output_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Add title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Add metadata if provided
            if metadata:
                meta_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if metadata.get('author'):
                    meta_text += f" | Author: {metadata['author']}"
                story.append(Paragraph(meta_text, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
            
            # Process content sections
            for section in content:
                section_type = section.get('type', 'paragraph')
                data = section.get('data', '')
                
                if section_type == 'heading':
                    story.append(Paragraph(data, styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                
                elif section_type == 'paragraph':
                    story.append(Paragraph(data, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
                
                elif section_type == 'table':
                    # data should be a list of lists
                    # Apply Edward Tufte principles for tables:
                    # - Minimal borders (only horizontal rules)
                    # - No background colors
                    # - Right-align numbers, left-align text
                    # - Clear typography
                    # - Generous white space
                    table = Table(data)

                    # Determine which columns contain numbers for right-alignment
                    num_cols = len(data[0]) if data else 0

                    # Base style with Tufte principles
                    table_style = [
                        # Header styling - minimal, clear
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1A202C')),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('TOPPADDING', (0, 0), (-1, 0), 8),

                        # Data rows styling
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2D3748')),
                        ('TOPPADDING', (0, 1), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),

                        # Minimal borders - only horizontal rules
                        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#2D3748')),  # Header bottom
                        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),  # Table bottom

                        # Left-align text by default
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),

                        # No background colors (white background)
                        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                    ]

                    # Right-align numeric columns (detect by checking if first data row contains numbers)
                    if len(data) > 1:
                        for col_idx in range(num_cols):
                            try:
                                # Try to convert to float - if successful, it's a number column
                                float(str(data[1][col_idx]).replace(',', '').replace('$', ''))
                                table_style.append(('ALIGN', (col_idx, 1), (col_idx, -1), 'RIGHT'))
                            except (ValueError, IndexError):
                                # Not a number, keep left-aligned
                                pass

                    table.setStyle(TableStyle(table_style))
                    story.append(table)
                    story.append(Spacer(1, 0.3*inch))  # Generous white space
                
                elif section_type == 'spacer':
                    height = section.get('height', 0.2)
                    story.append(Spacer(1, height*inch))
                
                elif section_type == 'page_break':
                    story.append(PageBreak())
            
            # Build PDF
            doc.build(story)
            
            # Read file and convert to base64
            with open(filepath, 'rb') as f:
                pdf_data = f.read()
                base64_data = base64.b64encode(pdf_data).decode('utf-8')

            # Get absolute file path for logging
            absolute_path = os.path.abspath(filepath)

            logger.info(f"✅ PDF report generated: {filename}")
            logger.info(f"📁 File location: {absolute_path}")
            logger.info(f"📂 Storage directory: {os.path.dirname(absolute_path)}")
            logger.info(f"📊 File size: {len(pdf_data)} bytes ({len(pdf_data)/1024:.2f} KB)")

            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath),
                "absolute_path": absolute_path,
                "storage_directory": os.path.dirname(absolute_path),
                "base64_data": base64_data,
                "file_size": len(pdf_data),
                "mime_type": "application/pdf",
                "download_url": f"/api/v1/files/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate PDF: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== EXCEL GENERATION ====================
    
    def generate_excel_spreadsheet(
        self,
        sheets: List[Dict[str, Any]],
        filename: Optional[str] = None,
        include_charts: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an Excel spreadsheet
        
        Args:
            sheets: List of sheet definitions, each with 'name', 'data', 'headers'
            filename: Optional custom filename
            include_charts: Whether to include charts
        
        Returns:
            Dictionary with file path, base64 data, and metadata
        """
        try:
            # Generate filename
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"spreadsheet_{timestamp}.xlsx"
            
            filepath = self.output_dir / filename
            
            # Create workbook
            wb = openpyxl.Workbook()
            wb.remove(wb.active)  # Remove default sheet

            # Define Tufte-inspired styles:
            # - Minimal borders (only bottom border for header)
            # - No background colors
            # - Clear typography
            # - Right-align numbers, left-align text

            header_font = Font(bold=True, color="1A202C", size=11)
            header_alignment = Alignment(horizontal="left", vertical="center")

            # Minimal border - only bottom line for header
            thin_bottom_border = Border(
                bottom=Side(style='thin', color='2D3748')
            )

            # Process each sheet
            for sheet_def in sheets:
                sheet_name = sheet_def.get('name', 'Sheet1')
                headers = sheet_def.get('headers', [])
                data = sheet_def.get('data', [])

                ws = wb.create_sheet(title=sheet_name)

                # Add headers with Tufte styling
                if headers:
                    for col_idx, header in enumerate(headers, 1):
                        cell = ws.cell(row=1, column=col_idx, value=header)
                        cell.font = header_font
                        cell.alignment = header_alignment
                        cell.border = thin_bottom_border
                        # No background fill - keep white

                # Add data
                for row_idx, row_data in enumerate(data, 2):
                    for col_idx, value in enumerate(row_data, 1):
                        cell = ws.cell(row=row_idx, column=col_idx, value=value)

                        # Apply Tufte principle: Right-align numbers, left-align text
                        if isinstance(value, (int, float)):
                            cell.alignment = Alignment(horizontal="right", vertical="center")
                        else:
                            # Try to detect if it's a number string
                            try:
                                float(str(value).replace(',', '').replace('$', ''))
                                cell.alignment = Alignment(horizontal="right", vertical="center")
                            except (ValueError, AttributeError):
                                cell.alignment = Alignment(horizontal="left", vertical="center")

                        # No borders on data cells (Tufte principle)
                        cell.border = Border()

                # Auto-adjust column widths with generous spacing
                for column in ws.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except:
                            pass
                    # Add extra padding for readability (Tufte: generous white space)
                    adjusted_width = min(max_length + 4, 50)
                    ws.column_dimensions[column_letter].width = adjusted_width
            
            # Save workbook
            wb.save(filepath)
            
            # Read file and convert to base64
            with open(filepath, 'rb') as f:
                excel_data = f.read()
                base64_data = base64.b64encode(excel_data).decode('utf-8')

            # Get absolute file path for logging
            absolute_path = os.path.abspath(filepath)

            logger.info(f"✅ Excel spreadsheet generated: {filename}")
            logger.info(f"📁 File location: {absolute_path}")
            logger.info(f"📂 Storage directory: {os.path.dirname(absolute_path)}")
            logger.info(f"📊 File size: {len(excel_data)} bytes ({len(excel_data)/1024:.2f} KB)")

            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath),
                "absolute_path": absolute_path,
                "storage_directory": os.path.dirname(absolute_path),
                "base64_data": base64_data,
                "file_size": len(excel_data),
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "download_url": f"/api/v1/files/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Excel: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== CSV GENERATION ====================
    
    def generate_csv_file(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a CSV file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_{timestamp}.csv"
            
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            
            # Read and encode
            with open(filepath, 'rb') as f:
                csv_data = f.read()
                base64_data = base64.b64encode(csv_data).decode('utf-8')

            # Get absolute file path for logging
            absolute_path = os.path.abspath(filepath)

            logger.info(f"✅ CSV file generated: {filename}")
            logger.info(f"📁 File location: {absolute_path}")
            logger.info(f"📂 Storage directory: {os.path.dirname(absolute_path)}")
            logger.info(f"📊 File size: {len(csv_data)} bytes ({len(csv_data)/1024:.2f} KB)")

            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath),
                "absolute_path": absolute_path,
                "storage_directory": os.path.dirname(absolute_path),
                "base64_data": base64_data,
                "file_size": len(csv_data),
                "mime_type": "text/csv",
                "download_url": f"/api/v1/files/download/{filename}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate CSV: {e}")
            return {"success": False, "error": str(e)}

    # ==================== GRAPHICS GENERATION ====================

    def generate_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        filename: Optional[str] = None,
        title: Optional[str] = None,
        use_plotly: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a chart/visualization

        Args:
            chart_type: Type of chart ('bar', 'line', 'pie', 'scatter', 'histogram')
            data: Chart data (format depends on chart_type)
            filename: Optional custom filename
            title: Chart title
            use_plotly: Use Plotly instead of Matplotlib (creates interactive HTML)

        Returns:
            Dictionary with file path, base64 data, and metadata
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ext = 'html' if use_plotly else 'png'
                filename = f"chart_{timestamp}.{ext}"

            filepath = self.output_dir / filename

            if use_plotly:
                result = self._generate_plotly_chart(chart_type, data, filepath, title)
            else:
                result = self._generate_matplotlib_chart(chart_type, data, filepath, title)

            if result["success"]:
                # Get absolute file path for logging
                absolute_path = os.path.abspath(filepath)

                logger.info(f"✅ Chart generated: {filename}")
                logger.info(f"📁 File location: {absolute_path}")
                logger.info(f"📂 Storage directory: {os.path.dirname(absolute_path)}")
                if 'file_size' in result:
                    logger.info(f"📊 File size: {result['file_size']} bytes ({result['file_size']/1024:.2f} KB)")

                # Add additional metadata to result
                result["absolute_path"] = absolute_path
                result["storage_directory"] = os.path.dirname(absolute_path)
                result["download_url"] = f"/api/v1/files/download/{filename}"

            return result

        except Exception as e:
            logger.error(f"❌ Failed to generate chart: {e}")
            return {"success": False, "error": str(e)}

    def _generate_matplotlib_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        filepath: Path,
        title: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate chart using Matplotlib with Edward Tufte design principles:
        - High data-ink ratio (minimize chartjunk)
        - Clear typography
        - Direct labeling where possible
        - Minimal grid lines
        - No 3D effects or unnecessary decorations
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Apply Tufte principles: Remove chartjunk
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('#333333')

            # Set background to white (no gray backgrounds)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            if chart_type == 'bar':
                x = data.get('x', [])
                y = data.get('y', [])
                # Use subtle color, no borders
                bars = ax.bar(x, y, color='#4A5568', edgecolor='none', width=0.6)
                ax.set_xlabel(data.get('x_label', 'X'), fontsize=11, color='#333333')
                ax.set_ylabel(data.get('y_label', 'Y'), fontsize=11, color='#333333')

                # Direct labeling: Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9, color='#333333')

                # Minimal grid (only horizontal, subtle)
                ax.grid(axis='y', alpha=0.2, linewidth=0.5, linestyle='-', color='#CCCCCC')
                ax.set_axisbelow(True)

            elif chart_type == 'line':
                x = data.get('x', [])
                y = data.get('y', [])
                # Simple line, no heavy markers
                ax.plot(x, y, color='#2D3748', linewidth=1.5, marker='o',
                       markersize=4, markerfacecolor='#2D3748', markeredgewidth=0)
                ax.set_xlabel(data.get('x_label', 'X'), fontsize=11, color='#333333')
                ax.set_ylabel(data.get('y_label', 'Y'), fontsize=11, color='#333333')

                # Minimal grid (only horizontal, subtle)
                ax.grid(axis='y', alpha=0.2, linewidth=0.5, linestyle='-', color='#CCCCCC')
                ax.set_axisbelow(True)

            elif chart_type == 'pie':
                # Pie charts are generally discouraged by Tufte, but if needed:
                labels = data.get('labels', [])
                values = data.get('values', [])
                # Use grayscale or minimal colors, no 3D
                colors_tufte = ['#2D3748', '#4A5568', '#718096', '#A0AEC0', '#CBD5E0']
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                      colors=colors_tufte[:len(values)], textprops={'fontsize': 10, 'color': '#333333'})
                ax.axis('equal')

            elif chart_type == 'scatter':
                x = data.get('x', [])
                y = data.get('y', [])
                # Subtle points, no heavy styling
                ax.scatter(x, y, color='#4A5568', alpha=0.6, s=50, edgecolors='none')
                ax.set_xlabel(data.get('x_label', 'X'), fontsize=11, color='#333333')
                ax.set_ylabel(data.get('y_label', 'Y'), fontsize=11, color='#333333')

                # Minimal grid
                ax.grid(alpha=0.2, linewidth=0.5, linestyle='-', color='#CCCCCC')
                ax.set_axisbelow(True)

            elif chart_type == 'histogram':
                values = data.get('values', [])
                bins = data.get('bins', 10)
                # Subtle bars, minimal borders
                ax.hist(values, bins=bins, color='#4A5568', alpha=0.8, edgecolor='#2D3748', linewidth=0.5)
                ax.set_xlabel(data.get('x_label', 'Value'), fontsize=11, color='#333333')
                ax.set_ylabel(data.get('y_label', 'Frequency'), fontsize=11, color='#333333')

                # Minimal grid
                ax.grid(axis='y', alpha=0.2, linewidth=0.5, linestyle='-', color='#CCCCCC')
                ax.set_axisbelow(True)

            # Title with clear typography (if provided)
            if title:
                ax.set_title(title, fontsize=14, fontweight='normal', color='#1A202C', pad=20)

            # Set tick label colors
            ax.tick_params(colors='#333333', labelsize=10)

            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            # Read and encode
            with open(filepath, 'rb') as f:
                img_data = f.read()
                base64_data = base64.b64encode(img_data).decode('utf-8')

            return {
                "success": True,
                "filename": filepath.name,
                "filepath": str(filepath),
                "base64_data": base64_data,
                "file_size": len(img_data),
                "mime_type": "image/png"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_plotly_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        filepath: Path,
        title: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate interactive chart using Plotly with Edward Tufte design principles:
        - Minimal styling and decorations
        - High data-ink ratio
        - Clear typography
        - No unnecessary gridlines or backgrounds
        """
        try:
            # Tufte-inspired color palette (grayscale)
            tufte_color = '#4A5568'
            tufte_colors = ['#2D3748', '#4A5568', '#718096', '#A0AEC0', '#CBD5E0']

            if chart_type == 'bar':
                fig = go.Figure(data=[go.Bar(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    marker_color=tufte_color,
                    marker_line_width=0,  # No borders
                    text=data.get('y', []),  # Direct labeling
                    textposition='outside',
                    textfont=dict(size=10, color='#333333')
                )])

            elif chart_type == 'line':
                fig = go.Figure(data=[go.Scatter(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    mode='lines+markers',
                    line=dict(color=tufte_color, width=1.5),
                    marker=dict(size=4, color=tufte_color)
                )])

            elif chart_type == 'pie':
                fig = go.Figure(data=[go.Pie(
                    labels=data.get('labels', []),
                    values=data.get('values', []),
                    marker=dict(colors=tufte_colors, line=dict(width=0)),
                    textfont=dict(size=11, color='#333333')
                )])

            elif chart_type == 'scatter':
                fig = go.Figure(data=[go.Scatter(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    mode='markers',
                    marker=dict(color=tufte_color, size=8, opacity=0.6, line=dict(width=0))
                )])

            # Apply Tufte layout principles
            fig.update_layout(
                title=dict(
                    text=title if title else '',
                    font=dict(size=14, color='#1A202C', family='Arial'),
                    x=0.5,
                    xanchor='center'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial', size=11, color='#333333'),
                xaxis=dict(
                    showgrid=False,
                    showline=True,
                    linewidth=0.5,
                    linecolor='#333333',
                    tickfont=dict(size=10, color='#333333'),
                    title_font=dict(size=11, color='#333333')
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='#E2E8F0',
                    showline=False,
                    tickfont=dict(size=10, color='#333333'),
                    title_font=dict(size=11, color='#333333')
                ),
                margin=dict(l=60, r=40, t=60, b=60),
                showlegend=False  # Prefer direct labeling over legends
            )

            fig.write_html(filepath)

            # Read and encode
            with open(filepath, 'rb') as f:
                html_data = f.read()
                base64_data = base64.b64encode(html_data).decode('utf-8')

            return {
                "success": True,
                "filename": filepath.name,
                "filepath": str(filepath),
                "base64_data": base64_data,
                "file_size": len(html_data),
                "mime_type": "text/html"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

