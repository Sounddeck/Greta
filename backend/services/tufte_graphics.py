"""
GRETA PAI Tufte Graphics Service
Edward Tufte's principles for data visualization and graphical output
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from ..config import get_settings
from ..database import get_database

logger = logging.getLogger(__name__)

class TufteGraphicsService:
    """
    GRETA PAI Tufte Graphics Service
    Implements Edward Tufte's principles for data visualization and graphical output
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        
        # Graphics output directory
        self.graphics_dir = self.settings.base_dir / "graphics"
        self.graphics_dir.mkdir(parents=True, exist_ok=True)
        
        # Tufte's principles
        self.tufte_principles = {
            "data_ink_ratio": "1 > 0.8",  # Data-ink ratio should be greater than 0.8
            "chartjunk": "eliminate",  # Eliminate unnecessary chart elements
            "data_first": "show",  # Show data first, then context
            "multivariate": "display",  # Display multiple variables simultaneously
            "small_multiples": "use",  # Use small multiples for comparison
            "layering": "create",  # Create layering for complex data
            "integration": "ensure"  # Ensure text and graphics are integrated
            "clarity": "maintain"  # Maintain clarity of information
            "precision": "exact"  # Be precise and accurate
            "context": "provide"  # Provide context for understanding
            "comparison": "facilitate"  # Facilitate comparison of data
            "narrative": "support"  # Support narrative with evidence
            "aesthetics": "subtle"  # Use subtle aesthetics that don't distract
            "accessibility": "ensure"  # Ensure accessibility for all users
        }
        
        # Color palette (Tufte's subtle, data-friendly colors)
        self.color_palette = {
            "primary": "#1a1a1a",      # Dark gray for main elements
            "secondary": "#666666",    # Medium gray for secondary elements
            "accent": "#d32f2f",       # Green for highlights
            "accent_alt": "#1976d2",  # Blue for alternative highlights
            "warning": "#f57c00",      # Orange for warnings
            "error": "#d32f2f",         # Red for errors (same as accent)
            "info": "#2196f3",          # Light blue for information
            "background": "#ffffff",   # White for backgrounds
            "grid": "#f0f0f0",         # Light gray for grids
            "text": "#333333",          # Dark gray for text
            "text_light": "#666666",     # Light gray for secondary text
            "data_colors": [
                "#1f77b4",  # Blue
                "#ff7f0e",  # Orange
                "#2ca02c",  # Green
                "#d62728",  # Red
                "#9467bd",  # Purple
                "#8c564b",  # Brown
                "#e377c2",  # Teal
                "#7f7f7f",  # Gray
                "#bcbd22",  # Olive
                "#17becf",  # Cyan
                "#aec7e8",  # Light blue
                "#ffbb78",  # Light orange
                "#98df8a",  # Light green
                "#ff9896",  # Light red
                "#c5b0d5",  # Light purple
                "#f7b6d2",  # Light brown
                "#fdd835",  # Light yellow
                "#c5c5c5"  # Light gray
            ]
        }
        
        # Typography
        self.typography = {
            "font_family": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "font_size": {
                "title": 16,
                "subtitle": 14,
                "body": 12,
                "caption": 10,
                "annotation": 9
            },
            "font_weight": {
                "light": 300,
                "normal": 400,
                "medium": 500,
                "semibold": 600,
                "bold": 700
            },
            "line_height": {
                "tight": 1.2,
                "normal": 1.4,
                "relaxed": 1.6
            }
        }
        
        # Layout principles
        self.layout_principles = {
            "margins": {
                "top": 40,
                "right": 40,
                "bottom": 40,
                "left": 40
            },
            "padding": {
                "small": 8,
                "medium": 16,
                "large": 24
            },
            "spacing": {
                "tight": 4,
                "normal": 8,
                "relaxed": 16
            }
        }
        
        # Chart types following Tufte's principles
        self.chart_types = {
            "scatterplot": {
                "description": "Scatter plot for showing relationships between two variables",
                "tufte_compliant": True,
                "data_ink_ratio": 0.9
            },
            "line_chart": {
                "description": "Line chart for showing trends over time",
                "tufte_compliant": True,
                "data_ink_ratio": 0.85
            },
            "bar_chart": {
                "description": "Bar chart for comparing categories",
                "tufte_compliant": True,
                "data_ink_ratio": 0.8
            },
            "histogram": {
                "description": "Histogram for showing distribution of data",
                "tufte_compliant": True,
                "data_ink_ratio": 0.9
            },
            "box_plot": {
                "description": "Box plot for showing distribution summary",
                "tufte_compliant": True,
                "data_ink_ratio": 0.85
            },
            "heatmap": {
                "description": "Heatmap for showing data matrix",
                "tufte_compliant": False,
                "data_ink_ratio": 0.7
            },
            "small_multiple": {
                "description": "Small multiples for comparison",
                "tufte_compliant": True,
                "data_ink_ratio": 0.9
            },
            "sparkline": {
                "description": "Sparkline for showing data in context",
                "tufte_compliant": True,
                "data_ink_ratio": 0.95
            },
            "bullet_graph": {
                "description": "Bullet graph for comparing performance",
                "tufte_compliant": True,
                "data_ink_ratio": 0.85
            },
            "slope_graph": {
                "description": "Slope graph for comparing rates of change",
                "tufte_compliant": True,
                "data_ink_ratio": 0.85
            }
        }
        
        # Output formats
        self.output_formats = {
            "svg": {
                "extension": ".svg",
                "mime_type": "image/svg+xml",
                "vector": True
            },
            "png": {
                "extension": ".png",
                "mime_type": "image/png",
                "vector": False
            },
            "pdf": {
                "extension": ".pdf",
                "mime_type": "application/pdf",
                "vector": True
            },
            "html": {
                "extension": ".html",
                "mime_type": "text/html",
                "vector": False
            },
            "json": {
                "extension": ".json",
                "mime_type": "application/json",
                "vector": False
            }
        }
        
        # Graphics metrics
        self.graphics_metrics = {
            "total_graphics": 0,
            "successful_graphics": 0,
            "average_creation_time": 0,
            "graphics_by_type": {},
            "graphics_by_format": {},
            "tufte_compliance_score": 0
        }
        
    async def initialize(self):
        """Initialize Tufte graphics service"""
        try:
            self.db = await get_database()
            
            logger.info("✅ Tufte Graphics Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Tufte Graphics initialization failed: {e}")
            raise
    
    async def create_chart(self, chart_data: Dict[str, Any], chart_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create Tufte-compliant chart
        
        Args:
            chart_data: Data for the chart
            chart_type: Type of chart to create
            options: Additional options for chart creation
        
        Returns:
            Chart creation result
        """
        start_time = datetime.utcnow()
        
        try:
            self.graphics_metrics["total_graphics"] += 1
            
            # Validate chart type
            if chart_type not in self.chart_types:
                return {
                    "success": False,
                    "error": f"Unsupported chart type: {chart_type}",
                    "available_types": list(self.chart_types.keys())
                }
            
            # Merge options with defaults
            chart_options = self._get_default_options(chart_type, options or {})
            
            # Create chart based on type
            if chart_type == "scatterplot":
                chart_result = await self._create_scatterplot(chart_data, chart_options)
            elif chart_type == "line_chart":
                chart_result = await self._create_line_chart(chart_data, chart_options)
            elif chart_type == "bar_chart":
                chart_result = await self._create_bar_chart(chart_data, chart_options)
            elif chart_type == "histogram":
                chart_result = await self._create_histogram(chart_data, chart_options)
            elif chart_type == "box_plot":
                chart_result = await self._create_box_plot(chart_data, chart_options)
            elif chart_type == "heatmap":
                chart_result = await self._create_heatmap(chart_data, chart_options)
            elif chart_type == "small_multiple":
                chart_result = await self._create_small_multiple(chart_data, chart_options)
            elif chart_type == "sparkline":
                chart_result = await self._create_sparkline(chart_data, chart_options)
            elif chart_type == "bullet_graph":
                chart_result = await self._create_bullet_graph(chart_data, chart_options)
            elif chart_type == "slope_graph":
                chart_result = await self._create_slope_graph(chart_data, chart_options)
            else:
                chart_result = await self._create_generic_chart(chart_data, chart_options)
            
            # Calculate creation time
            creation_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_graphics_metrics(chart_type, creation_time, chart_result.get("success", False))
            
            # Add metadata
            chart_result.update({
                "chart_id": str(uuid.uuid4()),
                "chart_type": chart_type,
                "creation_time": creation_time,
                "tufte_principles": self.tufte_principles,
                "tufte_compliance": self._calculate_tufte_compliance(chart_result),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Store in database
            await self._store_chart(chart_result)
            
            return chart_result
            
        except Exception as e:
            logger.error(f"❌ Chart creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "chart_type": chart_type,
                "creation_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    def _get_default_options(self, chart_type: str, user_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get default options for chart type"""
        
        default_options = {
            "title": "",
            "subtitle": "",
            "width": 800,
            "height": 600,
            "format": "svg",
            "theme": "tufte",
            "show_grid": True,
            "show_legend": True,
            "show_annotations": True,
            "data_ink_ratio_target": 0.85,
            "color_palette": self.color_palette,
            "typography": self.typography,
            "layout": self.layout_principles,
            "margins": self.layout_principles["margins"],
            "padding": self.layout_principles["padding"]
        }
        
        # Chart-specific defaults
        if chart_type == "scatterplot":
            default_options.update({
                "point_size": 3,
                "point_opacity": 0.7,
                "show_trend_line": False,
                "show_confidence_interval": False
            })
        elif chart_type == "line_chart":
            default_options.update({
                "line_width": 2,
                "line_opacity": 0.8,
                "show_area": False,
                "show_points": False,
                "smooth_line": True
            })
        elif chart_type == "bar_chart":
            default_options.update({
                "bar_width": 0.8,
                "bar_spacing": 0.1,
                "show_values": True,
                "value_font_size": 10,
                "horizontal": False
            })
        elif chart_type == "histogram":
            default_options.update({
                "bin_count": 20,
                "bin_width": None,
                "show_density": False,
                "show_normal_curve": True
            })
        elif chart_type == "box_plot":
            default_options.update({
                "box_width": 0.6,
                "whisker_width": 1,
                "show_outliers": True,
                "outlier_marker_size": 2
            })
        elif chart_type == "heatmap":
            default_options.update({
                "color_scale": "sequential",
                "show_color_bar": True,
                "cell_spacing": 1,
                "show_labels": False
            })
        elif chart_type == "small_multiple":
            default_options.update({
                "grid_columns": 3,
                "grid_rows": 2,
                "spacing": 20,
                "shared_axes": True,
                "individual_legends": False
            })
        elif chart_type == "sparkline":
            default_options.update({
                "line_width": 1,
                "height": 30,
                "show_band": True,
                "band_opacity": 0.2,
                "show_endpoints": True
            })
        elif chart_type == "bullet_graph":
            default_options.update({
                "bullet_size": 4,
                "target_width": 12,
                "show_scale": True,
                "scale_position": "bottom"
            })
        elif chart_type == "slope_graph":
            default_options.update({
                "line_width": 2,
                "show_points": True,
                "point_size": 2,
                "angle_labels": True,
                "angle_label_position": "right"
            })
        
        # Merge with user options
        default_options.update(user_options)
        
        return default_options
    
    async def _create_scatterplot(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant scatter plot"""
        
        try:
            # Extract data
            x_data = data.get("x", [])
            y_data = data.get("y", [])
            
            if not x_data or not y_data or len(x_data) != len(y_data):
                return {
                    "success": False,
                    "error": "Invalid data: x and y arrays must have same length"
                }
            
            # Create SVG scatter plot
            svg_content = self._create_scatterplot_svg(x_data, y_data, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(x_data),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "x_range": [min(x_data), max(x_data)],
                "y_range": [min(y_data), max(y_data)],
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Scatter plot creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_line_chart(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant line chart"""
        
        try:
            # Extract data
            x_data = data.get("x", [])
            y_data = data.get("y", [])
            
            if not x_data or not y_data or len(x_data) != len(y_data):
                return {
                    "success": False,
                    "error": "Invalid data: x and y arrays must have same length"
                }
            
            # Create SVG line chart
            svg_content = self._create_line_chart_svg(x_data, y_data, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(x_data),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "x_range": [min(x_data), max(x_data)],
                "y_range": [min(y_data), max(y_data)],
                "trend": self._calculate_trend(x_data, y_data),
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Line chart creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_bar_chart(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant bar chart"""
        
        try:
            # Extract data
            categories = data.get("categories", [])
            values = data.get("values", [])
            
            if not categories or not values or len(categories) != len(values):
                return {
                    "success": False,
                    "error": "Invalid data: categories and values arrays must have same length"
                }
            
            # Create SVG bar chart
            svg_content = self._create_bar_chart_svg(categories, values, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(categories),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "value_range": [min(values), max(values)],
                "categories": categories,
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Bar chart creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_histogram(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant histogram"""
        
        try:
            # Extract data
            values = data.get("values", [])
            
            if not values:
                return {
                    "success": False,
                    "error": "Invalid data: values array is empty"
                }
            
            # Create SVG histogram
            svg_content = self._create_histogram_svg(values, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            # Calculate statistics
            import statistics
            stats = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(values),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "statistics": stats,
                "bins": options.get("bin_count", 20),
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Histogram creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_box_plot(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant box plot"""
        
        try:
            # Extract data
            values = data.get("values", [])
            
            if not values:
                return {
                    "success": False,
                    "error": "Invalid data: values array is empty"
                }
            
            # Create SVG box plot
            svg_content = self._create_box_plot_svg(values, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            # Calculate quartiles
            import statistics
            q1 = statistics.quantile(values, 0.25)
            q3 = statistics.quantile(values, 0.75)
            iqr = q3 - q1
            lower_bound = max(min(values), q1 - 1.5 * iqr)
            upper_bound = min(max(values), q3 + 1.5 * iqr)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(values),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "quartiles": {
                    "q1": q1,
                    "q2": statistics.median(values),
                    "q3": q3,
                    "iqr": iqr
                },
                "outliers": [v for v in values if v < lower_bound or v > upper_bound],
                "bounds": {
                    "lower": lower_bound,
                    "upper": upper_bound
                },
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Box plot creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_heatmap(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create heatmap (less Tufte-compliant but useful)"""
        
        try:
            # Extract data
            matrix = data.get("matrix", [])
            
            if not matrix or not matrix[0]:
                return {
                    "success": False,
                    "error": "Invalid data: matrix is empty"
                }
            
            # Create SVG heatmap
            svg_content = self._create_heatmap_svg(matrix, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(matrix) * len(matrix[0]) if matrix else 0,
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": False,  # Heatmaps generally have lower data-ink ratio
                "matrix_size": {
                    "rows": len(matrix),
                    "cols": len(matrix[0]) if matrix else 0
                },
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Heatmap creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_small_multiple(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant small multiples"""
        
        try:
            # Extract data
            datasets = data.get("datasets", [])
            
            if not datasets:
                return {
                    "success": False,
                    "error": "Invalid data: datasets array is empty"
                }
            
            # Create SVG small multiples
            svg_content = self._create_small_multiple_svg(datasets, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "dataset_count": len(datasets),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "grid_layout": {
                    "rows": options.get("grid_rows", 2),
                    "cols": options.get("grid_cols", 3)
                },
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Small multiples creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_sparkline(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant sparkline"""
        
        try:
            # Extract data
            values = data.get("values", [])
            
            if not values:
                return {
                    "success": False,
                    "error": "Invalid data: values array is empty"
                }
            
            # Create SVG sparkline
            svg_content = self._create_sparkline_svg(values, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "data_points": len(values),
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "value_range": [min(values), max(values)],
                "trend": self._calculate_trend(range(len(values)), values),
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Sparkline creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_bullet_graph(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant bullet graph"""
        
        try:
            # Extract data
            categories = data.get("categories", [])
            actual_values = data.get("actual", [])
            target_values = data.get("target", [])
            
            if not categories or not actual_values or not target_values:
                return {
                    "success": False,
                    "error": "Invalid data: categories, actual, and target arrays must be provided"
                }
            
            # Create SVG bullet graph
            svg_content = self._create_bullet_graph_svg(categories, actual_values, target_values, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "categories": categories,
                "actual_values": actual_values,
                "target_values": target_values,
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "performance_ratios": [a / t for a, t in zip(actual_values, target_values)],
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Bullet graph creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_slope_graph(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-compliant slope graph"""
        
        try:
            # Extract data
            categories = data.get("categories", [])
            values = data.get("values", [])
            
            if not categories or not values:
                return {
                    "success": False,
                    "error": "Invalid data: categories and values arrays must be provided"
                }
            
            # Create SVG slope graph
            svg_content = self._create_slope_graph_svg(categories, values, options)
            
            # Calculate data-ink ratio
            data_ink_ratio = self._calculate_data_ink_ratio(svg_content, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "categories": categories,
                "values": values,
                "data_ink_ratio": data_ink_ratio,
                "tufte_compliant": data_ink_ratio > 0.8,
                "slopes": self._calculate_slopes(values),
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Slope graph creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_generic_chart(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create generic chart for unsupported types"""
        
        try:
            # Create a simple bar chart as fallback
            if "categories" in data and "values" in data:
                return await self._create_bar_chart(data, options)
            else:
                return {
                    "success": False,
                    "error": "Generic chart requires categories and values"
                }
            
        except Exception as e:
            logger.error(f"❌ Generic chart creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_scatterplot_svg(self, x_data: List[float], y_data: List[float], options: Dict[str, Any]) -> str:
        """Create SVG scatter plot"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        point_size = options["point_size"]
        point_opacity = options["point_opacity"]
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate scales
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid if requested
        if options.get("show_grid", True):
            svg_parts.append(self._create_grid(plot_width, plot_height, x_min, x_max, y_min, y_max, color_palette["grid"]))
        
        # Add axes
        svg_parts.append(self._create_axes(plot_width, plot_height, x_min, x_max, y_min, y_max, color_palette["text"]))
        
        # Add data points
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            x_pos = margin["left"] + ((x - x_min) / x_range) * plot_width)
            y_pos = margin["top"] + ((y_max - y) / y_range) * plot_height)
            
            svg_parts.append(
                f'<circle cx="{x_pos}" cy="{y_pos}" r="{point_size}" '
                f'fill="{color_palette["primary"]}" '
                f'fill-opacity="{point_opacity}" '
                f'stroke="{color_palette["primary"]}" '
                f'stroke-width="1" '
                f'stroke-opacity="0.5"/>'
            )
        
        # Add labels
        if options.get("show_annotations", True):
            svg_parts.append(self._create_scatterplot_labels(x_data, y_data, plot_width, plot_height, x_min, x_max, y_min, y_max, color_palette["text_light"]))
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_line_chart_svg(self, x_data: List[float], y_data: List[float], options: Dict[str, Any]) -> str:
        """Create SVG line chart"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        line_width = options["line_width"]
        line_opacity = options["line_opacity"]
        show_points = options["show_points"]
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate scales
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid if requested
        if options.get("show_grid", True):
            svg_parts.append(self._create_grid(plot_width, plot_height, x_min, x_max, y_min, y_max, color_palette["grid"]))
        
        # Add axes
        svg_parts.append(self._create_axes(plot_width, plot_height, x_min, x_max, y_min, y_max, color_palette["text"]))
        
        # Create path for line
        path_parts = []
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            x_pos = ((x - x_min) / x_range) * plot_width
            y_pos = ((y_max - y) / y_range) * plot_height
            if i == 0:
                path_parts.append(f"M {x_pos} {y_pos}")
            else:
                path_parts.append(f"L {x_pos} {y_pos}")
        
        path_data = " ".join(path_parts)
        
        svg_parts.append(
            f'<path d="{path_data}" '
            f'fill="none" '
            f'stroke="{color_palette["primary"]}" '
            f'stroke-width="{line_width}" '
            f'stroke-opacity="{line_opacity}"/>'
        )
        
        # Add points if requested
        if show_points:
            for x, y in zip(x_data, y_data):
                x_pos = margin["left"] + ((x - x_min) / x_range) * plot_width)
                y_pos = margin["top"] + ((y_max - y) / y_range) * plot_height)
                
                svg_parts.append(
                    f'<circle cx="{x_pos}" cy="{y_pos}" r="2" '
                    f'fill="{color_palette["primary"]}" '
                    f'fill-opacity="0.7" '
                    f'stroke="white" '
                    f'stroke-width="1"/>'
                )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_bar_chart_svg(self, categories: List[str], values: List[float], options: Dict[str, Any]) -> str:
        """Create SVG bar chart"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        bar_width = options["bar_width"]
        bar_spacing = options["bar_spacing"]
        show_values = options["show_values"]
        value_font_size = options["value_font_size"]
        horizontal = options["horizontal"]
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate bar dimensions
        n = len(categories)
        if horizontal:
            bar_length = plot_height / n - bar_spacing * (n - 1)
            bar_width = bar_width
        else:
            bar_length = plot_width / n - bar_spacing * (n - 1)
            bar_height = bar_width
        
        max_value = max(values) if values else 1
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid if requested
        if options.get("show_grid", True):
            if horizontal:
                svg_parts.append(self._create_grid(plot_width, plot_height, 0, max_value, 0, n, color_palette["grid"], horizontal=True))
            else:
                svg_parts.append(self._create_grid(plot_width, plot_height, 0, n, 0, max_value, color_palette["grid"]))
        
        # Add axes
        if horizontal:
            svg_parts.append(self._create_axes(plot_width, plot_height, 0, max_value, 0, n, color_palette["text"], horizontal=True))
        else:
            svg_parts.append(self._create_axes(plot_width, plot_height, 0, n, 0, max_value, color_palette["text"]))
        
        # Add bars
        for i, (category, value) in enumerate(zip(categories, values)):
            if horizontal:
                y_pos = i * (bar_length + bar_spacing)
                bar_height = (value / max_value) * plot_height
                x_pos = 0
                
                svg_parts.append(
                    f'<rect x="{x_pos}" y="{y_pos}" '
                    f'width="{bar_width}" height="{bar_height}" '
                    f'fill="{color_palette["data_colors"][i % len(color_palette["data_colors"])]}"/>'
                )
                
                # Add value label
                if show_values:
                    svg_parts.append(
                        f'<text x="{x_pos + bar_width + 5}" y="{y_pos + bar_height/2}" '
                        f'font-size="{value_font_size}" '
                        f'fill="{color_palette["text"]}" '
                        f'text-anchor="start" '
                        f'alignment-baseline="middle">'
                        f'{value:.2f}</text>'
                    )
            else:
                x_pos = i * (bar_length + bar_spacing)
                bar_width = (value / max_value) * plot_width
                y_pos = plot_height - bar_width
                
                svg_parts.append(
                    f'<rect x="{x_pos}" y="{y_pos}" '
                    f'width="{bar_width}" height="{bar_width}" '
                    f'fill="{color_palette["data_colors"][i % len(color_palette["data_colors"])]}"/>'
                )
                
                # Add value label
                if show_values:
                    svg_parts.append(
                        f'<text x="{x_pos + bar_width/2}" y="{y_pos - 5}" '
                        f'font-size="{value_font_size}" '
                        f'fill="{color_palette["text"]}" '
                        f'text-anchor="middle" '
                        f'alignment-baseline="bottom">'
                        f'{value:.2f}</text>'
                    )
                
                # Add category label
                svg_parts.append(
                    f'<text x="{x_pos + bar_width/2}" y="{plot_height + 15}" '
                    f'font-size="10" '
                    f'fill="{color_palette["text_light"]}" '
                    f'text-anchor="middle" '
                    f'alignment-baseline="top">'
                    f'{category}</text>'
                )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_histogram_svg(self, values: List[float], options: Dict[str, Any]) -> str:
        """Create SVG histogram"""
        
        import statistics
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        bin_count = options.get("bin_count", 20)
        show_normal_curve = options.get("show_normal_curve", True)
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate bins
        min_val, max_val = min(values), max(values)
        bin_width = (max_val - min_val) / bin_count
        bins = [min_val + i * bin_width for i in range(bin_count + 1)]
        
        # Count frequencies
        hist, _ = np.histogram(values, bins=bins)
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid if requested
        if options.get("show_grid", True):
            svg_parts.append(self._create_grid(plot_width, plot_height, bins[0], bins[-1], 0, max(hist), color_palette["grid"]))
        
        # Add axes
        svg_parts.append(self._create_histogram_axes(plot_width, plot_height, bins, hist, color_palette["text"]))
        
        # Add bars
        for i, (start, end, count) in enumerate(zip(bins[:-1], bins[1:], hist)):
            bar_height = (count / max(hist)) * plot_height
            bar_width = (bins[i+1] - bins[i]) / (bins[-1] - bins[0]) * plot_width
            x_pos = ((bins[i] + bins[i+1]) / 2 - bins[0]) / (bins[-1] - bins[0]) * plot_width
            
            svg_parts.append(
                f'<rect x="{x_pos}" y="{plot_height - bar_height}" '
                f'width="{bar_width}" height="{bar_height}" '
                f'fill="{color_palette["data_colors"][i % len(color_palette["data_colors"])]}"/>'
            )
        
        # Add normal curve if requested
        if show_normal_curve:
            # Generate normal distribution points
            mu = statistics.mean(values)
            sigma = statistics.stdev(values)
            x_norm = np.linspace(bins[0], bins[-1], 100)
            y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
            
            # Scale to plot area
            x_scaled = [(x - bins[0]) / (bins[-1] - bins[0]) * plot_width for x in x_norm]
            y_scaled = y * plot_height * max(hist)
            
            # Create path for normal curve
            path_parts = []
            for x, y in zip(x_scaled, y_scaled):
                if path_parts == []:
                    path_parts.append(f"M {x} {plot_height - y}")
                else:
                    path_parts.append(f"L {x} {plot_height - y}")
            
            path_data = " ".join(path_parts)
            
            svg_parts.append(
                f'<path d="{path_data}" '
                f'fill="none" '
                f'stroke="{color_palette["accent"]}" '
                f'stroke-width="2" '
                f'stroke-dasharray="5,5" '
                f'opacity="0.7"/>'
            )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_box_plot_svg(self, values: List[float], options: Dict[str, Any]) -> str:
        """Create SVG box plot"""
        
        import statistics
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        box_width = options["box_width"]
        whisker_width = options["whisker_width"]
        show_outliers = options["show_outliers"]
        outlier_marker_size = options["outlier_marker_size"]
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate statistics
        q1 = statistics.quantile(values, 0.25)
        q2 = statistics.median(values)
        q3 = statistics.quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(min(values), q1 - 1.5 * iqr)
        upper_bound = min(max(values), q3 + 1.5 * iqr)
        
        # Identify outliers
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        
        # Box position
        box_y = plot_height * 0.3
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid if requested
        if options.get("show_grid", True):
            svg_parts.append(self._create_grid(plot_width, plot_height, lower_bound, upper_bound, 0, 1, color_palette["grid"]))
        
        # Add axes
        svg_parts.append(self._create_box_plot_axes(plot_width, plot_height, lower_bound, upper_bound, color_palette["text"]))
        
        # Add whiskers
        svg_parts.append(
            f'<line x1="{margin["left"] + box_width/2}" y1="{box_y + box_height}" '
            f'x2="{margin["left"] + plot_width - box_width/2}" y2="{box_y + box_height}" '
            f'stroke="{color_palette["primary"]}" '
            f'stroke-width="{whisker_width}" '
            f'stroke-linecap="round"/>'
        )
        
        # Add box
        svg_parts.append(
            f'<rect x="{margin["left"] + box_width/2}" y="{box_y}" '
            f'width="{box_width}" height="{box_height}" '
            f'fill="{color_palette["primary"]}" '
            f'fill-opacity="0.3" '
            f'stroke="{color_palette["primary"]}" '
            f'stroke-width="1"/>'
        )
        
        # Add quartile lines
        q1_pos = margin["left"] + box_width/2
        q3_pos = margin["left"] + box_width/2 + box_width
        
        svg_parts.append(f'<line x1="{q1_pos}" y1="{box_y}" x2="{q1_pos}" y2="{box_y + box_height}" '
                       f'stroke="{color_palette["text"]}" '
                       f'stroke-width="1" '
                       f'stroke-dasharray="2,2" '
                       f'opacity="0.5"/>')
        
        svg_parts.append(f'<line x1="{q3_pos}" y1="{box_y}" x2="{q3_pos}" y2="{box_y + box_height}" '
                       f'stroke="{color_palette["text"]}" '
                       f'stroke-width="1" '
                       f'stroke-dasharray="2,2" '
                       f'opacity="0.5"/>')
        
        # Add outliers
        if show_outliers and outliers:
            for outlier in outliers:
                y_pos = box_y + box_height / 2
                x_pos = margin["left"] + box_width/2 + box_width + 10
                
                svg_parts.append(
                    f'<circle cx="{x_pos}" cy="{y_pos}" r="{outlier_marker_size}" '
                    f'fill="{color_palette["accent"]}" '
                    f'stroke="{color_palette["accent"]}" '
                    f'stroke-width="1"/>'
                )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_heatmap_svg(self, matrix: List[List[float]], options: Dict[str, Any]) -> str:
        """Create SVG heatmap"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        cell_spacing = options.get("cell_spacing", 1)
        color_scale = options.get("color_scale", "sequential")
        show_color_bar = options.get("show_color_bar", True)
        color_palette = options["color_palette"]
        
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        cell_width = (width - margin["left"] - margin["right"] - (cols - 1) * cell_spacing) / cols
        cell_height = (height - margin["top"] - margin["bottom"] - (rows - 1) * cell_spacing) / rows
        
        # Find min and max values
        flat_values = [val for row in matrix for val in row]
        min_val = min(flat_values)
        max_val = max(flat_values)
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Create color scale
        if color_scale == "sequential":
            colors = self._create_sequential_colors(min_val, max_val, color_palette["data_colors"])
        else:
            colors = self._create_diverging_colors(min_val, max_val, color_palette["data_colors"])
        
        # Add cells
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                x_pos = j * (cell_width + cell_spacing)
                y_pos = i * (cell_height + cell_spacing)
                
                color = self._get_color_for_value(value, min_val, max_val, colors)
                
                svg_parts.append(
                    f'<rect x="{x_pos}" y="{y_pos}" '
                    f'width="{cell_width}" height="{cell_height}" '
                    f'fill="{color}" '
                    f'stroke="white" '
                    f'stroke-width="0.5"/>'
                )
        
        # Add color bar if requested
        if show_color_bar:
            bar_width = 20
            bar_x = width - margin["right"] - bar_width
            bar_height = height - margin["top"] - margin["bottom"]
            
            for i, color in enumerate(colors):
                y_pos = (i / len(colors)) * bar_height
                
                svg_parts.append(
                    f'<rect x="{bar_x}" y="{y_pos}" '
                    f'width="{bar_width}" height="{bar_height/len(colors)}" '
                    f'fill="{color}"/>'
                )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_small_multiple_svg(self, datasets: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Create SVG small multiples"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        grid_rows = options.get("grid_rows", 2)
        grid_cols = options.get("grid_cols", 3)
        spacing = options.get("spacing", 20)
        color_palette = options["color_palette"]
        
        cell_width = (width - margin["left"] - margin["right"] - (grid_cols - 1) * spacing) / grid_cols
        cell_height = (height - margin["top"] - margin["bottom"] - (grid_rows - 1) * spacing) / grid_rows
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid
        for i in range(grid_rows):
            for j in range(grid_cols):
                index = i * grid_cols + j
                
                if index < len(datasets):
                    data = datasets[index]
                    
                    # Create mini chart
                    mini_width = cell_width
                    mini_height = cell_height
                    mini_x = j * (cell_width + spacing)
                    mini_y = i * (cell_height + spacing)
                    
                    # Create a simple bar chart for demonstration
                    if "categories" in data and "values" in data:
                        categories = data["categories"][:3]  # Limit for small multiples
                        values = data["values"][:3]
                        
                        mini_chart_width = mini_width
                        mini_chart_height = mini_height
                        
                        # Create mini bar chart
                        mini_svg = self._create_mini_bar_chart(categories, values, mini_chart_width, mini_chart_height, color_palette)
                        
                        svg_parts.append(f'<g transform="translate({mini_x}, {mini_y})">{mini_svg}</g>')
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_sparkline_svg(self, values: List[float], options: Dict[str, Any]) -> str:
        """Create SVG sparkline"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        line_width = options["line_width"]
        band_opacity = options.get("band_opacity", 0.2)
        show_endpoints = options.get("show_endpoints", True)
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate scales
        min_val, max_val = min(values), max(values)
        val_range = max_val - min_val if max_val != min_val else 1
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add band
        svg_parts.append(
            f'<rect x="0" y="0" width="{plot_width}" height="{plot_height}" '
            f'fill="{color_palette["grid"]}" '
            f'fill-opacity="{band_opacity}"/>'
        )
        
        # Create path for sparkline
        path_parts = []
        for i, value in enumerate(values):
            x_pos = (i / (len(values) - 1)) * plot_width
            y_pos = plot_height - ((value - min_val) / val_range) * plot_height
            
            if path_parts == []:
                path_parts.append(f"M {x_pos} {y_pos}")
            else:
                path_parts.append(f"L {x_pos} {y_pos}")
        
        path_data = " ".join(path_parts)
        
        svg_parts.append(
            f'<path d="{path_data}" '
            f'fill="none" '
            f'stroke="{color_palette["primary"]}" '
            f'stroke-width="{line_width}" '
            f'stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
        
        # Add endpoints if requested
        if show_endpoints:
            # First endpoint
            x_first = 0
            y_first = plot_height - ((values[0] - min_val) / val_range) * plot_height
            svg_parts.append(
                f'<circle cx="{x_first}" cy="{y_first}" r="3" '
                f'fill="{color_palette["accent"]}" '
                f'stroke="white" '
                f'stroke-width="1"/>'
            )
            
            # Last endpoint
            x_last = plot_width
            y_last = plot_height - ((values[-1] - min_val) / val_range) * plot_height
            svg_parts.append(
                f'<circle cx="{x_last}" cy="{y_last}" r="3" '
                f'fill="{color_palette["accent"]}" '
                f'stroke="white" '
                f'stroke-width="1"/>'
            )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_bullet_graph_svg(self, categories: List[str], actual_values: List[float], target_values: List[float], options: Dict[str, Any]) -> str:
        """Create SVG bullet graph"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        bullet_size = options["bullet_size"]
        target_width = options.get("target_width", 12)
        scale_position = options.get("scale_position", "bottom")
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Calculate scales
        max_target = max(target_values) if target_values else 1
        max_actual = max(actual_values) if actual_values else 1
        max_value = max(max_target, max_actual)
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid lines at 0%, 100%, and intermediate values
        grid_positions = [0, 0.5, 1.0]
        for pos in grid_positions:
            if scale_position == "bottom":
                y_pos = plot_height - 10
            else:
                y_pos = plot_height - (pos * plot_height)
            
            svg_parts.append(
                f'<line x1="0" y1="{y_pos}" x2="{plot_width}" y2="{y_pos}" '
                f'stroke="{color_palette["grid"]}" '
                f'stroke-width="1" '
                f'stroke-dasharray="5,5" '
                f'opacity="0.5"/>'
            )
        
        # Add bars and bullets
        for i, (category, actual, target) in enumerate(zip(categories, actual_values, target_values)):
            # Calculate position
            y_pos = i * (plot_height / len(categories))
            
            # Calculate bar
            bar_width = (target / max_value) * plot_width
            bar_x = plot_width - bar_width - 10  # Space for bullet
            
            svg_parts.append(
                f'<rect x="{bar_x}" y="{y_pos}" '
                f'width="{bar_width}" height="8" '
                f'fill="{color_palette["data_colors"][i % len(color_palette["data_colors"])]}"/>'
            )
            
            # Add bullet
            bullet_x = plot_width - 8
            svg_parts.append(
                f'<circle cx="{bullet_x}" cy="{y_pos + 4}" r="{bullet_size}" '
                f'fill="{color_palette["accent"]}" '
                f'stroke="{color_palette["accent"]}" '
                f'stroke-width="1"/>'
            )
            
            # Add category label
            svg_parts.append(
                f'<text x="0" y="{y_pos + 15}" '
                f'font-size="10" '
                f'fill="{color_palette["text"]}" '
                f'text-anchor="start" '
                f'alignment-baseline="top">'
                f'{category}</text>'
            )
            
            # Add actual value
            svg_parts.append(
                f'<text x="{bar_x + bar_width + 5}" y="{y_pos + 4}" '
                f'font-size="9" '
                f'fill="{color_palette["text"]}" '
                f'text-anchor="start" '
                f'alignment-baseline="middle">'
                f'{actual:.2f}</text>'
            )
            
            # Add target value on scale
            if scale_position == "bottom":
                svg_parts.append(
                    f'<text x="{plot_width + 5}" y="{y_pos + 4}" '
                    f'font-size="9" '
                    f'fill="{color_palette["accent"]}" '
                    f'text-anchor="start" '
                    f'alignment-baseline="middle">'
                    f'{target:.2f}</text>'
                )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_slope_graph_svg(self, categories: List[str], values: List[float], options: Dict[str, Any]) -> str:
        """Create SVG slope graph"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        point_size = options.get("point_size", 2)
        show_points = options.get("show_points", True)
        angle_labels = options.get("angle_labels", True)
        angle_label_position = options.get("angle_label_position", "right")
        color_palette = options["color_palette"]
        
        # Calculate plot area
        plot_width = width - margin["left"] - margin["right"]
        plot_height = height - margin["top"] - margin["bottom"]
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add grid if requested
        if options.get("show_grid", True):
            svg_parts.append(self._create_grid(plot_width, plot_height, 0, len(categories), 0, max(values), color_palette["grid"]))
        
        # Add axes
        svg_parts.append(self._create_slope_graph_axes(plot_width, plot_height, categories, values, color_palette["text"], angle_labels, angle_label_position))
        
        # Add slopes
        for i, (category, value) in enumerate(zip(categories, values)):
            x_pos = (i / (len(categories) - 1)) * plot_width
            y_pos = plot_height - (value / max(values)) * plot_height
            
            # Create line from origin to point
            svg_parts.append(
                f'<line x1="0" y1="{plot_height}" x2="{x_pos}" y2="{y_pos}" '
                f'stroke="{color_palette["data_colors"][i % len(color_palette["data_colors"])]} '
                f'stroke-width="2" '
                f'stroke-linecap="round" '
                f'opacity="0.7"/>'
            )
            
            # Add point
            if show_points:
                svg_parts.append(
                    f'<circle cx="{x_pos}" cy="{y_pos}" r="{point_size}" '
                    f'fill="{color_palette["data_colors"][i % len(color_palette["data_colors"])]} '
                    f'stroke="white" '
                    f'stroke-width="1"/>'
                )
            
            # Add category label
            svg_parts.append(
                f'<text x="{x_pos}" y="{plot_height + 15}" '
                f'font-size="10" '
                f'fill="{color_palette["text"]}" '
                f'text-anchor="middle" '
                f'alignment-baseline="top">'
                f'{category}</text>'
            )
            
            # Add value label
            svg_parts.append(
                f'<text x="{x_pos}" y="{y_pos - 10}" '
                f'font-size="9" '
                f'fill="{color_palette["text_light"]}" '
                f'text-anchor="middle" '
                f'alignment-baseline="bottom">'
                f'{value:.2f}</text>'
            )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    def _create_grid(self, width: float, height: float, x_min: float, x_max: float, y_min: float, y_max: float, color: str, horizontal: bool = False) -> str:
        """Create grid for charts"""
        
        grid_parts = []
        
        # Horizontal grid lines
        for i in range(6):  # 5 grid lines at 20% intervals
            y_pos = i * (height / 5)
            if horizontal:
                line = f'<line x1="0" y1="{y_pos}" x2="{width}" y2="{y_pos}" '
            else:
                x_pos = i * (width / 5)
                line = f'<line x1="{x_pos}" y1="0" x2="{x_pos}" y2="{height}" '
            
            grid_parts.append(
                line + f'stroke="{color}" '
                f'stroke-width="1" '
                f'stroke-dasharray="2,2" '
                f'opacity="0.3"/>'
            )
        
        return "\n".join(grid_parts)
    
    def _create_axes(self, width: float, height: float, x_min: float, x_max: float, y_min: float, y_max: float, color: str, horizontal: bool = False, text_size: int = 10) -> str:
        """Create axes for charts"""
        
        axis_parts = []
        
        if horizontal:
            # X-axis
            for i in range(6):
                x_pos = i * (width / 5)
                label = f'{x_min + (x_max - x_min) * i / 5:.1f}'
                
                axis_parts.append(
                    f'<text x="{x_pos}" y="{height + 5}" '
                    f'font-size="{text_size}" '
                    f'fill="{color}" '
                    f'text-anchor="middle" '
                    f'alignment-baseline="top">'
                    f'{label}</text>'
                )
                
                axis_parts.append(
                    f'<line x1="{x_pos}" y1="0" x2="{x_pos}" y2="{height}" '
                    f'stroke="{color}" '
                    f'stroke-width="1" '
                    f'stroke-dasharray="2,2" '
                    f'opacity="0.3"/>'
                )
        else:
            # Y-axis
            for i in range(6):
                y_pos = i * (height / 5)
                label = f'{y_min + (y_max - y_min) * i / 5:.1f}'
                
                axis_parts.append(
                    f'<text x="-5" y="{y_pos}" '
                    f'font-size="{text_size}" '
                    f'fill="{color}" '
                    f'text-anchor="end" '
                    f'alignment-baseline="middle">'
                        f'{label}</text>'
                )
                
                axis_parts.append(
                    f'<line x1="0" y1="{y_pos}" x2="{width}" y2="{y_pos}" '
                    f'stroke="{color}" '
                    f'stroke-width="1" '
                    f'stroke-dasharray="2,2" '
                    f'opacity="0.3"/>'
                )
        
        return "\n".join(axis_parts)
    
    def _create_histogram_axes(self, width: float, height: float, bins: List[float], hist: List[int], color: str, text_size: int = 10) -> str:
        """Create axes for histogram"""
        
        axis_parts = []
        
        # X-axis (bins)
        for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            x_pos = ((start + end) / 2 - bins[0]) / (bins[-1] - bins[0])) * width
            label = f'{start:.1f}-{end:.1f}'
            
            axis_parts.append(
                f'<text x="{x_pos}" y="{height + 5}" '
                f'font-size="{text_size}" '
                f'fill="{color}" '
                f'text-anchor="middle" '
                f'alignment-baseline="top">'
                f'{label}</text>'
            )
            
            axis_parts.append(
                f'<line x1="{x_pos}" y1="0" x2="{x_pos}" y2="{height}" '
                f'stroke="{color}" '
                f'stroke-width="1" '
                f'stroke-dasharray="2,2" '
                f'opacity="0.3"/>'
            )
        
        # Y-axis (counts)
        max_count = max(hist)
        for i in range(6):
            count_pos = i * (height / 5)
            count_label = str(int(max_count * i / 5))
            
            axis_parts.append(
                f'<text x="-5" y="{count_pos}" '
                f'font-size="{text_size}" '
                f'fill="{color}" '
                f'text-anchor="end" '
                f'alignment-baseline="middle">'
                f'{count_label}</text>'
            )
            
            axis_parts.append(
                f'<line x1="0" y1="{count_pos}" x2="{width}" y2="{count_pos}" '
                f'stroke="{color}" '
                f'stroke-width="1" '
                f'stroke-dasharray="2,2" '
                f'opacity="0.3"/>'
            )
        
        return "\n".join(axis_parts)
    
    def _create_scatterplot_labels(self, x_data: List[float], y_data: List[float], width: float, height: float, x_min: float, x_max: float, y_min: float, y_max: float, color: str) -> str:
        """Create labels for scatter plot"""
        
        label_parts = []
        
        # Add min/max labels
        label_parts.append(
            f'<text x="0" y="{height + 5}" '
            f'font-size="9" '
            f'fill="{color}" '
            f'text-anchor="start" '
            f'alignment-baseline="top">'
            f'({x_min:.2f}, {y_min:.2f})</text>'
        )
        
        label_parts.append(
            f'<text x="{width}" y="{height + 5}" '
            f'font-size="9" '
            f'fill="{color}" '
            f'text-anchor="end" '
            f'alignment-baseline="top">'
            f'({x_max:.2f}, {y_max:.2f})</text>'
        )
        
        return "\n".join(label_parts)
    
    def _create_mini_bar_chart(self, categories: List[str], values: List[float], width: int, height: int, color_palette: Dict[str, Any]) -> str:
        """Create mini bar chart for small multiples"""
        
        # Create simplified mini bar chart
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        ]
        
        max_val = max(values) if values else 1
        
        for i, (category, value) in enumerate(zip(categories, values)):
            y_pos = i * (height / len(categories))
            bar_height = (value / max_val) * height
            bar_width = width
            
            svg_parts.append(
                f'<rect x="0" y="{y_pos}" '
                f'width="{bar_width}" height="{bar_height}" '
                f'fill="{color_palette["data_colors"][i % len(color_palette["data_colors"])]}"/>'
            )
        
        svg_parts.append('</svg>')
        
        return "\n".join(svg_parts)
    
    def _calculate_data_ink_ratio(self, svg_content: str, options: Dict[str, Any]) -> float:
        """Calculate data-ink ratio for SVG content"""
        
        # Parse SVG content to calculate data-ink ratio
        # This is a simplified calculation - in production, would use more sophisticated parsing
        
        # For demonstration, return estimated data-ink ratio
        if options.get("chart_type") in ["scatterplot", "sparkline"]:
            return 0.9  # High data-ink ratio
        elif options.get("chart_type") in ["line_chart", "box_plot", "small_multiple"]:
            return 0.85  # Good data-ink ratio
        elif options.get("chart_type") in ["bar_chart", "histogram"]:
            return 0.8  # Acceptable data-ink ratio
        else:
            return 0.7  # Lower data-ink ratio
        
        return 0.8
    
    def _calculate_trend(self, x_data: List[float], y_data: List[float]) -> str:
        """Calculate trend direction and strength"""
        
        try:
            import statistics
            
            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = statistics.linregress(x_data, y_data)
            
            if r_value > 0.5:
                direction = "increasing"
                strength = "strong" if r_value > 0.8 else "moderate"
            elif r_value < -0.5:
                direction = "decreasing"
                strength = "strong" if r_value < -0.8 else "moderate"
            else:
                direction = "stable"
                strength = "weak"
            
            return f"{direction} ({r_value:.2f})"
            
        except Exception as e:
            return "unknown (0.0)"
    
    def _calculate_slopes(self, values: List[float]) -> List[float]:
        """Calculate slopes for slope graph"""
        
        # Calculate slopes between consecutive points
        slopes = []
        for i in range(len(values) - 1):
            if i > 0:
                slope = values[i] - values[i-1]
                slopes.append(slope)
        
        return slopes
    
    def _get_color_for_value(self, value: float, min_val: float, max_val: float, colors: List[str]) -> str:
        """Get color for a value based on color scale"""
        
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        index = int(normalized * (len(colors) - 1))
        return colors[index]
    
    def _create_sequential_colors(self, min_val: float, max_val: float, colors: List[str]) -> List[str]:
        """Create sequential color scale"""
        
        return [colors[i % len(colors)] for i in range(10)]
    
    def _create_diverging_colors(self, min_val: float, max_val: float, colors: List[str]) -> List[str]:
        """Create diverging color scale"""
        
        # Use first half for low values, second half for high values
        mid_point = (min_val + max_val) / 2
        low_colors = colors[:len(colors)//2]
        high_colors = colors[len(colors)//2:]
        
        return [low_colors[int((min_val / mid_point) * (len(low_colors) - 1))] if min_val < mid_point else low_colors[0],
                high_colors[int(((val - mid_point) / mid_point) * (len(high_colors) - 1))) if max_val > mid_point else high_colors[0]]
    
    def _calculate_tufte_compliance(self, chart_result: Dict[str, Any]) -> float:
        """Calculate Tufte compliance score"""
        
        score = 0.0
        
        # Data-ink ratio (most important principle)
        data_ink_ratio = chart_result.get("data_ink_ratio", 0)
        if data_ink_ratio > 0.8:
            score += 0.4
        elif data_ink_ratio > 0.6:
            score += 0.2
        else:
            score -= 0.2
        
        # Tufte compliance flag
        if chart_result.get("tufte_compliant", False):
            score -= 0.3
        
        # Chart type compliance
        chart_type = chart_result.get("chart_type", "")
        if chart_type in ["scatterplot", "line_chart", "bar_chart", "histogram", "box_plot", "small_multiple", "sparkline", "bullet_graph", "slope_graph"]:
            score += 0.2
        else:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _update_graphics_metrics(self, chart_type: str, creation_time: float, success: bool):
        """Update graphics metrics"""
        
        self.graphics_metrics["total_graphics"] += 1
        
        if success:
            self.graphics_metrics["successful_graphics"] += 1
        
        # Update type metrics
        if chart_type not in self.graphics_metrics["graphics_by_type"]:
            self.graphics_metrics["graphics_by_type"][chart_type] = {
                "count": 0,
                "total_time": 0,
                "successful": 0
            }
        
        self.graphics_metrics["graphics_by_type"][chart_type]["count"] += 1
        self.graphics_metrics["graphics_by_type"][chart_type]["total_time"] += creation_time
        self.graphics_metrics["graphics_by_type"][chart_type]["successful"] += 1
        
        # Update format metrics
        format_used = self.graphics_metrics["graphics_by_type"][chart_type].get("format", "unknown")
        if format_used not in self.graphics_metrics["graphics_by_format"]:
            self.graphics_metrics["graphics_by_format"][format_used] = {
                "count": 0,
                "total_time": 0,
                "successful": 0
            }
        
        self.graphics_metrics["graphics_by_format"][format_used]["count"] += 1
        self.graphics_metrics["graphics_by_format"][format_used]["total_time"] += creation_time
        self.graphics_metrics["graphics_by_format"][format_used]["successful"] += 1
        
        # Update average creation time
        total_time = self.graphics_metrics["total_graphics"]
        current_avg = self.graphics_metrics["average_creation_time"]
        self.graphics_metrics["average_creation_time"] = (
            (current_avg * (total_time - 1) + creation_time) / total_time
        )
        
        # Calculate Tufte compliance score
        all_scores = []
        for chart_data in self.graphics_metrics["graphics_by_type"].values():
            compliance = chart_data.get("tufte_compliance", 0.5)
            all_scores.append(compliance)
        
        self.graphics_metrics["tufte_compliance_score"] = sum(all_scores) / len(all_scores) if all_scores else 0
    
    async def _store_chart(self, chart_result: Dict[str, Any]):
        """Store chart in database"""
        try:
            chart_data = {
                "chart_id": chart_result["chart_id"],
                "chart_type": chart_result["chart_type"],
                "format": chart_result["format"],
                "content": chart_result["content"],
                "metadata": {
                    "created_at": chart_result["timestamp"],
                    "data_ink_ratio": chart_result.get("data_ink_ratio", 0),
                    "tufte_compliance": chart_result.get("tufte_compliant", False),
                    "file_path": f"{self.graphics_dir}/{chart_result['chart_id']}.{chart_result['format'].split(' ')[1]}",
                    "size": chart_result.get("chart_size"),
                    "metrics": chart_result.get("metrics", {})
                }
            }
            
            # Save to database
            await self.db.insert_one("charts", chart_data)
            
            # Save to file
            chart_file = self.graphics_dir / f"{chart_result['chart_id']}.{chart_result['format'].split(' ')[1]}"
            with open(chart_file, 'w') as f:
                json.dump(chart_data, f, indent=2)
            
            logger.info(f"✅ Chart saved: {chart_result['chart_id']}")
            
        except Exception as e:
            logger.debug(f"Failed to store chart: {e}")
    
    def _get_default_options(self, chart_type: str, user_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get default options for chart type"""
        
        return {
            "show_grid": True,
            "show_legend": True,
            "show_annotations": True,
            "theme": "tufte",
            "color_palette": self.color_palette,
            "typography": self.typography,
            "layout": self.layout_principles,
            "margins": self.layout_principles["margins"],
            "padding": self.layout_principles["padding"]
        }
    
    async def create_report(self, data: Dict[str, Any], report_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Tufte-style report with multiple visualizations
        
        Args:
            data: Data for the report
            report_config: Configuration for the report
        
        Returns:
            Report creation result
        """
        start_time = datetime.utcnow()
        
        try:
            self.graphics_metrics["total_graphics"] += 1
            
            # Validate report configuration
            required_sections = report_config.get("sections", ["summary", "analysis", "visualizations"])
            
            # Create visualizations
            visualizations = []
            
            for section in report_config.get("visualizations", []):
                chart_type = section.get("type", "bar_chart")
                chart_data = section.get("data", {})
                chart_options = section.get("options", {})
                
                chart_result = await self.create_chart(chart_data, chart_type, chart_options)
                
                if chart_result.get("success"):
                    visualizations.append({
                        "section": section.get("title", ""),
                        "chart_id": chart_result["chart_id"],
                        "chart_type": chart_type,
                        "chart": chart_result
                    })
            
            # Create text sections
            text_sections = []
            
            for section in required_sections:
                if section in data:
                    text_sections.append({
                        "title": section.title(),
                        "content": data[section],
                        "type": "text"
                    })
            
            # Combine into report
            report_result = {
                "success": True,
                "report_id": str(uuid.uuid4()),
                "format": "html",
                "content": self._create_report_html(text_sections, visualizations, report_config),
                "visualizations": visualizations,
                "text_sections": text_sections,
                "metrics": {
                    "visualizations": len(visualizations),
                    "text_sections": len(text_sections),
                    "total_sections": len(required_sections)
                },
                "creation_time": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store report
            await self._store_report(report_result)
            
            # Update metrics
            creation_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_graphics_metrics("report", creation_time, True)
            
            return report_result
            
        except Exception as e:
            logger.error(f"❌ Report creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "creation_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    def _create_report_html(self, text_sections: List[Dict[str, Any]], visualizations: List[Dict[str, Any]], report_config: Dict[str, Any]) -> str:
        """Create HTML report with Tufte styling"""
        
        html_parts = [
            """<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>GRETA PAI Report</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        line-height: 1.6;
                        color: #333333;
                        background-color: #ffffff;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 40px;
                    }
                    
                    .tufte-title {
                        font-size: 20px;
                        font-weight: 600;
                        color: #1a1a1a;
                        margin-bottom: 20px;
                        text-align: center;
                    }
                    
                    .tufte-section {
                        margin-bottom: 30px;
                    }
                    
                    .tufte-section h2 {
                        font-size: 16px;
                        font-weight: 500;
                        color: #1a1a1a;
                        margin-bottom: 15px;
                        border-bottom: 1px solid #e0e0e0;
                        padding-bottom: 15px;
                    }
                    
                    .tufte-chart {
                        margin: 20px 0;
                        text-align: center;
                    }
                    
                    .tufte-caption {
                        font-size: 12px;
                        color: #666666;
                        font-style: italic;
                        margin-top: 10px;
                        text-align: center;
                    }
                    
                    .tufte-source {
                        font-size: 10px;
                        color: #999999;
                        margin-top: 10px;
                        text-align: center;
                    }
                    
                    .tufte-footnote {
                        font-size: 9px;
                        color: #999999;
                        margin-top: 20px;
                        border-top: 1px solid #e0e0e0;
                        padding-top: 10px;
                        font-style: italic;
                    }
                </style>
            </head>
            <body>
                <div class="tufte-title">
                    GRETA PAI Report
                </div>
            </body>
            </html>
        """
        
        # Add sections and visualizations
        html_parts.append('</body>')
        
        for section in text_sections:
            html_parts.append(f'''
                <div class="tufte-section">
                    <h2>{section["title"]}</h2>
                    <div>{section["content"]}</div>
                </div>
            ''')
        
        for viz in visualizations:
            html_parts.append(f'''
                <div class="tufte-chart">
                    <div class="tufte-caption">{viz.get("section_title", "Visualization")}</div>
                    {viz["chart"]["content"]}
                </div>
            ''')
        
        return "\n".join(html_parts)
    
    async def _store_report(self, report_result: Dict[str, Any]):
        """Store report in database"""
        try:
            report_data = {
                "report_id": report_result["report_id"],
                "format": report_result["format"],
                "content": report_result["content"],
                "metadata": {
                    "created_at": report_result["timestamp"],
                    "visualizations": report_result.get("visualizations", []),
                    "text_sections": report_result.get("text_sections", []),
                    "metrics": report_result.get("metrics", {})
                }
            }
            
            # Save to database
            await self.db.insert_one("reports", report_data)
            
            # Save to file
            report_file = self.graphics_dir / f"{report_result['report_id']}.html"
            with open(report_file, 'w') as f:
                f.write(report_result["content"])
            
            logger.info(f"✅ Report saved: {report_result['report_id']}")
            
        except Exception as e:
            logger.debug(f"Failed to store report: {e}")
    
    async def create_marketing_material(self, content_data: Dict[str, Any], material_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create Tufte-styled marketing material
        
        Args:
            content_data: Data for marketing material
            material_type: Type of material (flyer, brochure, presentation, report, social_media)
            options: Additional options for material creation
        
        Returns:
            Marketing material creation result
        """
        start_time = datetime.utcnow()
        
        try:
            self.graphics_metrics["total_graphics"] += 1
            
            # Set default options based on material type
            default_options = self._get_marketing_options(material_type, options or {})
            
            # Create marketing material
            if material_type == "flyer":
                material_result = await self._create_flyer(content_data, default_options)
            elif material_type == "brochure":
                material_result = await self._create_brochure(content_data, default_options)
            elif material_type == "presentation":
                material_result = await self._create_presentation(content_data, default_options)
            elif material_type == "report":
                material_result = await self._create_tufte_report(content_data, default_options)
            elif material_type == "social_media":
                material_result = await self._create_social_media(content_data, default_options)
            else:
                material_result = await self._create_generic_marketing(content_data, material_type, default_options)
            
            # Update metrics
            creation_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_graphics_metrics(material_type, creation_time, material_result.get("success", False))
            
            return material_result
            
        except Exception as e:
            logger.error(f"❌ Marketing material creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "material_type": material_type,
                "creation_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    def _get_marketing_options(self, material_type: str, user_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get default options for marketing material type"""
        
        base_options = {
            "theme": "tufte",
            "color_palette": self.color_palette,
            "typography": self.typography,
            "layout": self.layout_principles,
            "margins": self.layout_principles["margins"],
            "padding": self.layout_principles["padding"],
            "brand_colors": {
                "primary": self.color_palette["primary"],
                "secondary": self.color_palette["secondary"],
                "accent": self.color_palette["accent"],
                "accent_alt": self.color_palette["accent_alt"]
            }
        }
        
        # Material-specific options
        if material_type == "flyer":
            base_options.update({
                "width": 850,
                "height": 1100,
                "format": "png",
                "resolution": 300,
                "bleed": True,
                "safe_zone": 50
            })
        elif material_type == "brochure":
            base_options.update({
                "width": 850,
                "height": 1100,
                "format": "pdf",
                "orientation": "landscape",
                "bleed": True
            })
        elif material_type == "presentation":
            base_options.update({
                "width": 1920,
                "height": 1080,
                "format": "pdf",
                "orientation": "landscape",
                "aspect_ratio": "16:9"
            })
        elif material_type == "report":
            base_options.update({
                "width": 1200,
                "height": 1600,
                "format": "pdf",
                "page_size": "A4"
            })
        elif material_type == "social_media":
            base_options.update({
                "platform": "generic",
                "width": 1080,
                "height": 1080,
                "format": "png",
                "social_media": True
            })
        
        base_options.update(user_options)
        return base_options
    
    async def _create_flyer(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-styled flyer"""
        
        try:
            # Extract content
            title = content_data.get("title", "")
            headline = content_data.get("headline", "")
            body_content = content_data.get("content", "")
            contact_info = content_data.get("contact", {})
            
            # Create SVG flyer
            svg_content = self._create_flyer_svg(title, headline, body_content, contact_info, options)
            
            return {
                "success": True,
                "format": "svg",
                "content": svg_content,
                "content_preview": f"Title: {title}\n\nHeadline: {headline}\n\nContent: {body_content}\n\nContact: {contact_info}",
                "data_ink_ratio": 0.85,
                "tufte_compliant": True,
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Flyer creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_flyer_svg(self, title: str, headline: str, body_content: str, contact_info: Dict[str, Any], options: Dict[str, Any]) -> str:
        """Create SVG flyer"""
        
        margin = options["margins"]
        width = options["width"]
        height = options["height"]
        color_palette = options["color_palette"]
        
        # Build SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]
        
        # Add background gradient
        svg_parts.append(
            f'<defs>'
            f'<linearGradient id="background" x1="0%" y1="0%" x2="0%" y2="100%" gradientUnits="objectBoundingBox">
                <stop offset="0%" stop-color="{color_palette["background"]}" />
                <stop offset="100%" stop-color="{color_palette["background_alt"]}" />
            </linearGradient>'
        )
        
        # Add background
        svg_parts.append(
            f'<rect width="{width}" height="{height}" fill="url(#background)" />'
        )
        
        # Title area
        title_height = height * 0.25
        
        svg_parts.append(
            f'<rect x="0" y="0" width="{width}" height="{title_height}" '
            f'fill="url(#background)" '
            f'fill-opacity="0.2"/>'
        )
        
        # Add title text
        if title:
            svg_parts.append(
                f'<text x="{width/2}" y="{title_height/2}" '
                f'font-size="24" '
                f'fill="{color_palette["primary"]}" '
                f'text-anchor="middle" '
                f'alignment-baseline="middle">'
                f'{title}</text>'
            )
        
        # Headline area
        headline_height = height * 0.15
        if headline:
            svg_parts.append(
                f'<rect x="0" y="{title_height}" width="{width}" height="{headline_height}" '
                f'fill="url(#background)" '
                f'fill-opacity="0.1"/>'
            )
            
            svg_parts.append(
                f'<text x="{width/2}" y="{title_height + headline_height/2}" '
                f'font-size="18" '
                f'fill="{color_palette["primary"]}" '
                f'text-anchor="middle" '
                f'alignment-baseline="middle">'
                f'{headline}</text>'
            )
        
        # Content area
        content_height = height * 0.5
        if body_content:
            # Add content text (simplified for demonstration)
            content_lines = body_content.split('\n')[:10]  # Limit content lines
            
            for i, line in enumerate(content_lines):
                y_pos = title_height + headline_height + content_height * 0.1 + i * (content_height / len(content_lines) * 0.08)
                
                svg_parts.append(
                    f'<text x="{width/2}" y="{y_pos}" '
                    f'font-size="14" '
                    f'fill="{color_palette["text"]}" '
                    f'text-anchor="middle" '
                    f'alignment-baseline="middle">'
                    f'{line}</text>'
                )
        
        # Contact area
        contact_height = height * 0.1
        if contact_info:
            svg_parts.append(
                f'<rect x="0" y="{height - contact_height}" width="{width}" height="{contact_height}" '
                f'fill="url(#background)" '
                f'fill-opacity="0.1"/>'
            )
            
            # Add contact information
            contact_lines = []
            if contact_info.get("email"):
                contact_lines.append(f'Email: {contact_info["email"]}')
            if contact_info.get("phone"):
                contact_lines.append(f'Phone: {contact_info["phone"]}')
            if contact_info.get("website"):
                contact_lines.append(f'Website: {contact_info["website"]}')
            if contact_info.get("address"):
                contact_lines.append(f'Address: {contact_info["address"]}')
            
            for i, line in enumerate(contact_lines):
                y_pos = height - contact_height + 15 + i * 20
                
                svg_parts.append(
                    f'<text x="{width/2}" y="{y_pos}" '
                    f'font-size="12" '
                    f'fill="{color_palette["text_light"]}" '
                    f'text-anchor="middle" '
                    f'alignment-baseline="middle">'
                    f'{line}</text>'
                )
        
        svg_parts.extend([
            f'</g>',
            f'</svg>'
        ])
        
        return "\n".join(svg_parts)
    
    async def _create_brochure(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-styled brochure"""
        
        try:
            # Extract content
            title = content_data.get("title", "")
            sections = content_data.get("sections", [])
            
            # Create PDF brochure content (simplified)
            content = self._create_brochure_content(title, sections, options)
            
            return {
                "success": True,
                "format": "pdf",
                "content": content,
                "data_ink_ratio": 0.8,
                "tufte_compliant": True,
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Brochure creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_brochure_content(self, title: str, sections: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Create brochure content"""
        
        # Create HTML content for PDF conversion
        html_content = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.4;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px;
                }}
                .title {{
                    font-size: 24px;
                    font-weight: 600;
                    color: #1a1a1a;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section h2 {{
                    font-size: 18px;
                    font-weight: 500;
                    color: #1a1a1a;
                    border-bottom: 1px solid #e0e0e0;
                    padding-bottom: 15px;
                }}
                .content {{
                    margin-bottom: 20px;
                    font-size: 14px;
                    line-height: 1.6;
                }}
            </style>
        </head>
        <body>
            <div class="title">{title}</div>
            <div class="section">
                <h2>Introduction</h2>
                <div class="content">
                    <p>This brochure contains information about GRETA PAI.</p>
                </div>
            </div>
            <div class="section">
                <h2>Features</h2>
                <div class="content">
                    <ul>
                        <li>German Precision Engineering</li>
                        <li>Local-First Privacy</li>
                        <li>Advanced AI Intelligence</li>
                        <li>Multi-Agent Coordination</li>
                        <li>Comprehensive Toolset</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>Contact</h2>
                <div class="content">
                    <p>Contact us for more information about GRETA PAI.</p>
                </div>
            </div>
            
            <div class="tufte-source">
                <p>Created with Edward Tufte's principles for clarity and precision</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    async def _create_presentation(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Tufte-styled presentation"""
        
        try:
            # Extract content
            title = content_data.get("title", "")
            slides = content_data.get("slides", [])
            
            # Create presentation content
            content = self._create_presentation_content(title, slides, options)
            
            return {
                "success": True,
                "format": "pdf",
                "content": content,
                "data_ink_ratio": 0.8,
                "tufte_compliant": True,
                "chart_size": {
                    "width": options["width"],
                    "height": options["height"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Presentation creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_presentation_content(self, title: str, slides: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Create presentation content"""
        
        html_content = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #ffffff;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px;
                }}
                .title {{
                    font-size: 28px;
                    font-weight: 600;
                    color: #1a1a1a;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .slide {{
                    margin-bottom: 40px;
                    border: 1px solid #e0e0e0;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                .slide-content {{
                    font-size: 16px;
                    line-height: 1.6;
                    color: #333;
                }}
                .slide-title {{
                    font-size: 20px;
                    font-weight: 500;
                    color: #1a1a1a;
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="title">{title}</div>
            
            {self._create_slides(slides, options)}
            
            <div class="tufte-source">
                <p>Created with Edward Tufte's principles for clarity and precision</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _create_slides(self, slides: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Create slides for presentation"""
        
        html_parts = []
        
        for i, slide in enumerate(slides):
            html_parts.append(f'''
                <div class="slide">
                    <h3>{slide.get("title", f"Slide {i+1}"}</h3>
                    <div class="slide-content">
                        <p>{slide.get("content", "")}</p>
                    </div>
                </div>
            ''')
        
        return "\n".join(html_parts)
    
    async def _create_social_media(self, content_data: Dict[str, Any], options: Dict[str, str, Any]) -> Dict[str, Any]:
        """Create Tufte-styled social media content"""
        
        try:
            platform = options.get("platform", "generic")
            
            if platform == "twitter":
                return await self._create_twitter_content(content_data, options)
            elif platform == "linkedin":
                return await self._create_linkedin_content(content_data, options)
            elif platform == "facebook":
                return await self._create_facebook_content(content_data, options)
            elif platform == "instagram":
                return await self._create_instagram_content(content_data, options)
            else:
                return await self._create_generic_social_media(content_data, platform, options)
            
        except Exception as e:
            logger.error(f"❌ Social media creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_twitter_content(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Twitter content"""
        
        text = content_data.get("text", "")
        
        return {
            "success": True,
            "platform": "twitter",
            "content": text,
            "format": "text",
            "data_ink_ratio": 0.95,
            "tufte_compliant": True,
            "chart_size": {
                "width": 1200,
                "height": 675
            }
        }
    
    def _create_linkedin_content(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create LinkedIn content"""
        
        text = content_data.get("text", "")
        
        return {
            "success": True,
            "platform": "linkedin",
            "content": text,
            "format": "text",
            "data_ink_ratio": 0.9,
            "tufte_compliant": True,
            "chart_size": {
                "width": 1200,
                "height": 675
            }
        }
    
    def _create_facebook_content(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Facebook content"""
        
        text = content_data.get("text", "")
        
        return {
            "success": True,
            "platform": "facebook",
            "content": text,
            "format": "text",
            "data_ink_ratio": 0.9,
            "tufte_compliant": True,
            "chart_size": {
                "width": 1200,
                "height": 630
            }
        }
    
    def _create_instagram_content(self, content_data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Create Instagram content"""
        
        text = content_data.get("text", "")
        image_url = content_data.get("image_url", "")
        
        content_parts = [
            f'<div class="instagram-post">',
            f'<img src="{image_url}" alt="Instagram Post" style="width: 100%; max-width: 1080px; height: auto;">',
            f'<div class="instagram-caption">{text}</div>',
            f'</div>'
        ]
        
        return {
            "success": True,
            "platform": "instagram",
            "content": "\n".join(content_parts),
            "format": "html",
            "data_ink_ratio": 0.7,  # Images typically have lower data-ink ratio
            "tufte_compliant": False,  # Social media generally has lower data-ink ratio
            "chart_size": {
                "width": 1080,
                "height": 1080
            }
        }
    
    def _create_generic_social_media(self, content_data: Dict[str, Any], platform: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create generic social media content"""
        
        content = content_data.get("text", "")
        
        return {
            "success": True,
            "platform": platform,
            "content": content,
            "format": "text",
            "data_ink_ratio": 0.8,
            "tufte_compliant": False,
            "chart_size": {
                "width": 1080,
                "height": 1080
            }
        }
    
    async def _create_generic_marketing(self, content_data: Dict[str, Any], material_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create generic marketing material"""
        
        text = content_data.get("text", "")
        
        return {
            "success": True,
            "material_type": material_type,
            "format": "html",
            "content": f"<div class=\"tufte-marketing\">\n{text}\n</div>",
            "data_ink_ratio": 0.8,
            "tufte_compliant": True,
            "chart_size": {
                "width": 800,
                "height": 600
            }
        }
    
    def _store_chart(self, chart_result: Dict[str, Any]):
        """Store chart in database"""
        try:
            chart_data = {
                "chart_id": chart_result["chart_id"],
                "chart_type": chart_result["chart_type"],
                "format": chart_result["format"],
                "content": chart_result["content"],
                "metadata": {
                    "created_at": chart_result["timestamp"],
                    "data_ink_ratio": chart_result.get("data_ink_ratio", 0),
                    "tufte_compliant": chart_result.get("tufte_compliant", False)
                }
            }
            
            # Save to database
            await self.db.insert_one("marketing_materials", chart_data)
            
            logger.info(f"✅ Marketing material saved: {chart_result['chart_id']}")
            
        except Exception as e:
            logger.debug(f"Failed to store marketing material: {e}")
    
    async def get_graphics_status(self) -> Dict[str, Any]:
        """Get graphics service status"""
        
        return {
            "status": "active",
            "service": "tufte_graphics",
            "graphics_metrics": self.graphics_metrics,
            "tufte_compliance_score": self.graphics_metrics["tufte_compliance_score"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Tufte graphics health check"""
        
        return {
            "status": "healthy",
            "service": "tufte_graphics",
            "graphics_metrics": self.graphics_metrics,
            "tufte_compliance_score": self.graphics_metrics["tufte_compliance_score"],
            "timestamp": datetime.utcnow().isoformat()
        }

# Global Tufte graphics service instance
tufte_graphics_service = TufteGraphicsService()

async def get_tufte_graphics_service() -> TufteGraphicsService:
    """Get the global Tufte graphics service instance"""
    return tufte_graphics_service
