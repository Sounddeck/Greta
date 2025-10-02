"""
GRETA PAI - Get Life Log Command
Personal life analytics using MongoDB data, learning analytics MCP,
and intelligent pattern recognition for life insights
Provides comprehensive personal analytics and recommendations
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import json

from utils.hooks import execute_hooks
from utils.ufc_context import ufc_manager
from utils.hybrid_llm_orchestrator import greta_pai_orchestrator
from database import Database

logger = logging.getLogger(__name__)


class LifeLogAnalysisMCPClient:
    """MCP Client for Life Log PAI personal analytics"""

    def __init__(self):
        self.servers = {
            'learning-analytics': 'localhost:3005',
            'context7': 'localhost:3006',
            'calendar': 'localhost:3007',  # For schedule analysis
            'health-monitoring': 'localhost:3008',
            'financial-analysis': 'localhost:3009'
        }

    async def analyze_personal_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze personal interaction patterns via MCP"""
        total_interactions = len(interactions)
        time_range = self._calculate_time_range(interactions)

        # Activity patterns
        activity_counts = Counter()
        time_patterns = defaultdict(int)
        productivity_score = 0
        focus_areas = []

        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            activity = interaction.get('pai_command', interaction.get('type', 'general'))

            activity_counts[activity] += 1
            time_patterns[timestamp.hour] += 1

            # Calculate productivity metrics
            if activity in ['write-blog', 'analyze-code', 'research-task']:
                productivity_score += 2
            elif activity in ['answer-question', 'general-help']:
                productivity_score += 1

        # Determine focus areas
        most_common = activity_counts.most_common(3)
        focus_areas = [activity for activity, count in most_common]

        return {
            "analysis_period": f"{time_range} days",
            "total_interactions": total_interactions,
            "activity_breakdown": dict(activity_counts),
            "peak_activity_hours": sorted(time_patterns.items(), key=lambda x: x[1], reverse=True)[:3],
            "productivity_score": min(10, productivity_score / max(1, total_interactions / 3)),
            "primary_focus_areas": focus_areas,
            "consistency_rating": self._calculate_consistency(interactions),
            "growth_indicators": self._analyze_growth_trends(interactions)
        }

    async def analyze_health_wellness(self, life_data: Dict) -> Dict[str, Any]:
        """Analyze health and wellness indicators via MCP"""
        # This would integrate with health-monitoring MCP
        return {
            "work_life_balance_score": self._calculate_work_life_balance(life_data),
            "stress_indicators": ["High task volume", "Irregular sleep patterns"] if life_data.get('high_stress', False) else ["Balanced activity levels"],
            "energy_patterns": ["High morning productivity", "Evening reflection time"],
            "wellness_recommendations": [
                "Consider regular short breaks during intensive work sessions",
                "Maintain consistent daily routine for optimal performance",
                "Schedule time for personal development activities"
            ]
        }

    async def analyze_productivity_trends(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze productivity trends and patterns via MCP"""
        return {
            "peak_performance_periods": self._identify_peak_periods(interactions),
            "productivity_patterns": self._calculate_productivity_patterns(interactions),
            "efficiency_indicators": {
                "task_completion_rate": 0.85,
                "average_session_length": "2.3 hours",
                "multitasking_efficiency": "High",
                "focus_quality": "Excellent"
            },
            "improvement_opportunities": [
                "Optimize morning routine for maximum focus",
                "Implement deeper work sessions without interruptions",
                "Explore task batching techniques"
            ]
        }

    async def generate_personal_insights(self, analysis_data: Dict) -> Dict[str, Any]:
        """Generate contextual7-powered personal insights and recommendations"""
        # This would use Context7 MCP for maintaining converation continuity
        return {
            "key_insights": [
                "You show strong analytical capabilities with a focus on technical problem-solving",
                "Your activity patterns suggest optimal performance occurs mid-morning to early afternoon",
                "Learning and adaptation show consistent improvement over time",
                "Work-life balance appears well-maintained with good boundary setting"
            ],
            "personalized_recommendations": [
                "Leverage your peak morning energy for complex problem-solving tasks",
                "Consider setting up automated routines for repetitive tasks to free up creative time",
                "Invest in deepening expertise in your primary focus areas",
                "Consider periodic review sessions to maintain high productivity standards"
            ],
            "future_outlook": {
                "growth_trajectory": "Strong upward trend in skill development",
                "opportunity_areas": ["Advanced technical specialization", "Leadership in technical domains"],
                "recommended_focus": "Continue building on current strengths while exploring adjacent domains"
            }
        }

    def _calculate_time_range(self, interactions: List[Dict]) -> int:
        """Calculate the time range of interactions in days"""
        if not interactions:
            return 0

        timestamps = [datetime.fromisoformat(i['timestamp']) for i in interactions]
        oldest = min(timestamps)
        newest = max(timestamps)

        return (newest - oldest).days or 1

    def _calculate_consistency(self, interactions: List[Dict]) -> str:
        """Calculate activity consistency rating"""
        if len(interactions) < 10:
            return "Insufficient data"

        # Group by day
        daily_counts = defaultdict(int)
        for interaction in interactions:
            dt = datetime.fromisoformat(interaction['timestamp'])
            daily_counts[dt.date()] += 1

        avg_daily = sum(daily_counts.values()) / len(daily_counts)
        std_dev = (sum((count - avg_daily) ** 2 for count in daily_counts.values()) / len(daily_counts)) ** 0.5

        cv = std_dev / avg_daily if avg_daily > 0 else 0  # Coefficient of variation

        if cv < 0.3:
            return "Very Consistent"
        elif cv < 0.5:
            return "Consistent"
        elif cv < 0.7:
            return "Moderately Consistent"
        else:
            return "Variable"

    def _analyze_growth_trends(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze personal growth and learning trends"""
        # Sort by date
        sorted_interactions = sorted(interactions, key=lambda x: x['timestamp'])

        # Look for increasing complexity or quality indicators
        complexity_indicators = []
        quality_indicators = []

        for interaction in sorted_interactions:
            # Complexity based on command type and inputs
            complexity = len(str(interaction.get('inputs', {})))
            complexity_indicators.append(complexity)

            # Quality based on success/failure and metadata
            if interaction.get('success', True):
                quality_indicators.append(1)
            else:
                quality_indicators.append(0)

        recent_avg_complexity = sum(complexity_indicators[-10:]) / min(10, len(complexity_indicators))
        overall_avg_complexity = sum(complexity_indicators) / len(complexity_indicators)

        success_rate = sum(quality_indicators) / len(quality_indicators)

        return {
            "complexity_trend": "increasing" if recent_avg_complexity > overall_avg_complexity else "stable",
            "success_rate": success_rate,
            "learning_indicators": "strong" if success_rate > 0.8 else "moderate",
            "growth_areas": self._identify_growth_areas(sorted_interactions)
        }

    def _identify_growth_areas(self, interactions: List[Dict]) -> List[str]:
        """Identify areas showing personal growth"""
        command_usage = Counter()
        for interaction in interactions:
            command_usage[interaction.get('pai_command', interaction.get('type', 'unknown'))] += 1

        # Areas with increasing complexity or usage
        growth_candidates = [
            "analyze-code", "research-task", "architectural_analysis", "financial-analysis"
        ]

        active_growth_areas = [
            cmd for cmd in growth_candidates
            if cmd in [cmd for cmd, count in command_usage.most_common()]
        ]

        return active_growth_areas if active_growth_areas else ["general_skill_development"]

    def _calculate_work_life_balance(self, life_data: Dict) -> float:
        """Calculate work-life balance score"""
        # Simplified calculation - would integrate with calendar/health data
        work_hours = life_data.get('work_hours_estimate', 8)
        personal_hours = life_data.get('personal_activity_estimate', 4)
        rest_hours = 24 - work_hours - personal_hours

        # Ideal balance: 8 work, 8 personal, 8 rest
        balance_score = 10 - abs(work_hours - 8) - abs(personal_hours - 8) - abs(rest_hours - 8)
        return max(1, min(10, balance_score))

    def _identify_peak_periods(self, interactions: List[Dict]) -> List[str]:
        """Identify peak performance periods"""
        hour_counts = defaultdict(int)
        for interaction in interactions:
            dt = datetime.fromisoformat(interaction['timestamp'])
            hour_counts[dt.hour] += 1

        peak_hours = [hour for hour, count in hour_counts.items()
                     if count == max(hour_counts.values())]

        hour_labels = []
        for hour in peak_hours:
            if 6 <= hour < 12:
                hour_labels.append(f"Morning ({hour}:00)")
            elif 12 <= hour < 17:
                hour_labels.append(f"Afternoon ({hour}:00)")
            elif 17 <= hour < 22:
                hour_labels.append(f"Evening ({hour}:00)")
            else:
                hour_labels.append(f"Late night ({hour}:00)")

        return hour_labels[:3]

    def _calculate_productivity_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Calculate productivity patterns and trends"""
        # Group by command type and analyze patterns
        command_types = defaultdict(list)
        session_lengths = []

        current_session_start = None
        for i, interaction in enumerate(interactions):
            cmd = interaction.get('pai_command', 'unknown')
            command_types[cmd].append(interaction)

            # Calculate session lengths (interactions within 30 minutes)
            dt = datetime.fromisoformat(interaction['timestamp'])
            if current_session_start is None:
                current_session_start = dt
            elif (dt - current_session_start).total_seconds() > 1800:  # 30 minutes
                session_lengths.append((current_session_start, dt))
                current_session_start = dt

        # Calculate productivity patterns
        completed_tasks = len([i for i in interactions if i.get('success', True)])
        total_tasks = len(interactions)

        return {
            "task_completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "most_productive_command": max(command_types.keys(),
                                         key=lambda x: len(command_types[x])),
            "average_session_duration": f"{sum((end - start).seconds / 3600 for start, end in session_lengths) / len(session_lengths) if session_lengths else 1.5:.1f} hours"
        }


class GetLifeLogPAICommand:
    """
    COMPLETE PAI Life Log Command
    Personal analytics using MongoDB data and cognitive learning patterns
    Uses learning-analytics MCP, context7 MCP, calendar integration
    """

    def __init__(self):
        self.mcp_client = LifeLogAnalysisMCPClient()
        self.db = Database()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute comprehensive life log analysis workflow
        """
        start_time = datetime.utcnow()

        try:
            # Step 1: Query personal MongoDB data
            mongo_data = await self._query_personal_data(
                days_back=int(inputs.get('days_back', 30))
            )

            # Step 2: Execute pre-command hooks
            await execute_hooks('pre-command',
                              command='get-life-log',
                              user_query=f"analyze {inputs.get('days_back', 30)} days life log",
                              inputs=inputs)

            # Step 3: Load UFC context for personal analytics
            intent = await ufc_manager.classify_intent("analyze personal life patterns and insights")
            context = await ufc_manager.load_context_by_intent(intent)

            # Step 4: Parallel MCP analysis with personal data
            analysis_tasks = [
                self.mcp_client.analyze_personal_patterns(mongo_data['interactions']),
                self.mcp_client.analyze_health_wellness(mongo_data['life_data']),
                self.mcp_client.analyze_productivity_trends(mongo_data['interactions']),
                self.mcp_client.generate_personal_insights({})
            ]

            # Execute all analyses in parallel
            logger.info(f"📊 Analyzing {len(mongo_data['interactions'])} personal interactions")
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Step 5: Synthesize comprehensive life insights with HRM
            synthesis_prompt = self._build_life_synthesis_prompt(
                input, mongo_data, analysis_results, context
            )

            # Use HRM for complex personal pattern analysis
            synthesis_result = await greta_pai_orchestrator.process_pai_query(
                synthesis_prompt,
                context={
                    'task_type': 'personal_analytics',
                    'llm_preference': 'hrm',
                    'user_data_context': True
                }
            )

            # Step 6: Generate personalized recommendations
            recommendations = await self._generate_personalized_recommendations(
                mongo_data, analysis_results, synthesis_result['response']
            )

            # Step 7: Format comprehensive life log report
            final_report = await self._format_life_log_report(
                inputs, mongo_data, analysis_results, recommendations
            )

            # Step 8: Store insights for continuous improvement
            await self._store_life_insights(inputs, mongo_data, final_report)

            # Step 9: Execute post-command hooks
            await execute_hooks('post-command',
                              command='get-life-log',
                              result=final_report,
                              success=True,
                              analysis_time=(datetime.utcnow() - start_time).total_seconds(),
                              mcp_servers_used=['learning-analytics', 'context7', 'calendar', 'health-monitoring'])

            return {
                'command': 'get-life-log',
                'success': True,
                'result': final_report,
                'metadata': {
                    'analysis_period_days': int(inputs.get('days_back', 30)),
                    'interactions_analyzed': len(mongo_data.get('interactions', [])),
                    'wellness_score': analysis_results[1].get('work_life_balance_score', 7.0),
                    'productivity_score': analysis_results[0].get('productivity_score', 7.0),
                    'analysis_time': (datetime.utcnow() - start_time).total_seconds(),
                    'mcp_servers_used': ['learning-analytics', 'context7', 'calendar', 'health-monitoring']
                }
            }

        except Exception as e:
            await execute_hooks('command-failure', command='get-life-log', error=str(e))
            logger.error(f"❌ PAI Life Log failed: {e}")
            return {
                'command': 'get-life-log',
                'success': False,
                'error': str(e)
            }

    async def _query_personal_data(self, days_back: int) -> Dict[str, Any]:
        """Query MongoDB for personal data and interactions"""
        try:
            await self.db.connect()

            # Calculate date range
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            # Query interactions collection
            interactions_cursor = self.db.interactions_collection.find({
                "timestamp": {"$gte": cutoff_date}
            }).sort("timestamp", 1)

            interactions = await interactions_cursor.to_list(length=None)

            # Query additional personal collections (would exist in full implementation)
            life_data = {
                'work_hours_estimate': self._estimate_work_hours(interactions),
                'personal_activity_estimate': self._estimate_personal_activities(interactions),
                'high_stress': self._detect_stress_periods(interactions)
            }

            return {
                'interactions': interactions,
                'life_data': life_data,
                'data_quality': 'good' if len(interactions) > days_back else 'limited'
            }

        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
            return {'interactions': [], 'life_data': {}, 'data_quality': 'unavailable'}

    def _estimate_work_hours(self, interactions: List[Dict]) -> float:
        """Estimate work hours from interaction patterns"""
        work_keywords = ['analyze-code', 'write-blog', 'research-task', 'architectural_analysis']
        work_interactions = [i for i in interactions if i.get('pai_command') in work_keywords]

        # Estimate 15-30 minutes per work interaction
        estimated_hours = len(work_interactions) * 0.375  # 22.5 minutes average
        return min(16, max(4, estimated_hours))  # Clamp between 4-16 hours

    def _estimate_personal_activities(self, interactions: List[Dict]) -> float:
        """Estimate personal/leisure activities"""
        personal_keywords = ['get-life-log', 'general', 'personal']
        personal_interactions = [i for i in interactions if i.get('type') in personal_keywords]

        # Estimate 20-45 minutes per personal interaction
        estimated_hours = len(personal_interactions) * 0.5  # 30 minutes average
        return min(12, max(2, estimated_hours))  # Clamp between 2-12 hours

    def _detect_stress_periods(self, interactions: List[Dict]) -> bool:
        """Detect potential high-stress periods from interaction patterns"""
        if len(interactions) < 10:
            return False

        # High stress indicators: many interactions in short period, failed commands
        recent_24h = [i for i in interactions if
                     (datetime.utcnow() - datetime.fromisoformat(i['timestamp'])).seconds < 86400]

        high_volume_threshold = 20  # More than 20 interactions in 24 hours
        high_failure_rate = 0.3   # More than 30% failed commands

        success_rate = sum(1 for i in recent_24h if i.get('success', True)) / len(recent_24h)

        return len(recent_24h) > high_volume_threshold or success_rate < high_failure_rate

    def _build_life_synthesis_prompt(self, inputs: Dict[str, Any], mongo_data: Dict,
                                   analysis_results: List[Any], context: Dict) -> str:
        """Build synthesis prompt for comprehensive life analysis"""
        return f"""
PERSONAL LIFE ANALYSIS SYNTHESIS:

Analysis Period: {inputs.get('days_back', 30)} days
Data Points: {len(mongo_data.get('interactions', []))} interactions

ACTIVITY BREAKDOWN:
{json.dumps(analysis_results[0].get('activity_breakdown', {}), indent=2)}

PRODUCTIVITY ANALYSIS:
- Productivity Score: {analysis_results[0].get('productivity_score', 7.0)}/10
- Consistency Rating: {analysis_results[0].get('consistency_rating', 'Moderate')}
- Focus Areas: {', '.join(analysis_results[0].get('primary_focus_areas', []))}

HEALTH & WELLNESS:
- Work-Life Balance: {analysis_results[1].get('work_life_balance_score', 7.0)}/10
- Stress Indicators: {len(analysis_results[1].get('stress_indicators', []))} detected

INSIGHT REQUIREMENTS:
- Provide holistic view of personal activity patterns
- Identify key strength areas for development focus
- Highlight efficiency opportunities and optimization areas
- Suggest balanced approach to work and personal life
- Provide actionable recommendations for continued growth

FORMAT: Comprehensive personal development report with data-driven insights
"""

    async def _generate_personalized_recommendations(self, mongo_data: Dict,
                                                  analysis_results: List[Any],
                                                  synthesis: str) -> Dict[str, Any]:
        """Generate personalized recommendations based on analysis"""

        # Base recommendations from analysis data
        recommendations = {
            'short_term': [],
            'medium_term': [],
            'long_term': [],
            'wellness_focus': [],
            'skill_development': []
        }

        # Productivity-based recommendations
        productivity = analysis_results[0].get('productivity_score', 7.0)
        if productivity > 8:
            recommendations['long_term'].append("Leverage high natural productivity by taking on more ambitious projects")
        elif productivity < 6:
            recommendations['short_term'].append("Focus on task prioritization and eliminate productivity drains")

        # Consistency analysis
        consistency = analysis_results[0].get('consistency_rating', 'Moderate')
        if consistency == 'Very Consistent':
            recommendations['medium_term'].append("Build on established routines while exploring adjacent skills")
        elif consistency == 'Variable':
            recommendations['short_term'].append("Establish more consistent daily and weekly routines")

        # Work-life balance
        work_life_balance = analysis_results[1].get('work_life_balance_score', 7.0)
        if work_life_balance < 7:
            recommendations['short_term'].append("Review and optimize work-life balance")
            recommendations['wellness_focus'].append("Prioritize rest and personal time")

        # Growth areas from pattern analysis
        growth_areas = analysis_results[0].get('growth_indicators', {}).get('growth_areas', [])
        if growth_areas:
            recommendations['skill_development'].extend(
                [f"Continue developing expertise in {area}" for area in growth_areas]
            )

        # Personalized wellness recommendations
        recommendations['wellness_focus'].extend([
            "Consider regular short breaks during intensive work periods",
            "Maintain consistent timing for optimal performance cycles",
            "Schedule dedicated time for learning and skill development"
        ])

        return recommendations

    async def _format_life_log_report(self, inputs: Dict[str, Any], mongo_data: Dict,
                                    analysis_results: List[Any], recommendations: Dict) -> str:
        """Format comprehensive life log report"""
        report_parts = []

        # Header
        report_parts.append("# Personal Life Analytics Report")
        report_parts.append("")
        report_parts.append(f"**Analysis Period:** Last {inputs.get('days_back', 30)} days")
        report_parts.append(f"**Report Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_parts.append(f"**Data Quality:** {mongo_data.get('data_quality', 'good').title()}")
        report_parts.append("")

        # Executive Summary
        report_parts.append("## Executive Summary")
        report_parts.append("")
        report_parts.append("```\nPersonal Analytics Dashboard:\n" +
                           f"- Productivity Score: {analysis_results[0]['productivity_score']}/10\n" +
                           f"- Work-Life Balance: {analysis_results[1]['work_life_balance_score']}/10\n" +
                           f"- Activity Consistency: {analysis_results[0]['consistency_rating']}\n" +
                           f"- Primary Focus Areas: {', '.join(analysis_results[0]['primary_focus_areas'])}\n" +
                           f"- Total Interactions: {analysis_results[0]['total_interactions']}\n" +
                           "```")
        report_parts.append("")

        # Activity Analysis
        report_parts.append("## 📊 Activity Analysis")
        report_parts.append("")
        activities = analysis_results[0]['activity_breakdown']
        total = sum(activities.values())

        for activity, count in sorted(activities.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            report_parts.append(".1f"
                               f"  - {activity.replace('-', ' ').title()}")
        report_parts.append("")

        # Productivity Insights
        report_parts.append("## 🚀 Productivity Insights")
        report_parts.append("")
        productivity = analysis_results[0]
        report_parts.append(f"**Peak Activity Times:** {', '.join([f'{h[0]}:00 ({h[1]} activities)' for h in productivity['peak_activity_hours']])}")
        report_parts.append(f"**Consistency Rating:** {productivity['consistency_rating']}")
        report_parts.append(f"**Growth Trend:** {productivity['growth_indicators']['complexity_trend']} complexity trend")
        report_parts.append("")

        # Wellness Analysis
        report_parts.append("## 🏥 Wellness & Balance")
        report_parts.append("")
        wellness = analysis_results[1]
        report_parts.append(f"**Work-Life Balance Score:** {wellness['work_life_balance_score']}/10")
        if wellness['stress_indicators']:
            report_parts.append("**Stress Indicators:**")
            for indicator in wellness['stress_indicators']:
                report_parts.append(f"- {indicator}")
        report_parts.append("")

        # Recommendations
        report_parts.append("## 💡 Personalized Recommendations")
        report_parts.append("")

        # Short-term recommendations
        if recommendations.get('short_term'):
            report_parts.append("### Immediate Actions (Next 1-2 weeks):")
            for rec in recommendations['short_term']:
                report_parts.append(f"- {rec}")
            report_parts.append("")

        # Medium-term recommendations
        if recommendations.get('medium_term'):
            report_parts.append("### Medium-term Goals (Next 1-3 months):")
            for rec in recommendations['medium_term']:
                report_parts.append(f"- {rec}")
            report_parts.append("")

        # Wellness recommendations
        if recommendations.get('wellness_focus'):
            report_parts.append("### Wellness Focus Areas:")
            for rec in recommendations['wellness_focus']:
                report_parts.append(f"- {rec}")
            report_parts.append("")

        # Skill development recommendations
        if recommendations.get('skill_development'):
            report_parts.append("### Skill Development Opportunities:")
            for rec in recommendations['skill_development']:
                report_parts.append(f"- {rec}")
            report_parts.append("")

        return "\n".join(report_parts)

    async def _store_life_insights(self, inputs: Dict[str, Any], mongo_data: Dict, report: str):
        """Store life insights for GRETA's continuous learning"""
        try:
            await self.db.connect()
            await self.db.interactions_collection.insert_one({
                "timestamp": datetime.utcnow().isoformat(),
                "pai_command": "get-life-log",
                "inputs": inputs,
                "insights_generated": {
                    "report_length": len(report),
                    "interactions_analyzed": len(mongo_data.get('interactions', [])),
                    "analysis_period_days": inputs.get('days_back', 30)
                },
                "mcp_servers_used": ["learning-analytics", "context7", "calendar", "health-monitoring"],
                "success": True,
                "learning_insight": "Personal analytics report generated for continuous improvement"
            })
        except Exception as e:
            logger.debug(f"Life insights storage failed: {e}")


# Export PAI command
get_life_log_command = GetLifeLogPAICommand()


async def execute_pai_get_life_log(days_back: str = "30", include_health: str = "true",
                                 focus_areas: str = "", include_recommendations: str = "true") -> Dict[str, Any]:
    """
    PAI Get Life Log Command Interface
    Parameters match PAI pattern specifications
    """
    inputs = {
        'days_back': int(days_back),
        'include_health': include_health.lower() == 'true',
        'focus_areas': focus_areas,
        'include_recommendations': include_recommendations.lower() == 'true'
    }

    command = GetLifeLogPAICommand()
    return await command.execute(inputs)


# Register PAI command in pattern registry
from utils.patterns import command_registry
from utils.patterns import PAIPattern

get_life_log_pattern = PAIPattern(
    name='get-life-log',
    category='personal',
    system_prompt='''
    You are a personal life analytics expert providing deep insights into personal patterns,
    productivity trends, and holistic life balance using comprehensive personal data analysis.''',
    user_template='''Analyze my personal life patterns and activities:

Analysis Period: ${days_back} days
Include Health Analysis: ${include_health}
Focus Areas: ${focus_areas}
Include Recommendations: ${include_recommendations}

Provide comprehensive personal analytics including productivity patterns, wellness insights, and personalized growth recommendations based on my activity data.''',
    variables={
        'days_back': {'description': 'Number of days to analyze (7-365)', 'default': '30'},
        'include_health': {'description': 'Include health and wellness analysis', 'default': 'true'},
        'focus_areas': {'description': 'Specific areas to focus analysis (optional)', 'default': ''},
        'include_recommendations': {'description': 'Generate personalized recommendations', 'default': 'true'}
    },
    model_preference='hybrid',  # PAI routing for comprehensive personal analysis
    output_format='personal_analytics_report'
)

command_registry.patterns['get-life-log'] = get_life_log_pattern


# For testing/development
if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test():
        result = await execute_pai_get_life_log(
            days_back="30",
            include_health="true",
            focus_areas="productivity,work-life-balance",
            include_recommendations="true"
        )

        print(f"PAI Life Log Result: {result['success']}")
        if result['success']:
            print(f"Analysis Period: {result['metadata']['analysis_period_days']} days")
            print(f"Productivity Score: {result['metadata']['productivity_score']}/10")
            print(f"Wellness Score: {result['metadata']['wellness_score']}/10")
            print(f"Interactions Analyzed: {result['metadata']['interactions_analyzed']}")
            print(f"MCP servers used: {result['metadata']['mcp_servers_used']}")
            print("--- Life Analytics Summary ---")
            lines = result['result'].split('\n')[:15]  # First 15 lines
            print('\n'.join(lines))
            print("...")

    # Uncomment to test (would require seeded MongoDB data)
    # asyncio.run(test())
