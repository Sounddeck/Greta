#!/usr/bin/env python3
"""
Vector Search Performance Benchmark

This script benchmarks the Vector Search Service to ensure query latency
meets the target of <100ms average latency.

Metrics collected:
- Minimum query latency
- Maximum query latency
- Average (mean) query latency
- P95 (95th percentile) query latency
- P99 (99th percentile) query latency
- Total benchmark execution time
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.vector_search_service import VectorSearchService


class VectorSearchBenchmark:
    """Performance benchmark for Vector Search Service"""
    
    def __init__(self):
        self.service = VectorSearchService()
        self.namespace = "benchmark"
        self.latencies: List[float] = []
        self.latencies_by_limit: Dict[int, List[float]] = {
            5: [],
            10: [],
            20: [],
            50: []
        }
        
    async def setup(self):
        """Setup: Store 100 test memories"""
        print("=" * 80)
        print("VECTOR SEARCH PERFORMANCE BENCHMARK")
        print("=" * 80)
        print("\n📊 Setup: Storing 100 test memories...")
        
        # Generate diverse test content
        test_memories = [
            # Technical content (20 memories)
            *[f"Python is a high-level programming language with dynamic typing and {i}" for i in range(5)],
            *[f"JavaScript enables interactive web applications with async/await patterns {i}" for i in range(5)],
            *[f"Machine learning models require training data and feature engineering {i}" for i in range(5)],
            *[f"Database optimization involves indexing, query planning, and caching strategies {i}" for i in range(5)],
            
            # Business content (20 memories)
            *[f"Customer relationship management systems track sales and interactions {i}" for i in range(5)],
            *[f"Project management methodologies include Agile, Scrum, and Kanban {i}" for i in range(5)],
            *[f"Financial analysis requires understanding revenue, costs, and profitability {i}" for i in range(5)],
            *[f"Marketing strategies focus on customer acquisition and retention {i}" for i in range(5)],
            
            # Scientific content (20 memories)
            *[f"Quantum mechanics describes the behavior of particles at atomic scales {i}" for i in range(5)],
            *[f"Climate change impacts global weather patterns and ecosystems {i}" for i in range(5)],
            *[f"Genetic engineering allows modification of DNA sequences {i}" for i in range(5)],
            *[f"Neuroscience studies the structure and function of the nervous system {i}" for i in range(5)],
            
            # General knowledge (20 memories)
            *[f"History teaches us about past civilizations and their achievements {i}" for i in range(5)],
            *[f"Geography explores the physical features of Earth and human societies {i}" for i in range(5)],
            *[f"Literature encompasses novels, poetry, and dramatic works {i}" for i in range(5)],
            *[f"Philosophy examines fundamental questions about existence and knowledge {i}" for i in range(5)],
            
            # Mixed content (20 memories)
            *[f"Artificial intelligence combines computer science with cognitive psychology {i}" for i in range(5)],
            *[f"Cloud computing provides scalable infrastructure for modern applications {i}" for i in range(5)],
            *[f"Cybersecurity protects systems from unauthorized access and attacks {i}" for i in range(5)],
            *[f"Data science extracts insights from large datasets using statistics {i}" for i in range(5)],
        ]
        
        # Store all memories
        start_time = time.time()
        for i, content in enumerate(test_memories):
            await self.service.store_memory(
                content=content,
                namespace=self.namespace,
                metadata={
                    "index": i,
                    "category": self._get_category(i),
                    "benchmark": True
                }
            )
            if (i + 1) % 20 == 0:
                print(f"  ✓ Stored {i + 1}/100 memories...")
        
        setup_time = time.time() - start_time
        print(f"✅ Setup complete: 100 memories stored in {setup_time:.2f}s\n")
        
    def _get_category(self, index: int) -> str:
        """Get category based on memory index"""
        if index < 20:
            return "technical"
        elif index < 40:
            return "business"
        elif index < 60:
            return "scientific"
        elif index < 80:
            return "general"
        else:
            return "mixed"
    
    async def benchmark_queries(self):
        """Run benchmark queries with different result limits"""
        print("🔍 Running benchmark queries...\n")
        
        # Test queries covering different topics
        test_queries = [
            "programming languages and software development",
            "machine learning and artificial intelligence",
            "database performance optimization",
            "customer relationship management",
            "project management methodologies",
            "quantum mechanics and physics",
            "climate change and environment",
            "cloud computing infrastructure",
            "cybersecurity and data protection",
            "data science and analytics"
        ]
        
        result_limits = [5, 10, 20, 50]
        
        for limit in result_limits:
            print(f"📈 Testing with result limit: {limit}")
            limit_latencies = []
            
            for query in test_queries:
                start_time = time.perf_counter()
                
                results = await self.service.search_memories(
                    query=query,
                    namespace=self.namespace,
                    limit=limit
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                limit_latencies.append(latency_ms)
                self.latencies.append(latency_ms)
            
            self.latencies_by_limit[limit] = limit_latencies
            avg_latency = statistics.mean(limit_latencies)
            print(f"  ✓ Average latency: {avg_latency:.2f}ms\n")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        metrics = {
            "total_queries": n,
            "min_latency_ms": min(self.latencies),
            "max_latency_ms": max(self.latencies),
            "mean_latency_ms": statistics.mean(self.latencies),
            "median_latency_ms": statistics.median(self.latencies),
            "stdev_latency_ms": statistics.stdev(self.latencies) if n > 1 else 0,
            "p95_latency_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "p99_latency_ms": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
        }
        
        # Calculate metrics by result limit
        metrics["by_limit"] = {}
        for limit, latencies in self.latencies_by_limit.items():
            if latencies:
                metrics["by_limit"][limit] = {
                    "queries": len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "mean_ms": statistics.mean(latencies),
                    "median_ms": statistics.median(latencies),
                }
        
        return metrics
    
    def print_report(self, metrics: Dict[str, Any], total_time: float):
        """Print comprehensive performance report"""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        print(f"\n📊 Overall Metrics ({metrics['total_queries']} queries):")
        print(f"  • Minimum latency:    {metrics['min_latency_ms']:>8.2f} ms")
        print(f"  • Maximum latency:    {metrics['max_latency_ms']:>8.2f} ms")
        print(f"  • Average latency:    {metrics['mean_latency_ms']:>8.2f} ms")
        print(f"  • Median latency:     {metrics['median_latency_ms']:>8.2f} ms")
        print(f"  • Std deviation:      {metrics['stdev_latency_ms']:>8.2f} ms")
        print(f"  • P95 latency:        {metrics['p95_latency_ms']:>8.2f} ms")
        print(f"  • P99 latency:        {metrics['p99_latency_ms']:>8.2f} ms")
        
        print(f"\n📈 Performance by Result Limit:")
        for limit in [5, 10, 20, 50]:
            if limit in metrics["by_limit"]:
                m = metrics["by_limit"][limit]
                print(f"\n  Limit {limit} results ({m['queries']} queries):")
                print(f"    • Min:     {m['min_ms']:>8.2f} ms")
                print(f"    • Max:     {m['max_ms']:>8.2f} ms")
                print(f"    • Average: {m['mean_ms']:>8.2f} ms")
                print(f"    • Median:  {m['median_ms']:>8.2f} ms")
        
        print(f"\n⏱️  Total benchmark time: {total_time:.2f}s")
        
        # Performance evaluation
        print("\n" + "=" * 80)
        print("PERFORMANCE EVALUATION")
        print("=" * 80)
        
        target_latency = 100.0  # ms
        mean_latency = metrics['mean_latency_ms']
        p95_latency = metrics['p95_latency_ms']
        
        if mean_latency < target_latency:
            print(f"\n✅ PASS: Average latency ({mean_latency:.2f}ms) is under target ({target_latency}ms)")
            margin = target_latency - mean_latency
            print(f"   Performance margin: {margin:.2f}ms ({(margin/target_latency)*100:.1f}% under target)")
        else:
            print(f"\n❌ FAIL: Average latency ({mean_latency:.2f}ms) exceeds target ({target_latency}ms)")
            excess = mean_latency - target_latency
            print(f"   Performance deficit: {excess:.2f}ms ({(excess/target_latency)*100:.1f}% over target)")
        
        if p95_latency < target_latency:
            print(f"✅ PASS: P95 latency ({p95_latency:.2f}ms) is under target ({target_latency}ms)")
        else:
            print(f"⚠️  WARNING: P95 latency ({p95_latency:.2f}ms) exceeds target ({target_latency}ms)")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if mean_latency < 50:
            print("   • Excellent performance! System is well-optimized.")
        elif mean_latency < target_latency:
            print("   • Good performance. System meets target requirements.")
        else:
            print("   • Consider optimizing:")
            print("     - Use batch queries for multiple searches")
            print("     - Implement query result caching")
            print("     - Optimize embedding model selection")
            print("     - Consider hardware upgrades (SSD, more RAM)")
        
        print("\n" + "=" * 80)
    
    async def cleanup(self):
        """Cleanup: Remove benchmark namespace"""
        print("\n🧹 Cleanup: Removing benchmark data...")
        # Note: ChromaDB doesn't have a direct delete collection method in the simple API
        # We'll just note that the benchmark namespace should be cleaned manually if needed
        print("✅ Benchmark complete. Namespace 'benchmark' can be cleaned manually if needed.\n")
    
    async def run(self):
        """Run complete benchmark"""
        start_time = time.time()
        
        try:
            await self.setup()
            await self.benchmark_queries()
            
            total_time = time.time() - start_time
            metrics = self.calculate_metrics()
            self.print_report(metrics, total_time)
            
            await self.cleanup()
            
            # Return exit code based on performance
            return 0 if metrics['mean_latency_ms'] < 100.0 else 1
            
        except Exception as e:
            print(f"\n❌ Benchmark failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1


async def main():
    """Main entry point"""
    benchmark = VectorSearchBenchmark()
    exit_code = await benchmark.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())

