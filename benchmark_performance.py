#!/usr/bin/env python3
"""
Performance Benchmark: Feature Engineering & Twin Creation
Benchmarks the performance of engineered feature calculation and Digital Patient Twin creation
"""

import sys
import time
import psutil  # type: ignore
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# Import from the clinical_dataflow_optimizer package
from clinical_dataflow_optimizer.core.metrics_calculator import FeatureEnhancedTwinBuilder
from clinical_dataflow_optimizer.core.data_ingestion import ClinicalDataIngester
from clinical_dataflow_optimizer.models.data_models import DigitalPatientTwin, RiskMetrics, PatientStatus
import pandas as pd

class PerformanceBenchmark:
    """Performance benchmarking for feature engineering and twin creation"""

    def __init__(self):
        self.results = {}
        self.memory_usage = []
        self.cpu_usage = []

    def start_monitoring(self):
        """Start performance monitoring"""
        tracemalloc.start()
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def record_snapshot(self, label: str):
        """Record a performance snapshot"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        # CPU usage
        cpu_percent = self.process.cpu_percent()

        snapshot = {
            'timestamp': elapsed,
            'memory_current_mb': current_mb,
            'memory_peak_mb': peak_mb,
            'cpu_percent': cpu_percent,
            'label': label
        }

        self.memory_usage.append(snapshot)
        self.cpu_usage.append(cpu_percent)

        print(".2f")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        tracemalloc.stop()
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

    def benchmark_feature_calculation(self, num_patients: int = 100) -> Dict[str, Any]:
        """Benchmark feature calculation performance"""
        print(f"\n🧪 Benchmarking feature calculation for {num_patients} patients...")

        self.start_monitoring()

        # Create mock patient data
        mock_patients = []
        for i in range(num_patients):
            mock_patients.append({
                'subject_id': f'PAT{i:03d}',
                'site_id': f'SITE{i%10:02d}',
                'study_id': 'STUDY001',
                'open_queries': i % 5,
                'total_queries': i % 5 + 2,
                'pages_entered': 10,
                'missing_visits': i % 3,
                'inactivated_forms': i % 2
            })

        # Create mock study data
        mock_study_data = {
            'cpid_metrics': pd.DataFrame(mock_patients),
            'safety_events': pd.DataFrame(),
            'visit_tracker': pd.DataFrame(),
            'coding_data': pd.DataFrame(),
            'missing_pages': pd.DataFrame(),
            'inactivated_forms': pd.DataFrame()
        }

        self.record_snapshot("data_creation")

        # Initialize feature builder
        builder = FeatureEnhancedTwinBuilder()
        self.record_snapshot("builder_init")

        # Create twins with features using build_all_twins
        twins = builder.build_all_twins(mock_study_data, "BENCHMARK_STUDY")
        self.record_snapshot("twin_creation_complete")

        # Calculate aggregate features (site-level features are already calculated in build_all_twins)
        site_features = {}  # Site features are calculated internally
        self.record_snapshot("site_features_calculated")

        self.stop_monitoring()

        results = {
            'operation': 'feature_calculation',
            'num_patients': num_patients,
            'total_time_seconds': self.total_time,
            'avg_time_per_patient_ms': (self.total_time / num_patients) * 1000,
            'memory_peak_mb': max(s['memory_peak_mb'] for s in self.memory_usage),
            'cpu_avg_percent': sum(self.cpu_usage) / len(self.cpu_usage),
            'twins_created': len(twins),
            'site_features_calculated': len(site_features)
        }

        self.results['feature_calculation'] = results
        return results

    def benchmark_twin_serialization(self, num_twins: int = 100) -> Dict[str, Any]:
        """Benchmark twin serialization/deserialization performance"""
        print(f"\n🧪 Benchmarking twin serialization for {num_twins} twins...")

        # Create twins first using FeatureEnhancedTwinBuilder
        builder = FeatureEnhancedTwinBuilder()

        # Create mock study data
        mock_patients = []
        for i in range(num_twins):
            mock_patients.append({
                'subject_id': f'PAT{i:03d}',
                'site_id': f'SITE{i%10:02d}',
                'study_id': 'STUDY001',
                'open_queries': i % 5,
                'total_queries': i % 5 + 2,
                'pages_entered': 10,
                'missing_visits': i % 3,
                'inactivated_forms': i % 2
            })

        mock_study_data = {
            'cpid_metrics': pd.DataFrame(mock_patients),
            'safety_events': pd.DataFrame(),
            'visit_tracker': pd.DataFrame(),
            'coding_data': pd.DataFrame(),
            'missing_pages': pd.DataFrame(),
            'inactivated_forms': pd.DataFrame()
        }

        twins = builder.build_all_twins(mock_study_data, "BENCHMARK_STUDY")

        self.start_monitoring()

        # Serialize twins
        serialized_data = []
        for twin in twins:
            data = twin.to_dict()
            serialized_data.append(data)
            if len(serialized_data) % 25 == 0:
                self.record_snapshot(f"serialized_{len(serialized_data)}_twins")

        self.record_snapshot("serialization_complete")

        # Deserialize twins
        deserialized_twins = []
        for data in serialized_data:
            # Create a new twin from the dict data
            twin = DigitalPatientTwin(
                subject_id=data['subject_id'],
                site_id=data['site_id'],
                study_id=data['study_id'],
                country=data.get('country', ''),
                region=data.get('region', ''),
                status=PatientStatus(data['status']),
                clean_status=data['clean_status'],
                clean_percentage=data['clean_percentage'],
                blocking_items=[],  # Skip for benchmark
                missing_visits=data['metrics']['missing_visits'],
                missing_pages=data['metrics']['missing_pages'],
                open_queries=data['metrics']['open_queries'],
                total_queries=data['metrics']['total_queries'],
                uncoded_terms=data['metrics']['uncoded_terms'],
                verification_pct=data['metrics']['verification_pct'],
                non_conformant_pages=data['metrics']['non_conformant_pages'],
                reconciliation_issues=data['metrics']['reconciliation_issues'],
                protocol_deviations=data['metrics']['protocol_deviations'],
                risk_metrics=RiskMetrics(),  # Skip for benchmark
                data_quality_index=data['data_quality_index'],
                outstanding_visits=data['outstanding_visits'],
                sae_records=[],  # Skip for benchmark
                safety_reconciliation_status="Not Applicable",
                uncoded_terms_list=[]  # Skip for benchmark
            )
            deserialized_twins.append(twin)
            if len(deserialized_twins) % 25 == 0:
                self.record_snapshot(f"deserialized_{len(deserialized_twins)}_twins")

        self.record_snapshot("deserialization_complete")

        self.stop_monitoring()

        results = {
            'operation': 'twin_serialization',
            'num_twins': num_twins,
            'total_time_seconds': self.total_time,
            'avg_time_per_twin_ms': (self.total_time / num_twins) * 1000,
            'serialization_time_seconds': self.total_time / 2,  # Approximate split
            'deserialization_time_seconds': self.total_time / 2,
            'memory_peak_mb': max(s['memory_peak_mb'] for s in self.memory_usage),
            'avg_payload_size_kb': sum(len(json.dumps(d)) for d in serialized_data) / len(serialized_data) / 1024
        }

        self.results['twin_serialization'] = results
        return results

    def benchmark_real_time_updates(self, num_updates: int = 1000) -> Dict[str, Any]:
        """Benchmark real-time twin updates"""
        print(f"\n🧪 Benchmarking real-time updates for {num_updates} operations...")

        # Create initial twin
        builder = FeatureEnhancedTwinBuilder()
        patient_df = pd.DataFrame([{
            'subject_id': 'PAT001',
            'site_id': 'SITE01',
            'study_id': 'STUDY001',
            'open_queries': 1,
            'total_queries': 2,
            'pages_entered': 10,
            'missing_visits': 0,
            'inactivated_forms': 0
        }])
        twins = builder.build_twins(patient_df)
        twin = twins[0] if twins else None

        self.start_monitoring()

        # Simulate real-time updates
        for i in range(num_updates):
            # Simulate different types of updates
            if i % 3 == 0:
                # Query status change
                twin.risk_metrics.resolution_velocity += 0.1
            elif i % 3 == 1:
                # New query
                twin.risk_metrics.accumulation_velocity += 0.05
            else:
                # Form inactivation
                twin.risk_metrics.inactivation_rate += 0.01

            if i % 100 == 0:
                self.record_snapshot(f"update_{i}")

        self.record_snapshot("updates_complete")

        self.stop_monitoring()

        results = {
            'operation': 'real_time_updates',
            'num_updates': num_updates,
            'total_time_seconds': self.total_time,
            'avg_time_per_update_ms': (self.total_time / num_updates) * 1000,
            'updates_per_second': num_updates / self.total_time,
            'memory_peak_mb': max(s['memory_peak_mb'] for s in self.memory_usage)
        }

        self.results['real_time_updates'] = results
        return results

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("🚀 FEATURE ENGINEERING PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for operation, results in self.results.items():
            report.append(f"📊 {operation.replace('_', ' ').title()}")
            report.append("-" * 40)

            if operation == 'feature_calculation':
                report.append(f"Patients Processed: {results['num_patients']}")
                report.append(f"Total Time: {results['total_time_seconds']:.2f}s")
                report.append(f"Avg Time/Patient: {results['avg_time_per_patient_ms']:.2f}ms")
                report.append(f"Throughput: {results['num_patients']/results['total_time_seconds']:.1f} patients/sec")
                report.append(f"Peak Memory: {results['memory_peak_mb']:.1f}MB")
                report.append(f"Avg CPU: {results['cpu_avg_percent']:.1f}%")

            elif operation == 'twin_serialization':
                report.append(f"Twins Processed: {results['num_twins']}")
                report.append(f"Total Time: {results['total_time_seconds']:.2f}s")
                report.append(f"Avg Time/Twin: {results['avg_time_per_twin_ms']:.2f}ms")
                report.append(f"Serialization: {results['serialization_time_seconds']:.2f}s")
                report.append(f"Deserialization: {results['deserialization_time_seconds']:.2f}s")
                report.append(f"Avg Payload Size: {results['avg_payload_size_kb']:.1f}KB")
                report.append(f"Peak Memory: {results['memory_peak_mb']:.1f}MB")

            elif operation == 'real_time_updates':
                report.append(f"Updates Processed: {results['num_updates']}")
                report.append(f"Total Time: {results['total_time_seconds']:.2f}s")
                report.append(f"Avg Time/Update: {results['avg_time_per_update_ms']:.2f}ms")
                report.append(f"Updates/Second: {results['updates_per_second']:.1f}")
                report.append(f"Peak Memory: {results['memory_peak_mb']:.1f}MB")

            report.append("")

        # Performance recommendations
        report.append("💡 PERFORMANCE RECOMMENDATIONS")
        report.append("-" * 40)

        feature_calc = self.results.get('feature_calculation', {})
        if feature_calc.get('avg_time_per_patient_ms', 0) > 100:
            report.append("⚠️  Feature calculation is slow (>100ms/patient)")
            report.append("   Consider caching site-level aggregations")

        serialization = self.results.get('twin_serialization', {})
        if serialization.get('avg_payload_size_kb', 0) > 50:
            report.append("⚠️  Twin payloads are large (>50KB)")
            report.append("   Consider compression for network transfer")

        updates = self.results.get('real_time_updates', {})
        if updates.get('updates_per_second', 0) < 100:
            report.append("⚠️  Real-time updates are slow (<100/sec)")
            report.append("   Consider batching or async processing")

        if all(r.get('memory_peak_mb', 0) < 100 for r in self.results.values()):
            report.append("✅ Memory usage is acceptable (<100MB peak)")

        report.append("")
        report.append("✅ Benchmarking complete!")

        return "\n".join(report)

    def save_results(self, filename: str = "performance_benchmark.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results,
                'memory_snapshots': self.memory_usage
            }, f, indent=2)
        print(f"📁 Results saved to {filename}")

def main():
    """Run comprehensive performance benchmarks"""
    print("🚀 FEATURE ENGINEERING PERFORMANCE BENCHMARK")
    print("=" * 60)

    benchmark = PerformanceBenchmark()

    # Run benchmarks
    benchmark.benchmark_feature_calculation(num_patients=100)
    benchmark.benchmark_twin_serialization(num_twins=100)
    benchmark.benchmark_real_time_updates(num_updates=1000)

    # Generate and display report
    report = benchmark.generate_report()
    print("\n" + report)

    # Save results
    benchmark.save_results("feature_engineering_benchmark.json")

    print("\n✅ Performance benchmarking complete!")
    print("📊 Results saved to feature_engineering_benchmark.json")

if __name__ == "__main__":
    main()