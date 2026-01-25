"""
LongCat API Timeout Diagnosis and Testing Script
Diagnoses, reproduces, and validates fixes for LongCat API timeout failures
"""

import requests
import json
import logging
import time
import socket
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.longcat_integration import LongCatClient
from config.settings import LongCatConfig, DEFAULT_LONGCAT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongCatTimeoutDiagnoser:
    """
    Comprehensive diagnosis tool for LongCat API timeout issues
    """

    def __init__(self):
        self.config = DEFAULT_LONGCAT_CONFIG
        self.test_results = []

    def run_full_diagnosis(self) -> Dict[str, Any]:
        """
        Run complete diagnosis suite
        """
        logger.info("="*60)
        logger.info("LONGCAT API TIMEOUT DIAGNOSIS")
        logger.info("="*60)

        results = {
            'network_connectivity': self.test_network_connectivity(),
            'dns_resolution': self.test_dns_resolution(),
            'tls_handshake': self.test_tls_handshake(),
            'latency_measurement': self.test_latency_measurement(),
            'timeout_reproduction': self.test_timeout_reproduction(),
            'payload_analysis': self.test_payload_analysis(),
            'concurrency_test': self.test_concurrency_impact(),
            'recommendations': []
        }

        # Generate recommendations
        results['recommendations'] = self.generate_recommendations(results)

        return results

    def test_network_connectivity(self) -> Dict[str, Any]:
        """Test basic network connectivity to api.longcat.chat"""
        logger.info("\n[1] TESTING NETWORK CONNECTIVITY")

        try:
            # Test basic connectivity with short timeout
            response = requests.get(
                "https://api.longcat.chat/health",
                timeout=5,
                verify=True
            )
            return {
                'status': 'SUCCESS',
                'response_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'error': None
            }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'FAILED',
                'response_code': None,
                'response_time': None,
                'error': str(e)
            }

    def test_dns_resolution(self) -> Dict[str, Any]:
        """Test DNS resolution for api.longcat.chat"""
        logger.info("\n[2] TESTING DNS RESOLUTION")

        import socket
        try:
            start_time = time.time()
            ip_address = socket.gethostbyname('api.longcat.chat')
            resolution_time = time.time() - start_time

            return {
                'status': 'SUCCESS',
                'ip_address': ip_address,
                'resolution_time': resolution_time,
                'error': None
            }
        except socket.gaierror as e:
            return {
                'status': 'FAILED',
                'ip_address': None,
                'resolution_time': None,
                'error': str(e)
            }

    def test_tls_handshake(self) -> Dict[str, Any]:
        """Test TLS handshake success"""
        logger.info("\n[3] TESTING TLS HANDSHAKE")

        try:
            import ssl
            context = ssl.create_default_context()
            with socket.create_connection(('api.longcat.chat', 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname='api.longcat.chat') as ssock:
                    start_time = time.time()
                    ssock.do_handshake()
                    handshake_time = time.time() - start_time

                    return {
                        'status': 'SUCCESS',
                        'handshake_time': handshake_time,
                        'cipher': ssock.cipher(),
                        'error': None
                    }
        except Exception as e:
            return {
                'status': 'FAILED',
                'handshake_time': None,
                'cipher': None,
                'error': str(e)
            }

    def test_latency_measurement(self) -> Dict[str, Any]:
        """Measure latency with multiple requests"""
        logger.info("\n[4] MEASURING LATENCY (P50/P95)")

        latencies = []
        errors = []

        for i in range(10):
            try:
                start_time = time.time()
                response = requests.head(
                    "https://api.longcat.chat",
                    timeout=10,
                    allow_redirects=True
                )
                latency = time.time() - start_time
                latencies.append(latency)
            except Exception as e:
                errors.append(str(e))
                latencies.append(10.0)  # Max timeout

            time.sleep(0.5)  # Small delay between requests

        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies)//2]
            p95 = latencies[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[0]

            return {
                'status': 'COMPLETED',
                'sample_size': len(latencies),
                'p50_latency': p50,
                'p95_latency': p95,
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'errors_count': len(errors),
                'errors': errors[:3]  # First 3 errors
            }
        else:
            return {
                'status': 'FAILED',
                'sample_size': 0,
                'error': 'No successful requests'
            }

    def test_timeout_reproduction(self) -> Dict[str, Any]:
        """Reproduce the exact timeout failure"""
        logger.info("\n[5] REPRODUCING TIMEOUT FAILURE")

        client = LongCatClient()

        # Test with the exact same payload that causes timeouts
        test_payload = {
            "context": "Patient status change analysis",
            "task": "Explain the reason for this status change",
            "data": {
                "previous_status": "Dirty",
                "new_status": "Clean",
                "blocking_factors": ["Open Query", "Missing Visit"],
                "cleanliness_score": 85.5
            }
        }

        results = []
        for attempt in range(3):
            logger.info(f"  Attempt {attempt + 1}/3")
            start_time = time.time()

            try:
                response = client.generate_agent_reasoning(**test_payload)
                elapsed = time.time() - start_time

                results.append({
                    'attempt': attempt + 1,
                    'status': 'SUCCESS',
                    'elapsed_time': elapsed,
                    'response_length': len(response) if response else 0,
                    'error': None
                })

            except Exception as e:
                elapsed = time.time() - start_time
                results.append({
                    'attempt': attempt + 1,
                    'status': 'FAILED',
                    'elapsed_time': elapsed,
                    'response_length': 0,
                    'error': str(e)
                })

            # Wait before retry
            if attempt < 2:
                time.sleep(2 ** attempt)

        return {
            'test_type': 'timeout_reproduction',
            'results': results,
            'consistent_failure': all(r['status'] == 'FAILED' for r in results),
            'timeout_pattern': any('Read timed out' in r['error'] for r in results if r['error'])
        }

    def test_payload_analysis(self) -> Dict[str, Any]:
        """Test with different payload sizes"""
        logger.info("\n[6] ANALYZING PAYLOAD IMPACT")

        client = LongCatClient()

        # Minimal payload
        minimal_payload = {
            "context": "Test",
            "task": "Respond with OK",
            "data": {"test": "minimal"}
        }

        # Full payload (similar to failing case)
        full_payload = {
            "context": "Patient status change analysis for clinical trial data quality monitoring",
            "task": "Analyze the patient status change and provide detailed reasoning about data quality implications, potential root causes, recommended actions, and risk assessment for the clinical trial integrity",
            "data": {
                "previous_status": "Dirty",
                "new_status": "Clean",
                "blocking_factors": [
                    "Open Query: Subject has 3 unresolved data queries pending response from site",
                    "Missing Visit: Scheduled visit V3 was due 15 days ago but no data entered",
                    "Uncoded Term: 2 adverse event terms require MedDRA coding",
                    "Verification Incomplete: Only 67% of forms have been source data verified"
                ],
                "cleanliness_score": 85.5,
                "site_id": "S001",
                "subject_id": "P001",
                "study_phase": "Phase 2",
                "therapeutic_area": "Oncology",
                "visit_compliance_rate": 78.5,
                "query_resolution_time_avg": 12.3,
                "data_entry_timeliness": "Delayed",
                "protocol_deviations": ["Minor scheduling deviation"],
                "safety_events": [],
                "last_updated": "2024-01-23T10:30:00Z"
            }
        }

        results = {}

        # Test minimal payload
        logger.info("  Testing minimal payload...")
        try:
            start_time = time.time()
            response = client.generate_agent_reasoning(**minimal_payload)
            elapsed = time.time() - start_time
            results['minimal'] = {
                'status': 'SUCCESS',
                'elapsed_time': elapsed,
                'payload_size': len(json.dumps(minimal_payload)),
                'response_length': len(response)
            }
        except Exception as e:
            elapsed = time.time() - start_time
            results['minimal'] = {
                'status': 'FAILED',
                'elapsed_time': elapsed,
                'payload_size': len(json.dumps(minimal_payload)),
                'error': str(e)
            }

        # Test full payload
        logger.info("  Testing full payload...")
        try:
            start_time = time.time()
            response = client.generate_agent_reasoning(**full_payload)
            elapsed = time.time() - start_time
            results['full'] = {
                'status': 'SUCCESS',
                'elapsed_time': elapsed,
                'payload_size': len(json.dumps(full_payload)),
                'response_length': len(response)
            }
        except Exception as e:
            elapsed = time.time() - start_time
            results['full'] = {
                'status': 'FAILED',
                'elapsed_time': elapsed,
                'payload_size': len(json.dumps(full_payload)),
                'error': str(e)
            }

        return results

    def test_concurrency_impact(self) -> Dict[str, Any]:
        """Test concurrent requests to identify throttling"""
        logger.info("\n[7] TESTING CONCURRENCY IMPACT")

        import concurrent.futures
        import threading

        client = LongCatClient()
        lock = threading.Lock()

        def single_request(request_id: int):
            payload = {
                "context": f"Concurrent test {request_id}",
                "task": "Respond briefly",
                "data": {"request_id": request_id}
            }

            try:
                start_time = time.time()
                response = client.generate_agent_reasoning(**payload)
                elapsed = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': 'SUCCESS',
                    'elapsed_time': elapsed,
                    'response_length': len(response)
                }
            except Exception as e:
                elapsed = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': 'FAILED',
                    'elapsed_time': elapsed,
                    'error': str(e)
                }

        # Test with different concurrency levels
        concurrency_levels = [1, 3, 5]
        results = {}

        for concurrency in concurrency_levels:
            logger.info(f"  Testing with {concurrency} concurrent requests...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(single_request, i) for i in range(concurrency)]
                concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

            success_count = sum(1 for r in concurrent_results if r['status'] == 'SUCCESS')
            avg_latency = sum(r['elapsed_time'] for r in concurrent_results) / len(concurrent_results)

            results[f'concurrency_{concurrency}'] = {
                'total_requests': concurrency,
                'successful_requests': success_count,
                'success_rate': success_count / concurrency,
                'avg_latency': avg_latency,
                'results': concurrent_results
            }

        return results

    def generate_recommendations(self, diagnosis_results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on diagnosis"""
        recommendations = []

        # Network connectivity issues
        if diagnosis_results['network_connectivity']['status'] == 'FAILED':
            recommendations.append("CRITICAL: Base network connectivity to api.longcat.chat failed. Check firewall, proxy, or DNS settings.")

        # DNS issues
        if diagnosis_results['dns_resolution']['status'] == 'FAILED':
            recommendations.append("CRITICAL: DNS resolution failed for api.longcat.chat. Check DNS configuration.")

        # TLS issues
        if diagnosis_results['tls_handshake']['status'] == 'FAILED':
            recommendations.append("CRITICAL: TLS handshake failed. Check SSL/TLS configuration and certificates.")

        # High latency
        latency = diagnosis_results['latency_measurement']
        if latency['status'] == 'COMPLETED' and latency['p95_latency'] > 5.0:
            recommendations.append(f"HIGH LATENCY: P95 latency is {latency['p95_latency']:.2f}s. Consider increasing timeout from 30s.")

        # Timeout reproduction
        timeout_test = diagnosis_results['timeout_reproduction']
        if timeout_test['consistent_failure']:
            recommendations.append("TIMEOUT ISSUE: All test requests failed with timeouts. API may be overloaded or have performance issues.")

        # Payload impact
        payload_test = diagnosis_results['payload_analysis']
        if payload_test.get('minimal', {}).get('status') == 'SUCCESS' and payload_test.get('full', {}).get('status') == 'FAILED':
            recommendations.append("PAYLOAD SIZE: Minimal payload succeeds but full payload fails. Implement payload trimming or chunking.")

        # Concurrency issues
        concurrency_test = diagnosis_results['concurrency_test']
        for level, data in concurrency_test.items():
            if data['success_rate'] < 0.8:  # Less than 80% success
                recommendations.append(f"CONCURRENCY: {level} shows {data['success_rate']:.1%} success rate. Implement request throttling.")

        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Separate connect_timeout and read_timeout in HTTP client configuration",
                "Implement exponential backoff with jitter for retries",
                "Add circuit breaker pattern for API unavailability",
                "Implement graceful degradation with cached responses",
                "Add structured logging for timeout diagnosis"
            ])

        return recommendations


def main():
    """Run the complete diagnosis"""
    diagnoser = LongCatTimeoutDiagnoser()
    results = diagnoser.run_full_diagnosis()

    # Save results
    output_file = PROJECT_ROOT / "reports" / f"longcat_timeout_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nDiagnosis complete. Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)

    for key, value in results.items():
        if key != 'recommendations':
            if isinstance(value, dict) and 'status' in value:
                status = value['status']
                print(f"{key.upper()}: {status}")
            else:
                print(f"{key.upper()}: COMPLETED")

    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")

    return results


if __name__ == "__main__":
    main()