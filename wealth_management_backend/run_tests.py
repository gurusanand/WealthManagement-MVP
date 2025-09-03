#!/usr/bin/env python3
"""
Comprehensive test runner for Wealth Management System
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print(f"\nSTDOUT:\n{result.stdout}")
    
    if result.stderr:
        print(f"\nSTDERR:\n{result.stderr}")
    
    return result.returncode == 0


def main():
    """Main test runner"""
    print("🧪 Wealth Management System - Comprehensive Test Suite")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Activate virtual environment
    venv_activate = "source venv/bin/activate"
    
    # Test categories
    test_categories = [
        {
            "name": "Unit Tests - Data Models",
            "command": f"{venv_activate} && python -m pytest tests/test_models.py -v --tb=short",
            "description": "Testing core data models and validation"
        },
        {
            "name": "Unit Tests - Portfolio Engine",
            "command": f"{venv_activate} && python -m pytest tests/test_portfolio_engine.py -v --tb=short -k 'not test_optimization_failure_handling'",
            "description": "Testing portfolio optimization, risk engine, and simulation"
        },
        {
            "name": "Unit Tests - Multi-Agent System",
            "command": f"{venv_activate} && python -m pytest tests/test_agents.py -v --tb=short -k 'not test_workflow_execution'",
            "description": "Testing agent communication and workflows"
        },
        {
            "name": "Integration Tests",
            "command": f"{venv_activate} && python -m pytest tests/test_integration.py -v --tb=short -k 'not test_concurrent_requests'",
            "description": "Testing system integration and API endpoints"
        },
        {
            "name": "Code Coverage Analysis",
            "command": f"{venv_activate} && python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --tb=short -q",
            "description": "Analyzing code coverage across all tests"
        }
    ]
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for category in test_categories:
        print(f"\n🔍 {category['name']}")
        success = run_command(category['command'], category['description'])
        results.append({
            'name': category['name'],
            'success': success
        })
        
        if not success:
            print(f"❌ {category['name']} - FAILED")
        else:
            print(f"✅ {category['name']} - PASSED")
    
    total_end_time = time.time()
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print()
    
    passed_count = sum(1 for result in results if result['success'])
    failed_count = len(results) - passed_count
    
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} - {result['name']}")
    
    print(f"\n📊 Results: {passed_count} passed, {failed_count} failed out of {len(results)} test categories")
    
    if failed_count == 0:
        print("\n🎉 ALL TESTS PASSED! The system is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {failed_count} test categories failed. Please review the output above.")
        return 1


def run_quick_tests():
    """Run a quick subset of tests for development"""
    print("🚀 Quick Test Suite - Development Mode")
    print("=" * 40)
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    venv_activate = "source venv/bin/activate"
    
    quick_tests = [
        {
            "name": "Basic Model Tests",
            "command": f"{venv_activate} && python -m pytest tests/test_models.py::TestClientModel::test_client_creation -v"
        },
        {
            "name": "Portfolio Optimizer Test",
            "command": f"{venv_activate} && python -m pytest tests/test_portfolio_engine.py::TestPortfolioOptimizer::test_sharpe_ratio_optimization -v"
        },
        {
            "name": "Agent Communication Test",
            "command": f"{venv_activate} && python -m pytest tests/test_agents.py::TestBaseAgent::test_agent_initialization -v"
        },
        {
            "name": "API Health Check",
            "command": f"{venv_activate} && python -m pytest tests/test_integration.py::TestAPIIntegration::test_health_endpoint -v"
        }
    ]
    
    results = []
    for test in quick_tests:
        print(f"\n🔍 {test['name']}")
        success = run_command(test['command'], test['name'])
        results.append({'name': test['name'], 'success': success})
    
    passed_count = sum(1 for result in results if result['success'])
    print(f"\n📊 Quick Test Results: {passed_count}/{len(results)} passed")
    
    return 0 if passed_count == len(results) else 1


def run_performance_tests():
    """Run performance and load tests"""
    print("⚡ Performance Test Suite")
    print("=" * 30)
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    venv_activate = "source venv/bin/activate"
    
    performance_tests = [
        {
            "name": "API Response Time",
            "command": f"{venv_activate} && python -m pytest tests/test_integration.py::TestPerformanceIntegration::test_api_response_time -v"
        },
        {
            "name": "Large Data Processing",
            "command": f"{venv_activate} && python -m pytest tests/test_integration.py::TestPerformanceIntegration::test_large_data_processing -v"
        },
        {
            "name": "Memory Usage",
            "command": f"{venv_activate} && python -m pytest tests/test_integration.py::TestPerformanceIntegration::test_memory_usage -v"
        }
    ]
    
    for test in performance_tests:
        print(f"\n🔍 {test['name']}")
        run_command(test['command'], test['name'])
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            sys.exit(run_quick_tests())
        elif sys.argv[1] == "performance":
            sys.exit(run_performance_tests())
        elif sys.argv[1] == "help":
            print("Usage:")
            print("  python run_tests.py          # Run full test suite")
            print("  python run_tests.py quick    # Run quick development tests")
            print("  python run_tests.py performance # Run performance tests")
            print("  python run_tests.py help     # Show this help")
            sys.exit(0)
    
    sys.exit(main())

