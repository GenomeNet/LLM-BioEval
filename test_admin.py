#!/usr/bin/env python3
"""
Admin Dashboard Test Suite
Simple integration tests for MicrobeLLM admin functionality
"""

import os
import sys
import time
import json
import shutil
import sqlite3
import requests
import subprocess
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5050"
TEST_DB = "test_microbellm_jobs.db"
ORIGINAL_DB = "microbellm_jobs.db"
TEST_MODEL = "test/model-v1"
TEST_SPECIES_FILE = "test_data/test_species.csv"

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class AdminTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.admin_process = None
        
    def print_result(self, test_name, success, message=""):
        """Print test result with color coding"""
        if success:
            print(f"{GREEN}✓{RESET} {test_name}")
            self.passed += 1
        elif success is None:  # Skipped
            print(f"{YELLOW}○{RESET} {test_name} - {message}")
            self.skipped += 1
        else:
            print(f"{RED}✗{RESET} {test_name} - {message}")
            self.failed += 1
    
    def setup_test_db(self):
        """Create test database with minimal schema"""
        try:
            # Copy original database for testing
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)
            shutil.copy2(ORIGINAL_DB, TEST_DB)
            return True
        except Exception as e:
            print(f"Failed to setup test database: {e}")
            return False
    
    def cleanup_test_db(self):
        """Remove test database"""
        try:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)
        except:
            pass
    
    def start_admin_server(self):
        """Start the admin server in test mode"""
        try:
            # Modify environment to use test database
            env = os.environ.copy()
            env['DATABASE_PATH'] = TEST_DB
            
            # Start admin server
            self.admin_process = subprocess.Popen(
                [sys.executable, '-m', 'microbellm.admin_app', '--port', '5050'],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            for i in range(10):
                try:
                    response = requests.get(f"{BASE_URL}/api/dashboard_data", timeout=1)
                    if response.status_code == 200:
                        return True
                except:
                    time.sleep(1)
            return False
        except Exception as e:
            print(f"Failed to start admin server: {e}")
            return False
    
    def stop_admin_server(self):
        """Stop the admin server"""
        if self.admin_process:
            self.admin_process.terminate()
            self.admin_process.wait(timeout=5)
    
    # Test Methods
    def test_dashboard_access(self):
        """Test dashboard endpoint accessibility"""
        try:
            response = requests.get(f"{BASE_URL}/api/dashboard_data")
            return response.status_code == 200
        except:
            return False
    
    def test_api_key_status(self):
        """Test API key status endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/api/api_key_status")
            data = response.json()
            return 'configured' in data or 'status' in data
        except:
            return False
    
    def test_model_operations(self):
        """Test adding and deleting models"""
        try:
            # First try to delete the test model if it exists
            requests.post(
                f"{BASE_URL}/api/delete_model",
                json={"model": TEST_MODEL}
            )
            
            # Add model
            response = requests.post(
                f"{BASE_URL}/api/add_model",
                json={"model": TEST_MODEL}
            )
            if response.status_code != 200:
                return False, "Failed to add model"
            
            # The model is successfully added if we got 200
            # (checking in dashboard_data is not reliable as it shows combinations, not managed_models)
            
            # Delete model
            response = requests.post(
                f"{BASE_URL}/api/delete_model",
                json={"model": TEST_MODEL}
            )
            if response.status_code != 200:
                return False, "Failed to delete model"
            
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def test_species_file_operations(self):
        """Test adding and deleting species files"""
        try:
            test_file = "test_species.csv"
            
            # Add species file
            response = requests.post(
                f"{BASE_URL}/api/add_species_file",
                json={"species_file": test_file}
            )
            if response.status_code != 200:
                return False, "Failed to add species file"
            
            # Delete species file
            response = requests.post(
                f"{BASE_URL}/api/delete_species_file",
                json={"species_file": test_file}
            )
            if response.status_code != 200:
                return False, "Failed to delete species file"
            
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def test_job_creation(self):
        """Test creating a job"""
        try:
            # First ensure test model and species file exist
            requests.post(
                f"{BASE_URL}/api/add_model",
                json={"model": "test/model"}
            )
            requests.post(
                f"{BASE_URL}/api/add_species_file",
                json={"species_file": "artificial.txt"}
            )
            
            # Create a simple job using correct field names
            response = requests.post(
                f"{BASE_URL}/api/create_job",
                json={
                    "species_file": "artificial.txt",
                    "model": "test/model",
                    "system_template": "templates/system/template1_phenotype.txt",
                    "user_template": "templates/user/template1_phenotype.txt"
                }
            )
            if response.status_code != 200:
                data = response.json() if response.text else {}
                return False, data.get('error', f"Status {response.status_code}")
            
            data = response.json()
            return 'job_id' in data or 'success' in data, ""
        except Exception as e:
            return False, str(e)
    
    def test_database_info(self):
        """Test database info endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/api/database_info")
            if response.status_code != 200:
                return False, "Failed to get database info"
            
            data = response.json()
            return 'tables' in data, ""
        except Exception as e:
            return False, str(e)
    
    def test_templates_endpoint(self):
        """Test templates listing"""
        try:
            response = requests.get(f"{BASE_URL}/api/templates")
            if response.status_code != 200:
                return False, "Failed to get templates"
            
            data = response.json()
            return 'templates' in data, ""
        except Exception as e:
            return False, str(e)
    
    def test_ground_truth_endpoints(self):
        """Test ground truth data endpoints"""
        try:
            # Test datasets endpoint
            response = requests.get(f"{BASE_URL}/api/ground_truth/datasets")
            if response.status_code != 200:
                return False, "Failed to get datasets"
            
            # Test distribution endpoint
            response = requests.get(f"{BASE_URL}/api/ground_truth/distribution")
            if response.status_code != 200:
                return False, "Failed to get distribution"
            
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def test_export_functionality(self):
        """Test CSV export endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/api/export_csv")
            # Should return 200 (with data) or 404 (no data), both are valid
            if response.status_code in [200, 404]:
                return True, ""
            else:
                return False, f"Unexpected status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def test_settings_endpoints(self):
        """Test settings and rate limit endpoints"""
        try:
            # Get current settings
            response = requests.get(f"{BASE_URL}/api/get_settings")
            if response.status_code != 200:
                return False, "Failed to get settings"
            
            # Try to set rate limit
            response = requests.post(
                f"{BASE_URL}/api/set_rate_limit",
                json={"requests_per_second": 10, "max_concurrent": 5}
            )
            if response.status_code != 200:
                return False, "Failed to set rate limit"
            
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def run_tests(self):
        """Run all tests"""
        print("\n" + "="*50)
        print("Testing MicrobeLLM Admin Dashboard")
        print("="*50 + "\n")
        
        # Check if admin is already running
        try:
            response = requests.get(f"{BASE_URL}/api/dashboard_data", timeout=1)
            if response.status_code == 200:
                print(f"{YELLOW}Admin dashboard already running on port 5050{RESET}")
                external_admin = True
            else:
                external_admin = False
        except:
            external_admin = False
        
        if not external_admin:
            # Setup test database
            print("Setting up test environment...")
            if not self.setup_test_db():
                print(f"{RED}Failed to setup test database{RESET}")
                return
            
            # Start admin server
            print("Starting admin server...")
            if not self.start_admin_server():
                print(f"{RED}Failed to start admin server{RESET}")
                self.cleanup_test_db()
                return
            
            print(f"{GREEN}Admin server started successfully{RESET}\n")
        else:
            print("Using existing admin instance\n")
        
        # Run tests
        print("Running tests:")
        print("-" * 30)
        
        # Basic connectivity
        result = self.test_dashboard_access()
        self.print_result("Dashboard accessibility", result)
        
        result = self.test_api_key_status()
        self.print_result("API key status endpoint", result)
        
        # Model operations
        result, msg = self.test_model_operations()
        self.print_result("Model add/delete operations", result, msg)
        
        # Species file operations
        result, msg = self.test_species_file_operations()
        self.print_result("Species file operations", result, msg)
        
        # Job creation
        result, msg = self.test_job_creation()
        self.print_result("Job creation", result, msg)
        
        # Database operations
        result, msg = self.test_database_info()
        self.print_result("Database info endpoint", result, msg)
        
        # Templates
        result, msg = self.test_templates_endpoint()
        self.print_result("Templates endpoint", result, msg)
        
        # Ground truth
        result, msg = self.test_ground_truth_endpoints()
        self.print_result("Ground truth endpoints", result, msg)
        
        # Export
        result, msg = self.test_export_functionality()
        self.print_result("Export functionality", result, msg)
        
        # Settings
        result, msg = self.test_settings_endpoints()
        self.print_result("Settings endpoints", result, msg)
        
        # Print summary
        print("-" * 30)
        print(f"\nTest Summary:")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        print(f"  {RED}Failed: {self.failed}{RESET}")
        if self.skipped > 0:
            print(f"  {YELLOW}Skipped: {self.skipped}{RESET}")
        
        # Overall result
        if self.failed == 0:
            print(f"\n{GREEN}All tests passed!{RESET}")
        else:
            print(f"\n{RED}Some tests failed.{RESET}")
        
        # Cleanup
        if not external_admin:
            print("\nCleaning up...")
            self.stop_admin_server()
            self.cleanup_test_db()
            print("Done!")
        
        return self.failed == 0

def main():
    """Main entry point"""
    tester = AdminTester()
    success = tester.run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()