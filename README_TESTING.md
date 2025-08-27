# Testing Guide for MicrobeLLM

This document explains how to set up and run the comprehensive test suite for MicrobeLLM.

## 🧪 Test Structure

```
tests/
├── conftest.py          # Pytest fixtures and configuration
├── test_api.py          # API endpoint tests
├── test_utils.py        # Utility function tests
├── test_database.py     # Database operation tests
└── __pycache__/         # Python cache (auto-generated)
```

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **pip** package manager
3. **MicrobeLLM application** running or accessible

## 🚀 Installation

1. **Install testing dependencies:**
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Optional: Install coverage reporting:**
   ```bash
   pip install coverage
   ```

## 🏃 Running Tests

### **Basic Test Run:**
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api.py

# Run specific test function
pytest tests/test_api.py::TestDashboardAPI::test_dashboard_data
```

### **Coverage Reporting:**
```bash
# Generate coverage report
pytest --cov=microbellm --cov-report=html

# View coverage report
open htmlcov/index.html
```

### **Parallel Test Execution:**
```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n 4
```

## 🧪 Test Categories

### **API Tests (`test_api.py`)**
- Dashboard endpoints
- Model management
- Species file operations
- Job creation and management
- Template validation
- Database operations
- Export/import functionality
- Ground truth operations
- Health check endpoints

### **Utility Tests (`test_utils.py`)**
- Knowledge level normalization
- CSV structure validation
- JSON processing and cleaning
- API response extraction

### **Database Tests (`test_database.py`)**
- Database connection
- Table existence verification
- Data insertion operations
- Data retrieval operations
- Database integrity checks
- Concurrent access tests

## 🛠️ Configuration

### **Test Database:**
Tests use temporary databases that are automatically created and cleaned up:
- Located in `/tmp/` directory
- Automatically removed after tests complete
- Copy of original database for realistic testing

### **Environment Variables:**
```bash
# Set test-specific environment variables
export DATABASE_PATH=/tmp/test_microbellm.db
export FLASK_ENV=testing
```

## 📊 Writing New Tests

### **Basic Test Structure:**
```python
class TestFeatureName:
    """Test class for specific feature"""

    def test_feature_functionality(self, client):
        """Test specific functionality"""
        response = client.get('/api/some_endpoint')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'expected_key' in data
```

### **Using Fixtures:**
```python
def test_with_sample_data(client, sample_data):
    """Test using sample data fixture"""
    response = client.post('/api/create_job', json=sample_data)
    assert response.status_code == 200
```

### **Mocking External Services:**
```python
def test_external_api_call(self, client, mocker):
    """Test with mocked external API"""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'result': 'success'}

    with patch('requests.get', return_value=mock_response):
        response = client.get('/api/external_data')
        assert response.status_code == 200
```

## 🔍 Debugging Tests

### **Common Issues:**

1. **Database Connection Errors:**
   - Ensure MicrobeLLM database exists
   - Check file permissions
   - Verify SQLite installation

2. **Import Errors:**
   - Install all dependencies from `requirements-test.txt`
   - Check Python path configuration
   - Ensure MicrobeLLM package is installed or in path

3. **API Endpoint Failures:**
   - Verify Flask application is running
   - Check endpoint URLs
   - Review API response formats

### **Debugging Commands:**
```bash
# Run tests with detailed output
pytest -v -s

# Run single failing test with debug info
pytest tests/test_api.py::TestDashboardAPI::test_dashboard_data -v -s

# Run tests with coverage and fail on first error
pytest --cov=microbellm --tb=short -x
```

## 📈 Continuous Integration

### **GitHub Actions Example:**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=microbellm --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 🎯 Best Practices

### **Test Naming:**
- Use descriptive test names: `test_should_return_200_when_valid_data`
- Group related tests in classes
- Use fixtures for common setup/teardown

### **Test Coverage:**
- Aim for 80%+ code coverage
- Test both success and error scenarios
- Include edge cases and boundary conditions

### **Test Data:**
- Use realistic test data
- Create fixtures for reusable test data
- Clean up test data after tests

### **Performance:**
- Keep tests fast (< 100ms per test)
- Use parallel execution for large test suites
- Mock external services to avoid network delays

## 🔧 Maintenance

### **Regular Tasks:**
1. **Update dependencies:** `pip install -r requirements-test.txt --upgrade`
2. **Review coverage:** Check coverage reports and add missing tests
3. **Update fixtures:** Keep test data current with application changes
4. **Clean up:** Remove obsolete tests and update documentation

### **Troubleshooting:**
- **Tests failing after code changes:** Update test expectations
- **New dependencies:** Add to `requirements-test.txt`
- **Database schema changes:** Update database test fixtures

## 📞 Support

For questions about testing:
1. Check existing test files for examples
2. Review pytest documentation
3. Check application logs for errors
4. Create issues for bugs or improvements

---

**Happy Testing! 🧪**
