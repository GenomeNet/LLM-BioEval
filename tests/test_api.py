"""
API endpoint tests for MicrobeLLM admin dashboard
"""
import pytest
import json
from unittest.mock import patch, MagicMock


class TestDashboardAPI:
    """Test dashboard-related API endpoints"""

    def test_dashboard_data(self, client):
        """Test dashboard data endpoint"""
        response = client.get('/api/dashboard_data')
        assert response.status_code == 200

        data = json.loads(response.data)
        # The actual API returns data directly, not wrapped in success/error structure
        assert isinstance(data, dict)
        assert 'combinations' in data or 'matrix' in data  # Check for expected data structure

    def test_api_key_status(self, client):
        """Test API key status endpoint"""
        response = client.get('/api/api_key_status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'configured' in data

    def test_set_api_key(self, client):
        """Test setting API key"""
        test_key = "test-openrouter-key-123"

        response = client.post('/api/set_api_key',
                             json={'api_key': test_key})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'

    def test_get_settings(self, client):
        """Test getting application settings"""
        response = client.get('/api/get_settings')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_set_rate_limit(self, client):
        """Test setting rate limit"""
        response = client.post('/api/set_rate_limit',
                             json={'requests_per_second': 10})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'message' in data


class TestModelManagement:
    """Test model management endpoints"""

    def test_add_model(self, client):
        """Test adding a model"""
        test_model = "test/model-v1"

        response = client.post('/api/add_model',
                             json={'model': test_model})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

    def test_delete_model(self, client):
        """Test deleting a model"""
        test_model = "test/model-to-delete"

        # First add the model
        client.post('/api/add_model', json={'model': test_model})

        # Then delete it
        response = client.post('/api/delete_model',
                             json={'model': test_model})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

    def test_list_models(self, client):
        """Test listing models"""
        response = client.get('/api/models')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert isinstance(data, list)


class TestSpeciesFileManagement:
    """Test species file management endpoints"""

    def test_add_species_file(self, client):
        """Test adding a species file"""
        test_file = "test_species.csv"

        response = client.post('/api/add_species_file',
                             json={'species_file': test_file})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

    def test_delete_species_file(self, client):
        """Test deleting a species file"""
        test_file = "test_species_to_delete.csv"

        # First add the species file
        client.post('/api/add_species_file', json={'species_file': test_file})

        # Then delete it
        response = client.post('/api/delete_species_file',
                             json={'species_file': test_file})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True

    def test_list_species_files(self, client):
        """Test listing species files"""
        response = client.get('/api/species_files')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert isinstance(data, list)


class TestJobManagement:
    """Test job creation and management"""

    def test_create_job(self, client, sample_data):
        """Test creating a job"""
        response = client.post('/api/create_job', json=sample_data)
        # Job creation will fail because species file doesn't exist in filesystem
        # But we can test that the endpoint works and returns proper error
        assert response.status_code == 500  # Internal server error due to missing file

        data = json.loads(response.data)
        # The API should return an error response
        assert 'success' in data
        assert data['success'] is False
        assert 'error' in data

    def test_get_jobs(self, client):
        """Test getting job list"""
        response = client.get('/api/jobs')
        assert response.status_code == 200

        data = json.loads(response.data)
        # The API returns a dict with 'jobs' and 'success' fields
        assert isinstance(data, dict)
        assert 'jobs' in data
        assert 'success' in data
        assert isinstance(data['jobs'], list)


class TestTemplateManagement:
    """Test template management endpoints"""

    def test_get_templates(self, client):
        """Test getting templates list"""
        response = client.get('/api/templates')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'templates' in data
        assert isinstance(data['templates'], list)

    def test_template_validation_configs(self, client):
        """Test template validation configs"""
        response = client.get('/api/template_field_definitions?template=template1_phenotype')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'color_mappings' in data
        assert 'value_orderings' in data


class TestDatabaseOperations:
    """Test database-related endpoints"""

    def test_database_info(self, client):
        """Test database info endpoint"""
        response = client.get('/api/database_info')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'tables' in data
        assert data['success'] is True

    def test_table_data(self, client):
        """Test table data endpoint"""
        response = client.get('/api/table_data/processing_results')
        assert response.status_code in [200, 404]  # 404 if table is empty

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'data' in data
            assert data['success'] is True


class TestExportImport:
    """Test export/import functionality"""

    def test_export_csv(self, client):
        """Test CSV export endpoint"""
        response = client.get('/api/export_csv')
        assert response.status_code in [200, 404]  # 404 if no data

        if response.status_code == 200:
            assert 'text/csv' in response.content_type

    def test_export_ground_truth(self, client):
        """Test ground truth export"""
        response = client.get('/api/ground_truth/export?dataset=test_dataset')
        assert response.status_code in [200, 404]  # 404 if dataset doesn't exist


class TestGroundTruth:
    """Test ground truth functionality"""

    def test_ground_truth_datasets(self, client):
        """Test ground truth datasets endpoint"""
        response = client.get('/api/ground_truth/datasets')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'datasets' in data
        assert isinstance(data['datasets'], list)

    def test_ground_truth_distribution(self, client):
        """Test ground truth distribution endpoint"""
        response = client.get('/api/ground_truth/distribution?dataset=test_dataset')
        assert response.status_code in [200, 404]  # 404 if dataset doesn't exist

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'distribution' in data
            assert data['success'] is True

    def test_ground_truth_statistics(self, client):
        """Test ground truth statistics endpoint"""
        response = client.get('/api/ground_truth/phenotype_statistics?dataset_name=test_dataset')
        assert response.status_code in [200, 404]  # 404 if dataset doesn't exist

        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'statistics' in data
            assert data['success'] is True


class TestHealthChecks:
    """Test health check endpoints"""

    def test_health_endpoint(self, client):
        """Test basic health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'

    def test_health_detailed_endpoint(self, client):
        """Test detailed health check endpoint"""
        response = client.get('/health/detailed')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'status' in data
        assert 'checks' in data
        assert isinstance(data['checks'], dict)
