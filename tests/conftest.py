"""
Pytest configuration and fixtures for MicrobeLLM testing
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from microbellm.admin_app import app, db_path

@pytest.fixture(scope="session", autouse=True)
def initialize_processing_manager():
    """Initialize the processing manager for tests"""
    try:
        from microbellm.admin_app import processing_manager, ProcessingManager
        if processing_manager is None:
            # Initialize the global processing manager for tests
            import microbellm.admin_app as app_module
            app_module.processing_manager = ProcessingManager()
    except (ImportError, Exception) as e:
        print(f"Warning: Could not initialize processing manager for tests: {e}")
        pass


@pytest.fixture(scope="session", autouse=True)
def initialize_database_tables():
    """Initialize database tables needed for tests"""
    try:
        from microbellm.admin_app import db_path
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create managed_models table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS managed_models (
                model TEXT PRIMARY KEY
            )
        ''')

        # Create managed_species_files table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS managed_species_files (
                species_file TEXT PRIMARY KEY
            )
        ''')

        # Add some test data
        cursor.execute("INSERT OR IGNORE INTO managed_models (model) VALUES (?)", ("test-model-v1",))
        cursor.execute("INSERT OR IGNORE INTO managed_species_files (species_file) VALUES (?)", ("test_species.csv",))

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Warning: Could not initialize database tables for tests: {e}")
        pass


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database"""
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()

    # Copy original database if it exists
    original_db = Path(db_path)
    if original_db.exists():
        shutil.copy2(original_db, temp_db.name)

    yield temp_db.name

    # Cleanup
    if os.path.exists(temp_db.name):
        os.unlink(temp_db.name)


@pytest.fixture
def client(test_db):
    """Flask test client with test database"""
    app.config['TESTING'] = True
    app.config['DATABASE_PATH'] = test_db

    with app.test_client() as client:
        yield client


@pytest.fixture
def authenticated_client(client):
    """Client with API key authentication"""
    # Set a test API key
    with client.application.test_request_context():
        from microbellm.admin_app import set_api_key
        set_api_key("test-api-key-12345")

    # Add API key to client headers
    client.environ_base['HTTP_AUTHORIZATION'] = 'Bearer test-api-key-12345'
    return client


@pytest.fixture
def sample_data():
    """Sample test data for testing"""
    return {
        'species_file': 'test_species.csv',
        'model': 'test-model-v1',
        'system_template': 'templates/system/template1_phenotype.txt',
        'user_template': 'templates/user/template1_phenotype.txt'
    }


@pytest.fixture
def ground_truth_data():
    """Sample ground truth data for testing"""
    return {
        'dataset_name': 'test_dataset',
        'template_name': 'template1_phenotype',
        'species_data': [
            {
                'binomial_name': 'Escherichia coli',
                'gram_staining': 'Gram stain negative',
                'motility': 'Motile',
                'aerophilicity': 'Facultative anaerobic'
            },
            {
                'binomial_name': 'Staphylococcus aureus',
                'gram_staining': 'Gram stain positive',
                'motility': 'Non-motile',
                'aerophilicity': 'Aerobic'
            }
        ]
    }
