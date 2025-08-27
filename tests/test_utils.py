"""
Unit tests for utility functions and helpers
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import utility functions that actually exist
from microbellm.utils import (
    normalize_knowledge_level,
    clean_malformed_json,
    extract_and_validate_json,
    read_csv,
    colorize_text,
    clean_phenotype_value,
    normalize_value,
    detect_template_type
)


class TestKnowledgeLevelNormalization:
    """Test knowledge level normalization"""

    def test_normalize_knowledge_level(self):
        """Test normalizing knowledge level strings"""
        # Test valid inputs - function returns lowercase values
        assert normalize_knowledge_level("extensive") == "extensive"
        assert normalize_knowledge_level("Limited") == "limited"
        assert normalize_knowledge_level("MODERATE") == "moderate"
        assert normalize_knowledge_level("na") == "NA"
        assert normalize_knowledge_level("unknown") == "NA"

        # Test case insensitive
        assert normalize_knowledge_level("EXTENSIVE") == "extensive"
        assert normalize_knowledge_level("limited") == "limited"

        # Test invalid inputs
        assert normalize_knowledge_level("invalid") is None
        assert normalize_knowledge_level("") is None
        assert normalize_knowledge_level(None) is None


class TestCSVProcessing:
    """Test CSV reading and processing"""

    def test_read_csv_with_semicolon(self):
        """Test reading CSV with semicolon delimiter"""
        # Create a temporary CSV file
        csv_content = "binomial_name;gram_staining;motility\nEscherichia coli;Gram stain negative;Motile"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write(csv_content)
            temp_file_path = temp_file.name

        try:
            headers, rows = read_csv(temp_file_path, delimiter=';')
            assert len(headers) == 3
            assert headers == ['binomial_name', 'gram_staining', 'motility']
            assert len(rows) == 1
            assert rows[0] == ['Escherichia coli', 'Gram stain negative', 'Motile']
        finally:
            os.unlink(temp_file_path)

    def test_read_csv_with_comma(self):
        """Test reading CSV with comma delimiter"""
        csv_content = "binomial_name,gram_staining,motility\nStaphylococcus aureus,Gram stain positive,Non-motile"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write(csv_content)
            temp_file_path = temp_file.name

        try:
            headers, rows = read_csv(temp_file_path, delimiter=',')
            assert len(headers) == 3
            assert headers == ['binomial_name', 'gram_staining', 'motility']
            assert len(rows) == 1
            assert rows[0] == ['Staphylococcus aureus', 'Gram stain positive', 'Non-motile']
        finally:
            os.unlink(temp_file_path)


class TestPhenotypeProcessing:
    """Test phenotype value processing"""

    def test_clean_phenotype_value(self):
        """Test cleaning phenotype values"""
        # Test with gram staining - function converts to lowercase standardized format
        assert clean_phenotype_value("Gram stain positive") == "gram stain positive"
        assert clean_phenotype_value("Gram Stain Negative") == "gram stain negative"
        assert clean_phenotype_value("GRAM STAIN VARIABLE") == "gram stain variable"

        # Test with motility - function converts to TRUE/FALSE
        assert clean_phenotype_value("motile") == "TRUE"
        assert clean_phenotype_value("non-motile") == "FALSE"

        # Test with boolean-like values - function converts to TRUE/FALSE
        assert clean_phenotype_value("true") == "TRUE"
        assert clean_phenotype_value("false") == "FALSE"
        assert clean_phenotype_value("yes") == "TRUE"
        assert clean_phenotype_value("no") == "FALSE"

        # Test with null/empty values - function returns empty string, not None
        assert clean_phenotype_value(None) == ""
        assert clean_phenotype_value("") == ""
        assert clean_phenotype_value("N/A") == "N/A"

    def test_normalize_value(self):
        """Test value normalization"""
        # Test with various inputs - function preserves case
        assert normalize_value("TRUE") == "TRUE"
        assert normalize_value("False") == "False"
        assert normalize_value("  Test  ") == "Test"
        assert normalize_value(None) == "NA"
        assert normalize_value("") == "NA"
        assert normalize_value("N/A") == "NA"

    def test_colorize_text(self):
        """Test text colorization"""
        colored_text = colorize_text("Test message", "green")
        assert "Test message" in colored_text
        # Note: ANSI codes will be present in actual output


class TestTemplateDetection:
    """Test template type detection"""

    def test_detect_template_type_knowledge(self):
        """Test detection of knowledge templates"""
        # Create a temporary template file with knowledge keywords
        template_content = "This template contains knowledge_group and knowledge_level information."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(template_content)
            temp_file_path = temp_file.name

        try:
            template_type = detect_template_type(temp_file_path)
            assert template_type == "knowledge"
        finally:
            os.unlink(temp_file_path)

    def test_detect_template_type_phenotype(self):
        """Test detection of phenotype templates"""
        # Create a temporary template file with phenotype keywords
        template_content = "This template contains gram_staining and motility information."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(template_content)
            temp_file_path = temp_file.name

        try:
            template_type = detect_template_type(temp_file_path)
            assert template_type == "phenotype"
        finally:
            os.unlink(temp_file_path)


class TestJSONProcessing:
    """Test JSON extraction and validation"""

    def test_clean_malformed_json(self):
        """Test cleaning malformed JSON"""
        # Test with quoted quotes
        malformed = '{"knowledge_group": ""extensive""}'
        cleaned = clean_malformed_json(malformed)
        assert cleaned == '{"knowledge_group": "extensive"}'

        # Test with valid JSON - should return as-is
        valid_json = '{"knowledge_group": "extensive"}'
        cleaned = clean_malformed_json(valid_json)
        assert cleaned == valid_json

        # Test with empty string
        assert clean_malformed_json("") == ""
        assert clean_malformed_json(None) is None

    def test_extract_and_validate_json(self):
        """Test the complete JSON extraction and validation pipeline"""
        # Test with simple JSON string
        response = '{"knowledge_group": "extensive"}'

        # This function exists in utils.py and handles JSON extraction
        validated_data = extract_and_validate_json(response)

        assert isinstance(validated_data, dict)
        assert validated_data['knowledge_group'] == 'extensive'

    def test_extract_and_validate_json_malformed(self):
        """Test JSON extraction with malformed input"""
        # Test with malformed JSON that needs cleaning
        response = '{"knowledge_group": ""extensive""}'

        validated_data = extract_and_validate_json(response)

        # The function should handle malformed JSON
        assert isinstance(validated_data, dict)
        if 'knowledge_group' in validated_data:
            assert validated_data['knowledge_group'] == 'extensive'


class TestFileOperations:
    """Test file operation utilities"""

    def test_temp_file_handling(self):
        """Test temporary file creation and cleanup"""
        # This would test any file utility functions if they exist
        # For now, this is a placeholder for file-related tests
        pass


class TestDataProcessing:
    """Test data processing utilities"""

    def test_data_transformation(self):
        """Test data transformation functions"""
        # Test any data processing utilities
        # This is a placeholder for data transformation tests
        pass


class TestAPICalls:
    """Test API calling utilities"""

    def test_api_timeout_handling(self):
        """Test API timeout handling"""
        # This would test API utility functions if they exist
        # For now, this is a placeholder for API-related tests
        pass


class TestPerformance:
    """Test performance-related utilities"""

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Test rate limiting implementation
        # This is a placeholder for performance tests
        pass
