"""
Unit tests for Plugin Registry Component

This module contains comprehensive unit tests for the Plugin Registry component,
validating all behavioral contracts defined in the design_unit.md specification.
Tests focus on behavioral requirements BR-PR-001 and BR-PR-002.

Author: Feature Engineering System
Date: 2025-07-10
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock


class MockPluginRegistrationResult:
    """Mock plugin registration result for testing purposes."""
    
    def __init__(self, success: bool = True, plugin_id: str = None, 
                 metadata_validated: bool = True, missing_metadata: List[str] = None):
        self.success = success
        self.plugin_id = plugin_id
        self.metadata_validated = metadata_validated
        self.missing_metadata = missing_metadata or []


class MockPluginQueryResult:
    """Mock plugin query result for testing purposes."""
    
    def __init__(self, success: bool = True, matching_plugins: List[Any] = None):
        self.success = success
        self.matching_plugins = matching_plugins or []


class TestPluginRegistryBehavior:
    """
    Test class for validating Plugin Registry behavioral contracts.
    
    This class tests the behavioral requirements:
    - BR-PR-001: Plugin registration with complete metadata validation
    - BR-PR-002: Plugin query and discovery by capabilities
    """

    # Test fixtures and setup
    @pytest.fixture
    def plugin_registry(self):
        """
        Fixture providing a PluginRegistry instance.
        
        Returns:
            PluginRegistry: Configured plugin registry for testing
        """
        from app.plugin_registry import PluginRegistry
        return PluginRegistry()

    @pytest.fixture
    def complete_plugin_metadata(self):
        """
        Fixture providing complete plugin metadata for testing.
        
        Returns:
            dict: Complete plugin metadata
        """
        return {
            "name": "technical_indicator",
            "version": "1.0.0",
            "description": "Technical indicator calculations for financial analysis",
            "author": "Feature Engineering Team",
            "license": "MIT",
            "interface_version": "1.0",
            "capabilities": ["technical_indicators", "real_time", "batch_processing"],
            "requirements": ["pandas>=1.3.0", "numpy>=1.20.0", "talib>=0.4.0"],
            "parameters": {
                "window_size": {
                    "type": "int",
                    "default": 20,
                    "min": 5,
                    "max": 100,
                    "description": "Moving window size for calculations"
                },
                "indicators": {
                    "type": "list",
                    "default": ["sma"],
                    "options": ["sma", "ema", "rsi", "macd", "bollinger"],
                    "description": "List of technical indicators to compute"
                }
            },
            "output_schema": {
                "type": "dataframe",
                "columns": {
                    "sma": {"type": "float", "description": "Simple Moving Average"},
                    "rsi": {"type": "float", "description": "Relative Strength Index"}
                }
            },
            "deterministic": True,
            "reproducible": True,
            "entry_point": "main.py",
            "test_suite": "test_technical_indicator.py",
            "documentation": "docs/technical_indicator.md"
        }

    @pytest.fixture
    def incomplete_plugin_metadata(self):
        """
        Fixture providing incomplete plugin metadata for testing.
        
        Returns:
            dict: Incomplete plugin metadata
        """
        return {
            "name": "incomplete_plugin",
            "version": "1.0.0"
            # Missing required fields: description, author, interface_version, etc.
        }

    @pytest.fixture
    def decomposition_plugin_metadata(self):
        """
        Fixture providing decomposition plugin metadata for testing.
        
        Returns:
            dict: Decomposition plugin metadata
        """
        return {
            "name": "decomposition_post_processor",
            "version": "1.0.0",
            "description": "Feature decomposition post processor",
            "author": "Feature Engineering Team",
            "license": "MIT",
            "interface_version": "1.0",
            "capabilities": ["post_processing", "decomposition", "time_series"],
            "requirements": ["pandas>=1.3.0", "numpy>=1.20.0", "scipy>=1.7.0", "pywt>=1.1.0"],
            "parameters": {
                "decomp_features": {
                    "type": "list",
                    "default": [],
                    "description": "List of features to decompose"
                },
                "decomp_method": {
                    "type": "string",
                    "default": "stl",
                    "options": ["stl", "wavelet", "mtm"],
                    "description": "Decomposition method to use"
                }
            },
            "output_schema": {
                "type": "dataframe",
                "replaces_features": True
            },
            "deterministic": True,
            "reproducible": True,
            "entry_point": "decomposition.py"
        }

    # BR-PR-001: Plugin Registration Tests
    def test_br_pr_001_registers_plugins_with_complete_metadata(self, plugin_registry, complete_plugin_metadata):
        """
        Verify that plugin registry registers plugins with complete
        metadata including capabilities and requirements.
        
        Behavioral Contract: BR-PR-001
        Test ID: UT-PR-001
        """
        # When: Registering plugin with complete metadata
        result = plugin_registry.register_plugin(complete_plugin_metadata)
        
        # Then: Plugin is registered with complete metadata
        assert result.success == True
        assert result.plugin_id is not None
        assert result.metadata_validated == True
        assert len(result.plugin_id) > 0
        
        # Verify plugin can be retrieved
        retrieved_plugin = plugin_registry.get_plugin(result.plugin_id)
        assert retrieved_plugin is not None
        assert retrieved_plugin.name == complete_plugin_metadata['name']
        assert retrieved_plugin.capabilities == complete_plugin_metadata['capabilities']
        
    def test_br_pr_001_validates_plugin_metadata_completeness(self, plugin_registry, incomplete_plugin_metadata):
        """
        Verify that plugin registry validates plugin metadata
        completeness and reports missing information.
        
        Behavioral Contract: BR-PR-001
        Test ID: UT-PR-002
        """
        # When: Attempting to register plugin with incomplete metadata
        result = plugin_registry.register_plugin(incomplete_plugin_metadata)
        
        # Then: Metadata validation errors are reported
        assert result.success == False
        assert result.missing_metadata is not None
        assert len(result.missing_metadata) > 0
        
        # Verify specific missing fields are identified
        expected_missing = ['description', 'author', 'interface_version', 'capabilities']
        for field in expected_missing:
            assert field in result.missing_metadata
            
    def test_br_pr_001_validates_plugin_replicability_requirements(self, plugin_registry):
        """
        Verify that plugin registry validates replicability requirements
        for plugins to ensure deterministic execution.
        
        Behavioral Contract: BR-PR-001
        Test ID: UT-PR-003
        """
        # Given: Plugin metadata without replicability guarantees
        non_replicable_metadata = {
            "name": "non_replicable_plugin",
            "version": "1.0.0",
            "description": "Non-replicable plugin",
            "author": "Test Author",
            "interface_version": "1.0",
            "capabilities": ["processing"],
            "deterministic": False,  # Not deterministic
            "reproducible": False    # Not reproducible
        }
        
        # When: Registering non-replicable plugin with strict validation
        result = plugin_registry.register_plugin(non_replicable_metadata, strict_replicability=True)
        
        # Then: Registration fails due to replicability requirements
        assert result.success == False
        assert 'replicability' in str(result.validation_errors).lower()
        
    def test_br_pr_001_validates_interface_version_compatibility(self, plugin_registry, complete_plugin_metadata):
        """
        Verify that plugin registry validates interface version compatibility
        to ensure plugins work with current system version.
        
        Behavioral Contract: BR-PR-001
        Test ID: UT-PR-004
        """
        # Given: Plugin with incompatible interface version
        incompatible_metadata = complete_plugin_metadata.copy()
        incompatible_metadata['interface_version'] = "2.0"  # Future version
        
        # When: Registering plugin with incompatible interface
        result = plugin_registry.register_plugin(incompatible_metadata, 
                                                required_interface_version="1.0")
        
        # Then: Registration fails due to interface incompatibility
        assert result.success == False
        assert 'interface_version' in str(result.validation_errors).lower()
        
    def test_br_pr_001_prevents_duplicate_plugin_registration(self, plugin_registry, complete_plugin_metadata):
        """
        Verify that plugin registry prevents duplicate plugin registration
        and handles conflicts appropriately.
        
        Behavioral Contract: BR-PR-001
        Test ID: UT-PR-005
        """
        # Given: Plugin already registered
        first_result = plugin_registry.register_plugin(complete_plugin_metadata)
        assert first_result.success == True
        
        # When: Attempting to register the same plugin again
        second_result = plugin_registry.register_plugin(complete_plugin_metadata)
        
        # Then: Duplicate registration is handled appropriately
        assert second_result.success == False
        assert 'duplicate' in str(second_result.validation_errors).lower() or \
               'already registered' in str(second_result.validation_errors).lower()

    # BR-PR-002: Plugin Query and Discovery Tests
    def test_br_pr_002_queries_plugins_by_capabilities(self, plugin_registry, complete_plugin_metadata, decomposition_plugin_metadata):
        """
        Verify that plugin registry supports querying plugins
        by their capabilities and requirements.
        
        Behavioral Contract: BR-PR-002
        Test ID: UT-PR-006
        """
        # Given: Registry with multiple registered plugins
        tech_result = plugin_registry.register_plugin(complete_plugin_metadata)
        decomp_result = plugin_registry.register_plugin(decomposition_plugin_metadata)
        assert tech_result.success == True
        assert decomp_result.success == True
        
        # When: Querying by capabilities
        result = plugin_registry.query_by_capabilities(['technical_indicators'])
        
        # Then: Matching plugins are returned
        assert result.success == True
        assert len(result.matching_plugins) >= 1
        assert any('technical_indicators' in p.capabilities for p in result.matching_plugins)
        
        # Verify specific plugin is found
        tech_plugin = next((p for p in result.matching_plugins if p.name == 'technical_indicator'), None)
        assert tech_plugin is not None
        assert 'technical_indicators' in tech_plugin.capabilities
        
    def test_br_pr_002_queries_plugins_by_multiple_capabilities(self, plugin_registry, complete_plugin_metadata, decomposition_plugin_metadata):
        """
        Verify that plugin registry supports querying plugins
        by multiple capabilities with AND/OR logic.
        
        Behavioral Contract: BR-PR-002
        Test ID: UT-PR-007
        """
        # Given: Registry with registered plugins
        plugin_registry.register_plugin(complete_plugin_metadata)
        plugin_registry.register_plugin(decomposition_plugin_metadata)
        
        # When: Querying by multiple capabilities (AND logic)
        result = plugin_registry.query_by_capabilities(['post_processing', 'decomposition'], logic='AND')
        
        # Then: Only plugins with ALL capabilities are returned
        assert result.success == True
        assert len(result.matching_plugins) >= 1
        
        for plugin in result.matching_plugins:
            assert 'post_processing' in plugin.capabilities
            assert 'decomposition' in plugin.capabilities
            
    def test_br_pr_002_queries_plugins_by_requirements(self, plugin_registry, complete_plugin_metadata):
        """
        Verify that plugin registry supports querying plugins
        by their dependency requirements and compatibility.
        
        Behavioral Contract: BR-PR-002
        Test ID: UT-PR-008
        """
        # Given: Registry with plugins having specific requirements
        plugin_registry.register_plugin(complete_plugin_metadata)
        
        # When: Querying by requirements
        result = plugin_registry.query_by_requirements(['pandas', 'numpy'])
        
        # Then: Plugins with matching requirements are returned
        assert result.success == True
        assert len(result.matching_plugins) >= 1
        
        # Verify requirements are compatible
        for plugin in result.matching_plugins:
            requirements_str = ' '.join(plugin.requirements)
            assert 'pandas' in requirements_str
            assert 'numpy' in requirements_str
            
    def test_br_pr_002_queries_plugins_by_interface_version(self, plugin_registry, complete_plugin_metadata, decomposition_plugin_metadata):
        """
        Verify that plugin registry supports querying plugins
        by interface version for compatibility filtering.
        
        Behavioral Contract: BR-PR-002
        Test ID: UT-PR-009
        """
        # Given: Registry with plugins of specific interface version
        plugin_registry.register_plugin(complete_plugin_metadata)
        plugin_registry.register_plugin(decomposition_plugin_metadata)
        
        # When: Querying by interface version
        result = plugin_registry.query_by_interface_version("1.0")
        
        # Then: Only compatible plugins are returned
        assert result.success == True
        assert len(result.matching_plugins) >= 2
        
        for plugin in result.matching_plugins:
            assert plugin.interface_version == "1.0"
            
    def test_br_pr_002_lists_all_registered_plugins(self, plugin_registry, complete_plugin_metadata, decomposition_plugin_metadata):
        """
        Verify that plugin registry can list all registered plugins
        with their metadata for discovery purposes.
        
        Behavioral Contract: BR-PR-002
        Test ID: UT-PR-010
        """
        # Given: Registry with multiple registered plugins
        tech_result = plugin_registry.register_plugin(complete_plugin_metadata)
        decomp_result = plugin_registry.register_plugin(decomposition_plugin_metadata)
        
        # When: Listing all plugins
        result = plugin_registry.list_all_plugins()
        
        # Then: All registered plugins are returned
        assert result.success == True
        assert len(result.plugins) >= 2
        
        # Verify specific plugins are included
        plugin_names = [p.name for p in result.plugins]
        assert 'technical_indicator' in plugin_names
        assert 'decomposition_post_processor' in plugin_names
        
    def test_br_pr_002_supports_plugin_metadata_search(self, plugin_registry, complete_plugin_metadata):
        """
        Verify that plugin registry supports searching plugins
        by metadata fields like name, description, author, etc.
        
        Behavioral Contract: BR-PR-002
        Test ID: UT-PR-011
        """
        # Given: Registry with registered plugin
        plugin_registry.register_plugin(complete_plugin_metadata)
        
        # When: Searching by metadata fields
        name_result = plugin_registry.search_by_metadata(name="technical_indicator")
        desc_result = plugin_registry.search_by_metadata(description="technical indicator")
        author_result = plugin_registry.search_by_metadata(author="Feature Engineering Team")
        
        # Then: Plugins matching search criteria are returned
        assert name_result.success == True
        assert len(name_result.matching_plugins) >= 1
        
        assert desc_result.success == True
        assert len(desc_result.matching_plugins) >= 1
        
        assert author_result.success == True
        assert len(author_result.matching_plugins) >= 1

    # Integration and Edge Case Tests
    def test_plugin_registry_handles_large_plugin_collections(self, plugin_registry):
        """
        Verify that plugin registry handles large collections of plugins
        efficiently without performance degradation.
        
        Test ID: UT-PR-012
        """
        # Given: Large number of plugins to register
        plugin_count = 100
        registered_plugins = []
        
        import time
        start_time = time.time()
        
        # When: Registering many plugins
        for i in range(plugin_count):
            plugin_metadata = {
                "name": f"test_plugin_{i}",
                "version": "1.0.0",
                "description": f"Test plugin {i}",
                "author": "Test Author",
                "interface_version": "1.0",
                "capabilities": [f"capability_{i % 5}"],  # 5 different capabilities
                "deterministic": True,
                "reproducible": True
            }
            result = plugin_registry.register_plugin(plugin_metadata)
            if result.success:
                registered_plugins.append(result.plugin_id)
                
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Then: Performance requirements are met
        assert len(registered_plugins) == plugin_count
        assert registration_time < 10.0  # Should register quickly
        
        # Verify query performance
        start_time = time.time()
        query_result = plugin_registry.query_by_capabilities(['capability_0'])
        end_time = time.time()
        query_time = end_time - start_time
        
        assert query_result.success == True
        assert query_time < 1.0  # Should query quickly
        
    def test_plugin_registry_handles_edge_cases_safely(self, plugin_registry):
        """
        Verify that plugin registry handles edge cases safely
        including None inputs, empty metadata, malformed data.
        
        Test ID: UT-PR-013
        """
        edge_cases = [
            None,  # None input
            {},    # Empty metadata
            {"name": None},  # None name
            {"name": ""},    # Empty name
            {"name": "test", "capabilities": None},  # None capabilities
        ]
        
        # When: Handling edge cases
        for edge_case in edge_cases:
            result = plugin_registry.register_plugin(edge_case, handle_errors=True)
            
            # Then: Edge cases are handled safely without crashes
            assert result is not None
            assert hasattr(result, 'success')
            # Each case should fail gracefully but not crash


# Helper functions for creating test data
def create_complete_plugin_metadata():
    """Create complete plugin metadata for testing."""
    return {
        "name": "test_plugin",
        "version": "1.0.0",
        "description": "Test plugin",
        "author": "Test Author",
        "interface_version": "1.0",
        "capabilities": ["testing"],
        "deterministic": True,
        "reproducible": True
    }


def create_incomplete_plugin_metadata():
    """Create incomplete plugin metadata for testing."""
    return {
        "name": "incomplete_plugin",
        "version": "1.0.0"
    }


if __name__ == '__main__':
    # Run the tests when script is executed directly
    pytest.main([__file__, '-v', '--tb=short'])
