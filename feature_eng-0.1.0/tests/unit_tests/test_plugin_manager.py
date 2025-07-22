"""
Unit tests for Plugin Manager Component

This module contains comprehensive unit tests for the Plugin Manager component,
validating all behavioral contracts defined in the design_unit.md specification.
Tests focus on behavioral requirements BR-PM-001 and BR-PM-002.

Author: Feature Engineering System
Date: 2025-07-10
"""

import pytest
import time
import threading
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock


class MockPluginInitializationResult:
    """Mock plugin initialization result for testing purposes."""
    
    def __init__(self, success: bool = True, plugin_state: str = 'initialized', 
                 configuration_applied: bool = True):
        self.success = success
        self.plugin_state = plugin_state
        self.configuration_applied = configuration_applied


class MockPluginDisposalResult:
    """Mock plugin disposal result for testing purposes."""
    
    def __init__(self, success: bool = True, resources_released: bool = True, 
                 plugin_state: str = 'disposed'):
        self.success = success
        self.resources_released = resources_released
        self.plugin_state = plugin_state


class MockPluginExecutionResult:
    """Mock plugin execution result for testing purposes."""
    
    def __init__(self, success: bool = True, execution_time: float = None, 
                 output: Any = None, timeout_occurred: bool = False,
                 resources_cleaned: bool = False):
        self.success = success
        self.execution_time = execution_time or 0.1
        self.output = output
        self.timeout_occurred = timeout_occurred
        self.resources_cleaned = resources_cleaned


class MockPlugin:
    """Mock plugin for testing purposes."""
    
    def __init__(self, name: str = "test_plugin", execution_time: float = 0.1):
        self.name = name
        self.execution_time = execution_time
        self.state = 'uninitialized'
        self.resources = []
        self.config = None
        
    def initialize(self, config):
        """Initialize the plugin."""
        self.config = config
        self.state = 'initialized'
        # Simulate resource allocation
        self.resources.append("resource_1")
        self.resources.append("resource_2")
        
    def process(self, data, config=None):
        """Process data with the plugin."""
        if self.state != 'initialized':
            raise RuntimeError("Plugin not initialized")
            
        # Simulate processing time
        time.sleep(self.execution_time)
        
        # Return processed data
        return {"processed": True, "data": data, "plugin": self.name}
        
    def dispose(self):
        """Dispose of plugin resources."""
        self.resources.clear()
        self.state = 'disposed'
        
    def validate_config(self, config):
        """Validate configuration."""
        return True, "Configuration valid"


class MockExecutionContext:
    """Mock execution context for testing purposes."""
    
    def __init__(self, data: Any = None, config: Dict[str, Any] = None, 
                 timeout: float = None):
        self.data = data or {"sample": "data"}
        self.config = config or {"param": "value"}
        self.timeout = timeout


class TestPluginManagerBehavior:
    """
    Test class for validating Plugin Manager behavioral contracts.
    
    This class tests the behavioral requirements:
    - BR-PM-001: Plugin lifecycle management (initialization, disposal)
    - BR-PM-002: Plugin execution management (execution, timeout handling)
    """

    # Test fixtures and setup
    @pytest.fixture
    def plugin_manager(self):
        """
        Fixture providing a PluginManager instance.
        
        Returns:
            PluginManager: Configured plugin manager for testing
        """
        from app.plugin_manager import PluginManager
        return PluginManager()

    @pytest.fixture
    def plugin_configuration(self):
        """
        Fixture providing plugin configuration for testing.
        
        Returns:
            dict: Plugin configuration
        """
        return {
            "window_size": 20,
            "indicators": ["sma", "rsi"],
            "replicability_config": {
                "random_seed": 42,
                "deterministic_mode": True
            }
        }

    @pytest.fixture
    def mock_plugin_instance(self):
        """
        Fixture providing a mock plugin instance.
        
        Returns:
            MockPlugin: Mock plugin for testing
        """
        return MockPlugin("technical_indicator")

    @pytest.fixture
    def slow_plugin_instance(self):
        """
        Fixture providing a slow plugin instance for timeout testing.
        
        Returns:
            MockPlugin: Slow mock plugin for testing
        """
        return MockPlugin("slow_plugin", execution_time=2.0)

    @pytest.fixture
    def initialized_plugin_with_resources(self, mock_plugin_instance, plugin_configuration):
        """
        Fixture providing an initialized plugin with resources.
        
        Returns:
            MockPlugin: Initialized plugin with resources
        """
        plugin = mock_plugin_instance
        plugin.initialize(plugin_configuration)
        return plugin

    @pytest.fixture
    def execution_context(self):
        """
        Fixture providing execution context.
        
        Returns:
            MockExecutionContext: Execution context for testing
        """
        return MockExecutionContext()

    @pytest.fixture
    def execution_context_with_timeout(self):
        """
        Fixture providing execution context with timeout.
        
        Returns:
            MockExecutionContext: Execution context with timeout
        """
        return MockExecutionContext(timeout=1.0)

    # BR-PM-001: Plugin Lifecycle Management Tests
    def test_br_pm_001_manages_plugin_initialization_properly(self, plugin_manager, 
                                                             mock_plugin_instance, 
                                                             plugin_configuration):
        """
        Verify that plugin manager manages plugin initialization
        with proper configuration and state setup.
        
        Behavioral Contract: BR-PM-001
        Test ID: UT-PM-001
        """
        # When: Initializing plugin
        result = plugin_manager.initialize_plugin(mock_plugin_instance, plugin_configuration)
        
        # Then: Plugin is initialized properly
        assert result.success == True
        assert result.plugin_state == 'initialized'
        assert result.configuration_applied == True
        
        # Verify plugin state
        assert mock_plugin_instance.state == 'initialized'
        assert mock_plugin_instance.config == plugin_configuration
        assert len(mock_plugin_instance.resources) > 0
        
    def test_br_pm_001_manages_plugin_cleanup_and_disposal(self, plugin_manager, 
                                                          initialized_plugin_with_resources):
        """
        Verify that plugin manager properly cleans up and disposes
        of plugin resources and state.
        
        Behavioral Contract: BR-PM-001
        Test ID: UT-PM-002
        """
        # Given: Initialized plugin with resources
        plugin_instance = initialized_plugin_with_resources
        initial_resources_count = len(plugin_instance.resources)
        assert initial_resources_count > 0
        
        # When: Disposing plugin
        result = plugin_manager.dispose_plugin(plugin_instance)
        
        # Then: Plugin resources are cleaned up properly
        assert result.success == True
        assert result.resources_released == True
        assert result.plugin_state == 'disposed'
        
        # Verify cleanup
        assert plugin_instance.state == 'disposed'
        assert len(plugin_instance.resources) == 0
        
    def test_br_pm_001_handles_initialization_errors_gracefully(self, plugin_manager):
        """
        Verify that plugin manager handles initialization errors gracefully
        with proper error reporting and state management.
        
        Behavioral Contract: BR-PM-001
        Test ID: UT-PM-003
        """
        # Given: Plugin that fails initialization
        failing_plugin = Mock()
        failing_plugin.initialize.side_effect = RuntimeError("Initialization failed")
        
        config = {"param": "value"}
        
        # When: Attempting to initialize failing plugin
        result = plugin_manager.initialize_plugin(failing_plugin, config)
        
        # Then: Initialization error is handled gracefully
        assert result.success == False
        assert result.error_message is not None
        assert "initialization failed" in result.error_message.lower()
        
    def test_br_pm_001_validates_plugin_configuration_during_initialization(self, plugin_manager):
        """
        Verify that plugin manager validates plugin configuration
        during initialization process.
        
        Behavioral Contract: BR-PM-001
        Test ID: UT-PM-004
        """
        # Given: Plugin with configuration validation
        plugin_with_validation = MockPlugin()
        plugin_with_validation.validate_config = Mock(return_value=(False, "Invalid config"))
        
        invalid_config = {"invalid": "configuration"}
        
        # When: Initializing with invalid configuration
        result = plugin_manager.initialize_plugin(plugin_with_validation, invalid_config)
        
        # Then: Configuration validation is enforced
        assert result.success == False
        assert result.configuration_validated == False
        assert "invalid config" in result.error_message.lower()

    # BR-PM-002: Plugin Execution Management Tests
    def test_br_pm_002_manages_plugin_execution_safely(self, plugin_manager, 
                                                       initialized_plugin_with_resources, 
                                                       execution_context):
        """
        Verify that plugin manager manages plugin execution with
        proper error handling and timeout management.
        
        Behavioral Contract: BR-PM-002
        Test ID: UT-PM-005
        """
        # Given: Initialized plugin and execution context
        plugin_instance = initialized_plugin_with_resources
        
        # When: Executing plugin
        result = plugin_manager.execute_plugin(plugin_instance, execution_context)
        
        # Then: Plugin executes safely
        assert result.success == True
        assert result.execution_time is not None
        assert result.execution_time > 0
        assert result.output is not None
        assert result.output["processed"] == True
        assert result.output["plugin"] == plugin_instance.name
        
    def test_br_pm_002_handles_plugin_execution_timeouts(self, plugin_manager, 
                                                         slow_plugin_instance, 
                                                         execution_context_with_timeout,
                                                         plugin_configuration):
        """
        Verify that plugin manager handles plugin execution timeouts
        and resource cleanup appropriately.
        
        Behavioral Contract: BR-PM-002
        Test ID: UT-PM-006
        """
        # Given: Slow plugin and execution context with timeout
        plugin_instance = slow_plugin_instance
        plugin_instance.initialize(plugin_configuration)
        execution_context = execution_context_with_timeout
        
        # When: Executing plugin with timeout
        result = plugin_manager.execute_plugin(plugin_instance, execution_context)
        
        # Then: Timeout is handled appropriately
        assert result.success == False
        assert result.timeout_occurred == True
        assert result.resources_cleaned == True
        assert result.execution_time >= execution_context.timeout
        
    def test_br_pm_002_handles_plugin_execution_errors_safely(self, plugin_manager, 
                                                              mock_plugin_instance,
                                                              plugin_configuration):
        """
        Verify that plugin manager handles plugin execution errors
        safely with proper error reporting and cleanup.
        
        Behavioral Contract: BR-PM-002
        Test ID: UT-PM-007
        """
        # Given: Plugin that fails during execution
        failing_plugin = mock_plugin_instance
        failing_plugin.initialize(plugin_configuration)
        failing_plugin.process = Mock(side_effect=RuntimeError("Processing failed"))
        
        execution_context = MockExecutionContext()
        
        # When: Executing failing plugin
        result = plugin_manager.execute_plugin(failing_plugin, execution_context)
        
        # Then: Execution error is handled safely
        assert result.success == False
        assert result.error_message is not None
        assert "processing failed" in result.error_message.lower()
        assert result.resources_cleaned == True
        
    def test_br_pm_002_validates_plugin_state_before_execution(self, plugin_manager, 
                                                              mock_plugin_instance):
        """
        Verify that plugin manager validates plugin state before execution
        and prevents execution of uninitialized plugins.
        
        Behavioral Contract: BR-PM-002
        Test ID: UT-PM-008
        """
        # Given: Uninitialized plugin
        uninitialized_plugin = mock_plugin_instance
        assert uninitialized_plugin.state == 'uninitialized'
        
        execution_context = MockExecutionContext()
        
        # When: Attempting to execute uninitialized plugin
        result = plugin_manager.execute_plugin(uninitialized_plugin, execution_context)
        
        # Then: Execution is prevented
        assert result.success == False
        assert result.error_message is not None
        assert "not initialized" in result.error_message.lower()
        
    def test_br_pm_002_manages_concurrent_plugin_executions(self, plugin_manager,
                                                           plugin_configuration):
        """
        Verify that plugin manager can manage concurrent plugin executions
        safely without resource conflicts.
        
        Behavioral Contract: BR-PM-002
        Test ID: UT-PM-009
        """
        # Given: Multiple plugins for concurrent execution
        plugin1 = MockPlugin("plugin_1", execution_time=0.2)
        plugin2 = MockPlugin("plugin_2", execution_time=0.2)
        plugin3 = MockPlugin("plugin_3", execution_time=0.2)
        
        plugins = [plugin1, plugin2, plugin3]
        
        # Initialize all plugins
        for plugin in plugins:
            plugin.initialize(plugin_configuration)
            
        # When: Executing plugins concurrently
        import concurrent.futures
        execution_context = MockExecutionContext()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(plugin_manager.execute_plugin, plugin, execution_context)
                for plugin in plugins
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Then: All executions complete successfully
        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.output["processed"] for result in results)

    # Integration and Edge Case Tests
    def test_plugin_manager_handles_plugin_lifecycle_completely(self, plugin_manager,
                                                               plugin_configuration):
        """
        Verify that plugin manager handles complete plugin lifecycle
        from initialization through execution to disposal.
        
        Test ID: UT-PM-010
        """
        # Given: Fresh plugin instance
        plugin = MockPlugin("lifecycle_test")
        execution_context = MockExecutionContext()
        
        # When: Complete lifecycle management
        # 1. Initialize
        init_result = plugin_manager.initialize_plugin(plugin, plugin_configuration)
        assert init_result.success == True
        
        # 2. Execute
        exec_result = plugin_manager.execute_plugin(plugin, execution_context)
        assert exec_result.success == True
        
        # 3. Dispose
        dispose_result = plugin_manager.dispose_plugin(plugin)
        assert dispose_result.success == True
        
        # Then: Complete lifecycle is managed successfully
        assert plugin.state == 'disposed'
        assert len(plugin.resources) == 0
        
    def test_plugin_manager_handles_edge_cases_safely(self, plugin_manager):
        """
        Verify that plugin manager handles edge cases safely
        including None inputs, invalid plugins, etc.
        
        Test ID: UT-PM-011
        """
        edge_cases = [
            (None, {}),  # None plugin
            ("not_a_plugin", {}),  # Invalid plugin
            (MockPlugin(), None),  # None config
        ]
        
        # When: Handling edge cases
        for plugin, config in edge_cases:
            result = plugin_manager.initialize_plugin(plugin, config, handle_errors=True)
            
            # Then: Edge cases are handled safely without crashes
            assert result is not None
            assert hasattr(result, 'success')
            # Each case should fail gracefully but not crash


# Helper functions for creating test data
def create_plugin_configuration():
    """Create plugin configuration for testing."""
    return {
        "window_size": 20,
        "indicators": ["sma", "rsi"],
        "replicability_config": {
            "random_seed": 42,
            "deterministic_mode": True
        }
    }


def create_mock_plugin_instance():
    """Create mock plugin instance for testing."""
    return MockPlugin("test_plugin")


def create_initialized_plugin_with_resources():
    """Create initialized plugin with resources for testing."""
    plugin = MockPlugin("resource_plugin")
    plugin.initialize(create_plugin_configuration())
    return plugin


def create_safe_plugin_instance():
    """Create safe plugin instance for testing."""
    return MockPlugin("safe_plugin", execution_time=0.1)


def create_slow_plugin_instance():
    """Create slow plugin instance for testing."""
    return MockPlugin("slow_plugin", execution_time=2.0)


def create_execution_context():
    """Create execution context for testing."""
    return MockExecutionContext()


def create_execution_context_with_timeout(timeout=1.0):
    """Create execution context with timeout for testing."""
    return MockExecutionContext(timeout=timeout)


if __name__ == '__main__':
    # Run the tests when script is executed directly
    pytest.main([__file__, '-v', '--tb=short'])
