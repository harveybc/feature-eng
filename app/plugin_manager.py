"""
Plugin Manager Component

This module provides comprehensive plugin lifecycle and execution management
functionality. Handles plugin initialization, execution with timeout support,
and proper resource cleanup with emphasis on replicability and deterministic
execution.

Author: Feature Engineering System
Date: 2025-07-10
"""

import time
import threading
import traceback
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


@dataclass
class PluginInitializationResult:
    """Result of plugin initialization operation."""
    success: bool = False
    plugin_state: str = 'uninitialized'
    configuration_applied: bool = False
    configuration_validated: bool = False
    error_message: str = None


@dataclass
class PluginDisposalResult:
    """Result of plugin disposal operation."""
    success: bool = False
    resources_released: bool = False
    plugin_state: str = 'unknown'
    error_message: str = None


@dataclass
class PluginExecutionResult:
    """Result of plugin execution operation."""
    success: bool = False
    execution_time: float = 0.0
    output: Any = None
    timeout_occurred: bool = False
    resources_cleaned: bool = False
    error_message: str = None


class PluginManager:
    """
    Plugin manager for lifecycle and execution management.
    
    This class provides comprehensive plugin management capabilities including
    initialization, execution with timeout support, and proper resource cleanup
    with emphasis on replicability and deterministic execution.
    """

    def __init__(self):
        """Initialize the plugin manager."""
        self.active_plugins = {}
        self.execution_lock = threading.Lock()
        self.default_timeout = 30.0
        
    def initialize_plugin(self, plugin_instance: Any, config: Dict[str, Any],
                         handle_errors: bool = False) -> PluginInitializationResult:
        """
        Initialize a plugin with proper configuration and state setup.
        
        Args:
            plugin_instance: Plugin instance to initialize
            config: Configuration dictionary for the plugin
            handle_errors: Whether to handle errors gracefully
            
        Returns:
            PluginInitializationResult: Initialization result with status details
        """
        result = PluginInitializationResult()
        
        # Handle edge cases safely
        if handle_errors:
            if plugin_instance is None:
                result.error_message = "Plugin instance is None"
                return result
            if config is None:
                result.error_message = "Configuration is None"
                return result
            if not hasattr(plugin_instance, 'initialize'):
                result.error_message = "Plugin does not have initialize method"
                return result
                
        # Validate plugin instance
        if not plugin_instance:
            result.error_message = "Plugin instance is required"
            return result
            
        if not config:
            result.error_message = "Configuration is required"
            return result
            
        # Check if plugin has required methods
        if not hasattr(plugin_instance, 'initialize'):
            result.error_message = "Plugin missing required 'initialize' method"
            return result
            
        try:
            # Validate configuration if plugin supports it
            if hasattr(plugin_instance, 'validate_config'):
                is_valid, validation_message = plugin_instance.validate_config(config)
                result.configuration_validated = is_valid
                
                if not is_valid:
                    result.error_message = f"Configuration validation failed: {validation_message}"
                    return result
            else:
                result.configuration_validated = True
                
            # Initialize the plugin
            plugin_instance.initialize(config)
            
            # Check plugin state after initialization
            if hasattr(plugin_instance, 'state'):
                result.plugin_state = plugin_instance.state
            else:
                result.plugin_state = 'initialized'
                
            # Success
            result.success = True
            result.configuration_applied = True
            
            # Store plugin reference
            plugin_id = id(plugin_instance)
            self.active_plugins[plugin_id] = plugin_instance
            
        except Exception as e:
            result.error_message = f"Plugin initialization failed: {str(e)}"
            result.plugin_state = 'error'
            
        return result
        
    def dispose_plugin(self, plugin_instance: Any) -> PluginDisposalResult:
        """
        Properly clean up and dispose of plugin resources and state.
        
        Args:
            plugin_instance: Plugin instance to dispose
            
        Returns:
            PluginDisposalResult: Disposal result with cleanup details
        """
        result = PluginDisposalResult()
        
        if not plugin_instance:
            result.error_message = "Plugin instance is required"
            return result
            
        try:
            # Clean up plugin resources
            if hasattr(plugin_instance, 'dispose'):
                plugin_instance.dispose()
            elif hasattr(plugin_instance, 'cleanup'):
                plugin_instance.cleanup()
            else:
                # Manual cleanup for plugins without explicit dispose method
                if hasattr(plugin_instance, 'resources'):
                    if hasattr(plugin_instance.resources, 'clear'):
                        plugin_instance.resources.clear()
                        
            # Update plugin state
            if hasattr(plugin_instance, 'state'):
                plugin_instance.state = 'disposed'
                result.plugin_state = 'disposed'
            else:
                result.plugin_state = 'disposed'
                
            # Remove from active plugins
            plugin_id = id(plugin_instance)
            if plugin_id in self.active_plugins:
                del self.active_plugins[plugin_id]
                
            # Success
            result.success = True
            result.resources_released = True
            
        except Exception as e:
            result.error_message = f"Plugin disposal failed: {str(e)}"
            result.plugin_state = 'error'
            
        return result
        
    def execute_plugin(self, plugin_instance: Any, execution_context: Any) -> PluginExecutionResult:
        """
        Execute a plugin with proper error handling and timeout management.
        
        Args:
            plugin_instance: Plugin instance to execute
            execution_context: Execution context with data and configuration
            
        Returns:
            PluginExecutionResult: Execution result with output and timing details
        """
        result = PluginExecutionResult()
        
        if not plugin_instance:
            result.error_message = "Plugin instance is required"
            return result
            
        if not execution_context:
            result.error_message = "Execution context is required"
            return result
            
        # Validate plugin state
        if hasattr(plugin_instance, 'state'):
            if plugin_instance.state != 'initialized':
                result.error_message = f"Plugin not initialized (state: {plugin_instance.state})"
                return result
        elif not hasattr(plugin_instance, 'process'):
            result.error_message = "Plugin missing required 'process' method"
            return result
            
        # Get timeout from execution context
        timeout = getattr(execution_context, 'timeout', self.default_timeout)
        
        try:
            with self.execution_lock:
                start_time = time.time()
                
                if timeout:
                    # Execute with timeout
                    try:
                        result_output = self._execute_with_timeout(
                            plugin_instance, execution_context, timeout
                        )
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        # Success
                        result.success = True
                        result.execution_time = execution_time
                        result.output = result_output
                        
                    except TimeoutError as te:
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        result.timeout_occurred = True
                        result.resources_cleaned = True
                        result.error_message = str(te)
                        result.execution_time = execution_time
                        return result
                        
                else:
                    # Execute without timeout
                    result_output = plugin_instance.process(
                        execution_context.data, 
                        execution_context.config
                    )
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Success
                    result.success = True
                    result.execution_time = execution_time
                    result.output = result_output
                
        except Exception as e:
            end_time = time.time()
            result.execution_time = end_time - start_time
            result.error_message = f"Plugin execution failed: {str(e)}"
            result.resources_cleaned = True
            
            # Clean up resources after error
            try:
                self._cleanup_plugin_resources(plugin_instance)
            except:
                pass  # Ignore cleanup errors
                
        return result
        
    def _execute_with_timeout(self, plugin_instance: Any, execution_context: Any, 
                             timeout: float) -> Any:
        """
        Execute plugin with timeout using ThreadPoolExecutor.
        
        Args:
            plugin_instance: Plugin instance to execute
            execution_context: Execution context
            timeout: Timeout in seconds
            
        Returns:
            Any: Plugin execution result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                plugin_instance.process,
                execution_context.data,
                execution_context.config
            )
            
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                # Cancel the future and clean up
                future.cancel()
                self._cleanup_plugin_resources(plugin_instance)
                raise TimeoutError(f"Plugin execution timed out after {timeout}s")
                
    def _cleanup_plugin_resources(self, plugin_instance: Any):
        """
        Clean up plugin resources after error or timeout.
        
        Args:
            plugin_instance: Plugin instance to clean up
        """
        try:
            if hasattr(plugin_instance, 'cleanup'):
                plugin_instance.cleanup()
            elif hasattr(plugin_instance, 'dispose'):
                plugin_instance.dispose()
        except:
            pass  # Ignore cleanup errors
            
    def get_active_plugins(self) -> List[Any]:
        """
        Get list of currently active plugins.
        
        Returns:
            List[Any]: List of active plugin instances
        """
        return list(self.active_plugins.values())
        
    def get_plugin_count(self) -> int:
        """
        Get count of currently active plugins.
        
        Returns:
            int: Number of active plugins
        """
        return len(self.active_plugins)
        
    def shutdown(self):
        """
        Shutdown plugin manager and dispose of all active plugins.
        """
        for plugin_instance in list(self.active_plugins.values()):
            try:
                self.dispose_plugin(plugin_instance)
            except:
                pass  # Ignore disposal errors during shutdown
                
        self.active_plugins.clear()
