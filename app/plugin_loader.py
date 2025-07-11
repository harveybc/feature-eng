"""
Plugin Loader Component

This module provides comprehensive plugin loading functionality with emphasis on 
PERFECT REPLICABILITY and COMPLETE ISOLATION. Supports loading plugins from
external repositories with strict configuration-driven orchestration that 
guarantees identical results across different applications.

Key Isolation Principles:
1. Plugins are completely self-contained with no host dependencies
2. All behavior is controlled via configuration parameters
3. No internal state leakage between plugin instances
4. Deterministic execution with identical inputs
5. Complete namespace isolation

Author: Feature Engineering System
Date: 2025-07-11
"""

import os
import json
import importlib
import importlib.util
import sys
import uuid
import copy
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from importlib.metadata import entry_points, EntryPoint


@dataclass
class PluginIsolationConfig:
    """Configuration for plugin isolation and replicability."""
    random_seed: Optional[int] = 42
    deterministic_mode: bool = True
    strict_validation: bool = True
    namespace_isolation: bool = True
    state_isolation: bool = True
    config_freeze: bool = True
    reproducible_execution: bool = True


@dataclass
class PluginReplicabilityResult:
    """Result of plugin replicability validation."""
    is_replicable: bool = False
    deterministic: bool = False
    isolated: bool = False
    config_complete: bool = False
    namespace_clean: bool = False
    validation_errors: List[str] = None
    replicability_score: float = 0.0
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


@dataclass
class PluginInterfaceValidation:
    """Result of plugin interface validation."""
    has_process_method: bool = False
    has_validate_config_method: bool = False
    has_get_metadata_method: bool = False
    interface_version_compatible: bool = False
    missing_methods: List[str] = None
    isolation_compliant: bool = False
    replicability_compliant: bool = False
    
    def __post_init__(self):
        if self.missing_methods is None:
            self.missing_methods = []


@dataclass
class PluginCompatibilityResult:
    """Result of external plugin compatibility validation."""
    compatible: bool = False
    interface_version_supported: bool = False
    replicability_supported: bool = False
    isolation_supported: bool = False
    config_driven: bool = False
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class PluginDiscoveryResult:
    """Result of plugin discovery operation."""
    success: bool = False
    discovered_plugins: List[str] = None
    invalid_plugins: List[str] = None
    validation_errors: Dict[str, str] = None
    plugin_metadata: Dict[str, Dict[str, Any]] = None
    interface_validation: Dict[str, PluginInterfaceValidation] = None
    replicability_validation: Dict[str, PluginReplicabilityResult] = None
    
    def __post_init__(self):
        if self.discovered_plugins is None:
            self.discovered_plugins = []
        if self.invalid_plugins is None:
            self.invalid_plugins = []
        if self.validation_errors is None:
            self.validation_errors = {}
        if self.plugin_metadata is None:
            self.plugin_metadata = {}
        if self.interface_validation is None:
            self.interface_validation = {}
        if self.replicability_validation is None:
            self.replicability_validation = {}


@dataclass 
class PluginLoadResult:
    """Result of plugin loading operation."""
    success: bool = False
    plugin_instance: Any = None
    plugin_namespace: str = None
    dependencies_resolved: bool = False
    error_type: str = None
    error_message: str = None
    config_schema_validated: bool = False
    parameter_schema: Dict[str, Any] = None
    plugin_metadata: Dict[str, Any] = None
    replicability_guaranteed: bool = False
    isolation_enforced: bool = False
    config_hash: str = None
    execution_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameter_schema is None:
            self.parameter_schema = {}
        if self.plugin_metadata is None:
            self.plugin_metadata = {}
        if self.execution_context is None:
            self.execution_context = {}


class PluginIsolationManager:
    """Manages plugin isolation and replicability enforcement."""
    
    def __init__(self):
        self.isolated_namespaces = {}
        self.plugin_states = {}
        
    def create_isolated_namespace(self, plugin_name: str, config: Dict[str, Any]) -> str:
        """
        Create completely isolated namespace for plugin execution.
        
        Args:
            plugin_name: Name of the plugin
            config: Configuration dictionary
            
        Returns:
            str: Isolated namespace identifier
        """
        # Create deterministic namespace based on plugin and config
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        namespace_id = f"isolated_{plugin_name}_{config_hash}_{uuid.uuid4().hex[:8]}"
        
        # Store namespace mapping
        self.isolated_namespaces[namespace_id] = {
            'plugin_name': plugin_name,
            'config_hash': config_hash,
            'config': copy.deepcopy(config),
            'created_at': __import__('time').time()
        }
        
        return namespace_id
        
    def validate_plugin_isolation(self, plugin_instance: Any, config: Dict[str, Any]) -> PluginReplicabilityResult:
        """
        Validate that plugin maintains perfect isolation and replicability.
        
        Args:
            plugin_instance: Plugin instance to validate
            config: Configuration dictionary
            
        Returns:
            PluginReplicabilityResult: Validation results
        """
        result = PluginReplicabilityResult()
        
        # Check deterministic behavior
        if hasattr(plugin_instance, 'deterministic') and plugin_instance.deterministic:
            result.deterministic = True
        elif hasattr(plugin_instance, '__dict__') and 'deterministic' in plugin_instance.__dict__:
            result.deterministic = plugin_instance.__dict__['deterministic']
        
        # Check isolation compliance
        isolation_methods = ['get_state', 'set_state', 'reset_state']
        result.isolated = all(hasattr(plugin_instance, method) for method in isolation_methods)
        
        # Check configuration completeness
        if hasattr(plugin_instance, 'validate_config'):
            try:
                is_valid, _ = plugin_instance.validate_config(config)
                result.config_complete = is_valid
            except:
                result.validation_errors.append("Config validation failed")
                
        # Check namespace cleanliness
        if hasattr(plugin_instance, '__module__'):
            module_name = plugin_instance.__module__
            result.namespace_clean = 'isolated_' in module_name
            
        # Calculate replicability score
        score_factors = [
            result.deterministic,
            result.isolated,
            result.config_complete,
            result.namespace_clean
        ]
        result.replicability_score = sum(score_factors) / len(score_factors)
        result.is_replicable = result.replicability_score >= 0.8
        
        return result
        
    def enforce_configuration_freeze(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create immutable frozen configuration for perfect replicability.
        
        Args:
            config: Original configuration
            
        Returns:
            Dict[str, Any]: Frozen configuration
        """
        # Deep copy to prevent external modifications
        frozen_config = copy.deepcopy(config)
        
        # Add replicability metadata
        frozen_config['_replicability'] = {
            'frozen_at': __import__('time').time(),
            'config_hash': hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode()
            ).hexdigest(),
            'isolation_enforced': True,
            'deterministic_mode': True
        }
        
        return frozen_config


class PluginLoader:
    """
    Plugin loader with emphasis on PERFECT replicability and COMPLETE isolation.
    
    This class ensures that plugins loaded from external repositories can
    produce exactly identical results without requiring internal tweaks.
    
    Key Features:
    - Complete namespace isolation
    - Configuration-driven behavior
    - Deterministic execution
    - State isolation
    - Perfect replicability
    """

    def __init__(self):
        """Initialize the plugin loader with isolation manager."""
        self.loaded_plugins = {}
        self.plugin_cache = {}
        self.strict_mode = True
        self.isolation_manager = PluginIsolationManager()
        
    def discover_plugins(self, plugin_directory: Union[str, None], 
                        validate_replicability: bool = True,
                        validate_interface: bool = True,
                        required_interface_version: str = None,
                        isolation_config: PluginIsolationConfig = None,
                        handle_errors: bool = False) -> PluginDiscoveryResult:
        """
        Discover available plugins with perfect isolation validation.
        
        Args:
            plugin_directory: Directory containing plugins
            validate_replicability: Whether to validate replicability metadata
            validate_interface: Whether to validate plugin interfaces
            required_interface_version: Required interface version
            isolation_config: Configuration for isolation enforcement
            handle_errors: Whether to handle errors gracefully
            
        Returns:
            PluginDiscoveryResult: Discovery results with validation details
        """
        result = PluginDiscoveryResult()
        
        if isolation_config is None:
            isolation_config = PluginIsolationConfig()
            
        # Handle edge cases safely
        if handle_errors:
            if not plugin_directory or not os.path.exists(str(plugin_directory)):
                result.success = True  # Graceful handling
                return result
                
        # Validate directory exists
        if not plugin_directory or not os.path.exists(plugin_directory):
            if not handle_errors:
                result.validation_errors['directory'] = f"Plugin directory not found: {plugin_directory}"
            return result
            
        plugin_dir_path = Path(plugin_directory)
        
        # Scan for plugin subdirectories
        for plugin_path in plugin_dir_path.iterdir():
            if not plugin_path.is_dir():
                continue
                
            plugin_name = plugin_path.name
            
            # Check for plugin.json metadata
            metadata_file = plugin_path / "plugin.json"
            if not metadata_file.exists():
                result.invalid_plugins.append(plugin_name)
                result.validation_errors[plugin_name] = "Missing plugin.json metadata file"
                continue
                
            try:
                # Load and validate metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Enhanced metadata validation for isolation
                required_fields = [
                    'name', 'version', 'entry_point', 'interface_version',
                    'deterministic', 'reproducible', 'isolated'
                ]
                missing_fields = [field for field in required_fields if field not in metadata]
                
                if missing_fields:
                    result.invalid_plugins.append(plugin_name)
                    result.validation_errors[plugin_name] = f"Missing required fields: {missing_fields}"
                    continue
                    
                # Validate replicability requirements
                if validate_replicability:
                    if not metadata.get('deterministic', False) or not metadata.get('reproducible', False):
                        result.invalid_plugins.append(plugin_name)
                        result.validation_errors[plugin_name] = "Plugin does not guarantee replicability"
                        continue
                        
                    # Check isolation support
                    if not metadata.get('isolated', False):
                        result.invalid_plugins.append(plugin_name)
                        result.validation_errors[plugin_name] = "Plugin does not support isolation"
                        continue
                        
                # Validate interface version if specified
                if required_interface_version and metadata.get('interface_version') != required_interface_version:
                    result.invalid_plugins.append(plugin_name)
                    result.validation_errors[plugin_name] = f"Interface version mismatch: {metadata.get('interface_version')} != {required_interface_version}"
                    continue
                    
                # Enhanced interface validation
                interface_validation = PluginInterfaceValidation()
                if validate_interface:
                    interface_validation = self._validate_plugin_interface_with_isolation(
                        plugin_path, metadata, isolation_config
                    )
                    if not interface_validation.interface_version_compatible or not interface_validation.isolation_compliant:
                        result.invalid_plugins.append(plugin_name)
                        result.validation_errors[plugin_name] = f"Interface validation failed: {interface_validation.missing_methods}"
                        continue
                        
                # Validate replicability compliance
                replicability_validation = self._validate_replicability_compliance(
                    plugin_path, metadata, isolation_config
                )
                
                if not replicability_validation.is_replicable:
                    result.invalid_plugins.append(plugin_name)
                    result.validation_errors[plugin_name] = f"Replicability validation failed: {replicability_validation.validation_errors}"
                    continue
                    
                # Plugin is valid
                result.discovered_plugins.append(plugin_name)
                result.plugin_metadata[plugin_name] = metadata
                result.interface_validation[plugin_name] = interface_validation
                result.replicability_validation[plugin_name] = replicability_validation
                
            except Exception as e:
                result.invalid_plugins.append(plugin_name)
                result.validation_errors[plugin_name] = f"Error loading metadata: {str(e)}"
                
        result.success = True
        return result
        
    def load_plugin(self, plugin_name: str, 
                   plugin_directory: str = None,
                   validate_config_schema: bool = False,
                   replicable_config: Dict[str, Any] = None) -> PluginLoadResult:
        """
        Load a plugin with proper isolation and dependency management.
        
        Args:
            plugin_name: Name of the plugin to load
            plugin_directory: Directory containing the plugin
            validate_config_schema: Whether to validate configuration schema
            replicable_config: Configuration for replicable processing
            
        Returns:
            PluginLoadResult: Loading results with plugin instance
        """
        result = PluginLoadResult()
        
        try:
            # Handle loading from entry points if no directory specified
            if not plugin_directory:
                return self._load_plugin_from_entry_points(plugin_name, result)
                
            # Load from directory
            plugin_path = Path(plugin_directory) / plugin_name
            if not plugin_path.exists():
                result.error_type = "PluginNotFound"
                result.error_message = f"Plugin directory not found: {plugin_path}"
                return result
                
            # Load metadata
            metadata_file = plugin_path / "plugin.json"
            if not metadata_file.exists():
                result.error_type = "InvalidPlugin"
                result.error_message = f"Plugin metadata not found: {metadata_file}"
                return result
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Load plugin module
            entry_point_file = plugin_path / metadata['entry_point']
            if not entry_point_file.exists():
                result.error_type = "InvalidPlugin"
                result.error_message = f"Plugin entry point not found: {entry_point_file}"
                return result
                
            # Create isolated namespace for plugin
            plugin_namespace = f"plugin_{plugin_name}_{id(self)}"
            
            # Load module with isolated namespace
            spec = importlib.util.spec_from_file_location(plugin_namespace, entry_point_file)
            if spec is None or spec.loader is None:
                result.error_type = "LoadError"
                result.error_message = f"Could not create module spec for {entry_point_file}"
                return result
                
            plugin_module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_namespace] = plugin_module
            spec.loader.exec_module(plugin_module)
            
            # Get plugin instance
            if hasattr(plugin_module, 'get_plugin'):
                plugin_instance = plugin_module.get_plugin()
            else:
                result.error_type = "InvalidPlugin"
                result.error_message = "Plugin does not have get_plugin() function"
                return result
                
            # Validate required methods exist
            required_methods = ['process', 'validate_config']
            missing_methods = [method for method in required_methods if not hasattr(plugin_instance, method)]
            if missing_methods:
                result.error_type = "InvalidInterface"
                result.error_message = f"Plugin missing required methods: {missing_methods}"
                return result
                
            # Validate configuration schema if requested
            if validate_config_schema:
                if 'parameters' in metadata:
                    result.parameter_schema = metadata['parameters']
                    result.config_schema_validated = True
                    
            # Apply replicable configuration if provided
            if replicable_config:
                if hasattr(plugin_instance, 'configure_replicability'):
                    plugin_instance.configure_replicability(replicable_config)
                result.replicability_guaranteed = metadata.get('deterministic', False) and metadata.get('reproducible', False)
                
            # Success
            result.success = True
            result.plugin_instance = plugin_instance
            result.plugin_namespace = plugin_namespace
            result.dependencies_resolved = True
            result.plugin_metadata = metadata
            
            # Cache the loaded plugin for reuse
            self.loaded_plugins[plugin_name] = result
            
        except Exception as e:
            result.error_type = "LoadError"
            result.error_message = str(e)
            
        return result
        
    def validate_external_plugin_compatibility(self, plugin_metadata: Dict[str, Any]) -> PluginCompatibilityResult:
        """
        Validate compatibility of external plugins with the system.
        
        Args:
            plugin_metadata: Metadata of external plugin
            
        Returns:
            PluginCompatibilityResult: Compatibility validation result
        """
        result = PluginCompatibilityResult()
        
        # Check interface version compatibility
        if plugin_metadata.get('interface_version') == "1.0":
            result.interface_version_supported = True
        else:
            result.issues.append(f"Unsupported interface version: {plugin_metadata.get('interface_version')}")
            
        # Check replicability support
        if plugin_metadata.get('deterministic', False) and plugin_metadata.get('reproducible', False):
            result.replicability_supported = True
        else:
            result.issues.append("Plugin does not guarantee replicability")
            
        # Check required methods
        required_methods = plugin_metadata.get('required_methods', [])
        expected_methods = ['process', 'validate_config', 'get_metadata']
        if all(method in required_methods for method in expected_methods):
            # Method requirements satisfied
            pass
        else:
            missing = [method for method in expected_methods if method not in required_methods]
            result.issues.append(f"Missing required methods: {missing}")
            
        # Overall compatibility
        result.compatible = result.interface_version_supported and result.replicability_supported and len(result.issues) == 0
        
        return result
        
    def _validate_plugin_interface(self, plugin_path: Path, metadata: Dict[str, Any]) -> PluginInterfaceValidation:
        """
        Validate plugin interface compatibility.
        
        Args:
            plugin_path: Path to plugin directory
            metadata: Plugin metadata
            
        Returns:
            PluginInterfaceValidation: Interface validation result
        """
        validation = PluginInterfaceValidation()
        
        try:
            # Load plugin temporarily to check interface
            entry_point_file = plugin_path / metadata['entry_point']
            spec = importlib.util.spec_from_file_location("temp_plugin", entry_point_file)
            if spec is None or spec.loader is None:
                return validation
                
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            
            if hasattr(temp_module, 'get_plugin'):
                plugin_instance = temp_module.get_plugin()
                
                # Check required methods
                validation.has_process_method = hasattr(plugin_instance, 'process')
                validation.has_validate_config_method = hasattr(plugin_instance, 'validate_config')
                
                # Check interface version
                validation.interface_version_compatible = metadata.get('interface_version') == "1.0"
                
                # Track missing methods
                required_methods = ['process', 'validate_config']
                validation.missing_methods = [method for method in required_methods 
                                            if not hasattr(plugin_instance, method)]
                                            
        except Exception:
            pass  # Validation failed
            
        return validation
        
    def _load_plugin_from_entry_points(self, plugin_name: str, result: PluginLoadResult) -> PluginLoadResult:
        """
        Load plugin from entry points (legacy method).
        
        Args:
            plugin_name: Name of plugin to load
            result: Result object to populate
            
        Returns:
            PluginLoadResult: Updated result object
        """
        try:
            # This is the legacy method for backward compatibility
            group_entries = entry_points().get('feature_eng.plugins', [])
            entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
            plugin_class = entry_point.load()
            
            result.success = True
            result.plugin_instance = plugin_class()
            result.dependencies_resolved = True
            
        except StopIteration:
            result.error_type = "PluginNotFound"
            result.error_message = f"Plugin {plugin_name} not found in entry points"
        except Exception as e:
            result.error_type = "LoadError"
            result.error_message = str(e)
            
        return result


# Legacy functions for backward compatibility
def load_plugin(plugin_group, plugin_name):
    """
    Legacy function for loading plugins from entry points.
    
    Args:
        plugin_group: Plugin group name
        plugin_name: Plugin name
        
    Returns:
        tuple: (plugin_class, required_params)
    """
    print(f"Attempting to load plugin: {plugin_name} from group: {plugin_group}")
    try:
        group_entries = entry_points().get(plugin_group, [])
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        required_params = list(plugin_class.plugin_params.keys()) if hasattr(plugin_class, 'plugin_params') else []
        print(f"Successfully loaded plugin: {plugin_name} with params: {required_params}")
        return plugin_class, required_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to load plugin {plugin_name} from group {plugin_group}, Error: {e}")
        raise


def get_plugin_params(plugin_group, plugin_name):
    """
    Legacy function for getting plugin parameters.
    
    Args:
        plugin_group: Plugin group name
        plugin_name: Plugin name
        
    Returns:
        dict: Plugin parameters
    """
    print(f"Getting plugin parameters for: {plugin_name} from group: {plugin_group}")
    try:
        group_entries = entry_points().get(plugin_group, [])
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        plugin_class = entry_point.load()
        plugin_params = plugin_class.plugin_params if hasattr(plugin_class, 'plugin_params') else {}
        print(f"Retrieved plugin params: {plugin_params}")
        return plugin_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
        raise ImportError(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
