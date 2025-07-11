"""
Plugin Registry Component

This module provides comprehensive plugin registry functionality for managing
plugin metadata, registration, and discovery. Supports querying plugins by
capabilities, requirements, and other metadata fields with emphasis on 
replicability and deterministic execution.

Author: Feature Engineering System
Date: 2025-07-10
"""

import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class PluginMetadata:
    """Plugin metadata container."""
    name: str
    version: str
    description: str
    author: str
    interface_version: str
    capabilities: List[str]
    requirements: List[str] = None
    deterministic: bool = False
    reproducible: bool = False
    plugin_id: str = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []
        if self.plugin_id is None:
            self.plugin_id = str(uuid.uuid4())


@dataclass
class PluginRegistrationResult:
    """Result of plugin registration operation."""
    success: bool = False
    plugin_id: str = None
    metadata_validated: bool = False
    missing_metadata: List[str] = None
    validation_errors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.missing_metadata is None:
            self.missing_metadata = []
        if self.validation_errors is None:
            self.validation_errors = {}


@dataclass
class PluginQueryResult:
    """Result of plugin query operation."""
    success: bool = False
    matching_plugins: List[PluginMetadata] = None
    
    def __post_init__(self):
        if self.matching_plugins is None:
            self.matching_plugins = []


@dataclass
class PluginListResult:
    """Result of plugin listing operation."""
    success: bool = False
    plugins: List[PluginMetadata] = None
    
    def __post_init__(self):
        if self.plugins is None:
            self.plugins = []


@dataclass
class PluginSearchResult:
    """Result of plugin search operation."""
    success: bool = False
    matching_plugins: List[PluginMetadata] = None
    
    def __post_init__(self):
        if self.matching_plugins is None:
            self.matching_plugins = []


class PluginRegistry:
    """
    Plugin registry for managing plugin metadata and discovery.
    
    This class provides comprehensive plugin registration, validation,
    and query capabilities with emphasis on replicability and
    deterministic execution.
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugins_by_name: Dict[str, str] = {}  # name -> plugin_id mapping
        self.strict_mode = True
        
    def register_plugin(self, plugin_metadata: Dict[str, Any], 
                       strict_replicability: bool = False,
                       required_interface_version: str = None,
                       handle_errors: bool = False) -> PluginRegistrationResult:
        """
        Register a plugin with the registry.
        
        Args:
            plugin_metadata: Plugin metadata dictionary
            strict_replicability: Whether to enforce strict replicability requirements
            required_interface_version: Required interface version
            handle_errors: Whether to handle errors gracefully
            
        Returns:
            PluginRegistrationResult: Registration result with validation details
        """
        result = PluginRegistrationResult()
        
        # Handle edge cases safely
        if handle_errors:
            if not plugin_metadata or not isinstance(plugin_metadata, dict):
                result.validation_errors['input'] = "Invalid plugin metadata input"
                return result
                
        # Validate metadata is provided
        if not plugin_metadata:
            result.validation_errors['metadata'] = "Plugin metadata is required"
            return result
            
        # Validate required fields
        required_fields = ['name', 'version', 'description', 'author', 'interface_version', 'capabilities']
        missing_fields = []
        
        for field in required_fields:
            if field not in plugin_metadata or not plugin_metadata[field]:
                missing_fields.append(field)
                
        if missing_fields:
            result.missing_metadata = missing_fields
            result.validation_errors['missing_fields'] = f"Missing required fields: {missing_fields}"
            return result
            
        plugin_name = plugin_metadata['name']
        
        # Check for duplicate registration
        if plugin_name in self.plugins_by_name:
            result.validation_errors['duplicate'] = f"Plugin '{plugin_name}' is already registered"
            return result
            
        # Validate interface version if required
        if required_interface_version:
            if plugin_metadata.get('interface_version') != required_interface_version:
                result.validation_errors['interface_version'] = f"Interface version mismatch: expected {required_interface_version}, got {plugin_metadata.get('interface_version')}"
                return result
                
        # Validate replicability requirements if strict mode
        if strict_replicability:
            if not plugin_metadata.get('deterministic', False) or not plugin_metadata.get('reproducible', False):
                result.validation_errors['replicability'] = "Plugin must guarantee deterministic and reproducible execution"
                return result
                
        # Create plugin metadata object
        try:
            plugin_meta = PluginMetadata(
                name=plugin_metadata['name'],
                version=plugin_metadata['version'],
                description=plugin_metadata['description'],
                author=plugin_metadata['author'],
                interface_version=plugin_metadata['interface_version'],
                capabilities=plugin_metadata['capabilities'],
                requirements=plugin_metadata.get('requirements', []),
                deterministic=plugin_metadata.get('deterministic', False),
                reproducible=plugin_metadata.get('reproducible', False)
            )
            
            # Store additional metadata fields
            for key, value in plugin_metadata.items():
                if not hasattr(plugin_meta, key):
                    setattr(plugin_meta, key, value)
                    
            # Register the plugin
            plugin_id = plugin_meta.plugin_id
            self.plugins[plugin_id] = plugin_meta
            self.plugins_by_name[plugin_name] = plugin_id
            
            # Success
            result.success = True
            result.plugin_id = plugin_id
            result.metadata_validated = True
            
        except Exception as e:
            result.validation_errors['registration'] = f"Failed to register plugin: {str(e)}"
            
        return result
        
    def get_plugin(self, plugin_id: str) -> Optional[PluginMetadata]:
        """
        Retrieve a plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            PluginMetadata or None: Plugin metadata if found
        """
        return self.plugins.get(plugin_id)
        
    def get_plugin_by_name(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Retrieve a plugin by name.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            PluginMetadata or None: Plugin metadata if found
        """
        plugin_id = self.plugins_by_name.get(plugin_name)
        if plugin_id:
            return self.plugins.get(plugin_id)
        return None
        
    def query_by_capabilities(self, capabilities: List[str], logic: str = 'OR') -> PluginQueryResult:
        """
        Query plugins by their capabilities.
        
        Args:
            capabilities: List of required capabilities
            logic: Query logic ('AND' or 'OR')
            
        Returns:
            PluginQueryResult: Query results with matching plugins
        """
        result = PluginQueryResult()
        matching_plugins = []
        
        for plugin in self.plugins.values():
            if logic.upper() == 'AND':
                # Plugin must have ALL specified capabilities
                if all(cap in plugin.capabilities for cap in capabilities):
                    matching_plugins.append(plugin)
            else:  # OR logic (default)
                # Plugin must have at least ONE specified capability
                if any(cap in plugin.capabilities for cap in capabilities):
                    matching_plugins.append(plugin)
                    
        result.success = True
        result.matching_plugins = matching_plugins
        return result
        
    def query_by_requirements(self, requirements: List[str]) -> PluginQueryResult:
        """
        Query plugins by their dependency requirements.
        
        Args:
            requirements: List of required dependencies
            
        Returns:
            PluginQueryResult: Query results with matching plugins
        """
        result = PluginQueryResult()
        matching_plugins = []
        
        for plugin in self.plugins.values():
            # Check if plugin has compatible requirements
            plugin_requirements_str = ' '.join(plugin.requirements).lower()
            
            has_requirements = True
            for req in requirements:
                if req.lower() not in plugin_requirements_str:
                    has_requirements = False
                    break
                    
            if has_requirements:
                matching_plugins.append(plugin)
                
        result.success = True
        result.matching_plugins = matching_plugins
        return result
        
    def query_by_interface_version(self, interface_version: str) -> PluginQueryResult:
        """
        Query plugins by interface version.
        
        Args:
            interface_version: Required interface version
            
        Returns:
            PluginQueryResult: Query results with matching plugins
        """
        result = PluginQueryResult()
        matching_plugins = []
        
        for plugin in self.plugins.values():
            if plugin.interface_version == interface_version:
                matching_plugins.append(plugin)
                
        result.success = True
        result.matching_plugins = matching_plugins
        return result
        
    def list_all_plugins(self) -> PluginListResult:
        """
        List all registered plugins.
        
        Returns:
            PluginListResult: Result with all registered plugins
        """
        result = PluginListResult()
        result.success = True
        result.plugins = list(self.plugins.values())
        return result
        
    def search_by_metadata(self, name: str = None, description: str = None, 
                          author: str = None, **kwargs) -> PluginSearchResult:
        """
        Search plugins by metadata fields.
        
        Args:
            name: Plugin name to search for
            description: Description text to search for
            author: Author name to search for
            **kwargs: Additional metadata fields to search
            
        Returns:
            PluginSearchResult: Search results with matching plugins
        """
        result = PluginSearchResult()
        matching_plugins = []
        
        for plugin in self.plugins.values():
            match = True
            
            # Check name match
            if name and name.lower() not in plugin.name.lower():
                match = False
                
            # Check description match
            if description and description.lower() not in plugin.description.lower():
                match = False
                
            # Check author match
            if author and author.lower() not in plugin.author.lower():
                match = False
                
            # Check additional metadata fields
            for key, value in kwargs.items():
                plugin_value = getattr(plugin, key, None)
                if plugin_value and value:
                    if str(value).lower() not in str(plugin_value).lower():
                        match = False
                        break
                        
            if match:
                matching_plugins.append(plugin)
                
        result.success = True
        result.matching_plugins = matching_plugins
        return result
        
    def register_multiple_test_plugins(self):
        """
        Register multiple test plugins for testing purposes.
        This method is used by unit tests.
        """
        # Technical indicator plugin
        tech_metadata = {
            "name": "technical_indicator",
            "version": "1.0.0",
            "description": "Technical indicator calculations",
            "author": "Test Author",
            "interface_version": "1.0",
            "capabilities": ["technical_indicators", "real_time"],
            "deterministic": True,
            "reproducible": True
        }
        
        # Decomposition plugin
        decomp_metadata = {
            "name": "decomposition_processor",
            "version": "1.0.0",
            "description": "Feature decomposition processor",
            "author": "Test Author",
            "interface_version": "1.0",
            "capabilities": ["post_processing", "decomposition"],
            "deterministic": True,
            "reproducible": True
        }
        
        # SSA plugin
        ssa_metadata = {
            "name": "ssa_processor",
            "version": "1.0.0",
            "description": "Singular Spectrum Analysis processor",
            "author": "Test Author",
            "interface_version": "1.0",
            "capabilities": ["signal_processing", "time_series"],
            "deterministic": True,
            "reproducible": True
        }
        
        # Register test plugins
        self.register_plugin(tech_metadata)
        self.register_plugin(decomp_metadata)
        self.register_plugin(ssa_metadata)
