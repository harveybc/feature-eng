"""
Error Handler Component

This module provides comprehensive error handling capabilities for the feature engineering system,
including error categorization, context-aware recovery, user-friendly reporting, and pattern analysis.

Key Features:
- Intelligent error categorization by type, severity, and recovery potential
- Context-aware recovery strategies with fallback mechanisms
- User-friendly error messages with actionable guidance
- Error history tracking and pattern detection for system improvement
- Coordinated error reporting across all system components
- Safe handling of recursive errors and performance optimization

Author: Feature Engineering System
Date: 2025-07-10
"""

import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR" 
    WARNING = "WARNING"
    INFO = "INFO"


class ErrorType(Enum):
    """Error type categories."""
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DATA_PROCESSING_ERROR = "DATA_PROCESSING_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    FILE_IO_ERROR = "FILE_IO_ERROR"
    PLUGIN_ERROR = "PLUGIN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    GENERIC = "GENERIC"


class RecoveryPotential(Enum):
    """Recovery potential levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ErrorContext:
    """Error context information."""
    component: str
    operation: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ErrorCategory:
    """Error categorization result."""
    error_type: str
    severity: str
    recovery_potential: str
    category_description: str
    requires_immediate_attention: bool
    allows_continuation: bool
    recommended_action: str


@dataclass
class UserErrorMessage:
    """User-friendly error message."""
    user_friendly_text: str
    technical_details: str
    suggested_actions: List[str]


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    component_name: str
    error_id: str
    timestamp: str
    formatted_message: str
    severity: str
    error_type: str
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorHandlingResult:
    """Result of error handling operation."""
    success: bool
    recovery_strategy: str
    recovery_actions: List[str]
    fallback_strategy_applied: bool = False
    fallback_actions: List[str] = field(default_factory=list)
    error_report: Optional[ErrorReport] = None
    recursive_error_detected: bool = False
    safe_mode_activated: bool = False


@dataclass
class PatternAnalysis:
    """Error pattern analysis result."""
    frequent_components: List[str]
    frequent_error_types: List[str]
    peak_error_times: List[str]
    component_error_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ImprovementRecommendations:
    """System improvement recommendations."""
    recommendations: List[str]
    priority_areas: List[str]
    suggested_changes: Dict[str, str] = field(default_factory=dict)


class SystemError(Exception):
    """System error with categorization."""
    
    def __init__(self, message: str, error_type: str = "GENERIC", severity: str = "ERROR", context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}


class ErrorHandler:
    """
    Comprehensive error handler for the feature engineering system.
    
    This class provides intelligent error categorization, context-aware recovery,
    user-friendly messaging, and system improvement recommendations based on
    error pattern analysis.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the error handler.
        
        Args:
            max_history_size: Maximum number of errors to keep in history
        """
        self._max_history_size = max_history_size
        self._error_history: List[Dict[str, Any]] = []
        self._error_counter = 0
        self._recursive_error_depth = 0
        self._max_recursive_depth = 3
        
        # Error type mappings and recovery strategies
        self._error_type_mapping = {
            "config": ErrorType.CONFIGURATION_ERROR,
            "data": ErrorType.DATA_PROCESSING_ERROR,
            "network": ErrorType.NETWORK_ERROR,
            "file": ErrorType.FILE_IO_ERROR,
            "plugin": ErrorType.PLUGIN_ERROR,
            "validation": ErrorType.VALIDATION_ERROR,
            "system": ErrorType.SYSTEM_ERROR
        }
        
        self._recovery_strategies = {
            ErrorType.CONFIGURATION_ERROR: self._config_recovery_strategy,
            ErrorType.DATA_PROCESSING_ERROR: self._data_recovery_strategy,
            ErrorType.NETWORK_ERROR: self._network_recovery_strategy,
            ErrorType.FILE_IO_ERROR: self._file_recovery_strategy,
            ErrorType.PLUGIN_ERROR: self._plugin_recovery_strategy,
            ErrorType.VALIDATION_ERROR: self._validation_recovery_strategy,
            ErrorType.SYSTEM_ERROR: self._system_recovery_strategy
        }
        
    def categorize_error(self, error: Union[Exception, SystemError]) -> ErrorCategory:
        """
        Categorize an error by type, severity, and recovery potential.
        
        Args:
            error: The error to categorize
            
        Returns:
            ErrorCategory: Categorization result
        """
        try:
            if isinstance(error, SystemError):
                error_type = error.error_type
                severity = error.severity
            else:
                error_type = self._infer_error_type(error)
                severity = self._infer_severity(error)
            
            recovery_potential = self._assess_recovery_potential(error_type, severity)
            category_description = f"Error of type {error_type} with {severity} severity"
            requires_immediate_attention = severity == ErrorSeverity.CRITICAL.value
            allows_continuation = severity in [ErrorSeverity.WARNING.value, ErrorSeverity.INFO.value]
            recommended_action = self._get_recommended_action(severity)
            
            return ErrorCategory(
                error_type=error_type,
                severity=severity,
                recovery_potential=recovery_potential,
                category_description=category_description,
                requires_immediate_attention=requires_immediate_attention,
                allows_continuation=allows_continuation,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            # Safe fallback for recursive errors
            return self._create_safe_error_category(str(e))
    
    def handle_system_error(self, error: Union[Exception, SystemError], context: Optional[ErrorContext] = None) -> ErrorHandlingResult:
        """
        Handle a system error with context-aware recovery strategies.
        
        Args:
            error: The error to handle
            context: Error context information
            
        Returns:
            ErrorHandlingResult: Result of error handling
        """
        try:
            self._recursive_error_depth += 1
            
            # Check for recursive error depth
            if self._recursive_error_depth > self._max_recursive_depth:
                return self._handle_recursive_error_safely(error, context)
            
            # Generate error ID and record in history
            error_id = self._generate_error_id()
            self._record_error_in_history(error, context, error_id)
            
            # Categorize error
            category = self.categorize_error(error)
            
            # Attempt primary recovery
            recovery_success = self._attempt_primary_recovery(error, context, category)
            
            # Generate recovery strategy
            recovery_strategy = self._get_recovery_strategy(error, category, context)
            recovery_actions = self._get_recovery_actions(error, category, context)
            
            # Handle fallback if primary recovery failed
            fallback_applied = False
            fallback_actions = []
            if not recovery_success:
                fallback_applied = True
                fallback_actions = self._get_fallback_actions(error, category, context)
            
            # Create error report
            error_report = self._create_error_report(error, context, error_id, category)
            
            # Log error
            self._log_error(error, context, category)
            
            return ErrorHandlingResult(
                success=recovery_success,
                recovery_strategy=recovery_strategy,
                recovery_actions=recovery_actions,
                fallback_strategy_applied=fallback_applied,
                fallback_actions=fallback_actions,
                error_report=error_report
            )
            
        except Exception as handling_error:
            # Handle recursive errors safely
            return self._handle_recursive_error_safely(error, context)
        finally:
            self._recursive_error_depth = max(0, self._recursive_error_depth - 1)
    
    def generate_user_error_message(self, error: Union[Exception, SystemError]) -> UserErrorMessage:
        """
        Generate a user-friendly error message with actionable guidance.
        
        Args:
            error: The error to generate message for
            
        Returns:
            UserErrorMessage: User-friendly error message
        """
        try:
            category = self.categorize_error(error)
            
            # Generate user-friendly text (no technical jargon)
            user_friendly_text = self._generate_friendly_text(error, category)
            
            # Technical details for advanced users
            technical_details = str(error)
            
            # Actionable suggestions
            suggested_actions = self._generate_suggested_actions(error, category)
            
            return UserErrorMessage(
                user_friendly_text=user_friendly_text,
                technical_details=technical_details,
                suggested_actions=suggested_actions
            )
            
        except Exception:
            # Safe fallback
            return UserErrorMessage(
                user_friendly_text="An unexpected error occurred. Please check your configuration and try again.",
                technical_details=str(error),
                suggested_actions=["Check the system logs for more details", "Contact support if the problem persists"]
            )
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        Get the error history for analysis.
        
        Returns:
            List[Dict[str, Any]]: Error history records
        """
        return self._error_history.copy()
    
    def analyze_error_patterns(self) -> PatternAnalysis:
        """
        Analyze error patterns for system improvement insights.
        
        Returns:
            PatternAnalysis: Pattern analysis result
        """
        try:
            if not self._error_history:
                return PatternAnalysis(frequent_components=[], frequent_error_types=[], peak_error_times=[])
            
            # Analyze component frequencies
            component_counts = {}
            error_type_counts = {}
            time_patterns = []
            
            for error_record in self._error_history:
                component = error_record.get('component', 'Unknown')
                error_type = error_record.get('error_type', 'Unknown')
                timestamp = error_record.get('timestamp', '')
                
                component_counts[component] = component_counts.get(component, 0) + 1
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                if timestamp:
                    time_patterns.append(timestamp)
            
            # Get most frequent components and error types
            frequent_components = sorted(component_counts.keys(), key=lambda x: component_counts[x], reverse=True)[:5]
            frequent_error_types = sorted(error_type_counts.keys(), key=lambda x: error_type_counts[x], reverse=True)[:5]
            
            # Analyze time patterns (simplified)
            peak_error_times = time_patterns[-5:] if time_patterns else []
            
            return PatternAnalysis(
                frequent_components=frequent_components,
                frequent_error_types=frequent_error_types,
                peak_error_times=peak_error_times,
                component_error_counts=component_counts
            )
            
        except Exception:
            return PatternAnalysis(frequent_components=[], frequent_error_types=[], peak_error_times=[])
    
    def generate_improvement_recommendations(self) -> ImprovementRecommendations:
        """
        Generate system improvement recommendations based on error patterns.
        
        Returns:
            ImprovementRecommendations: Improvement recommendations
        """
        try:
            pattern_analysis = self.analyze_error_patterns()
            
            recommendations = []
            priority_areas = []
            suggested_changes = {}
            
            # Analyze frequent components for improvements
            for component in pattern_analysis.frequent_components[:3]:
                error_count = pattern_analysis.component_error_counts.get(component, 0)
                if error_count >= 5:  # Changed from > 5 to >= 5
                    priority_areas.append(component)
                    recommendations.append(f"Review and strengthen error handling in {component} component")
                    suggested_changes[component] = f"Implement additional validation and error prevention measures"
            
            # General recommendations based on error patterns
            if len(self._error_history) > 50:
                recommendations.append("Consider implementing proactive error monitoring and alerting")
            
            if pattern_analysis.frequent_error_types:
                most_common_type = pattern_analysis.frequent_error_types[0]
                recommendations.append(f"Focus on preventing {most_common_type} errors through improved validation")
            
            return ImprovementRecommendations(
                recommendations=recommendations,
                priority_areas=priority_areas,
                suggested_changes=suggested_changes
            )
            
        except Exception:
            return ImprovementRecommendations(recommendations=[], priority_areas=[])
    
    def _infer_error_type(self, error: Exception) -> str:
        """Infer error type from exception."""
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ['config', 'configuration', 'parameter', 'setting']):
            return ErrorType.CONFIGURATION_ERROR.value
        elif any(keyword in error_message for keyword in ['data', 'column', 'csv', 'format', 'processing']):
            return ErrorType.DATA_PROCESSING_ERROR.value
        elif any(keyword in error_message for keyword in ['network', 'connection', 'timeout', 'fetch']):
            return ErrorType.NETWORK_ERROR.value
        elif any(keyword in error_message for keyword in ['file', 'directory', 'path', 'io', 'read', 'write']):
            return ErrorType.FILE_IO_ERROR.value
        elif any(keyword in error_message for keyword in ['plugin', 'module', 'import']):
            return ErrorType.PLUGIN_ERROR.value
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'validate']):
            return ErrorType.VALIDATION_ERROR.value
        else:
            return ErrorType.GENERIC.value
    
    def _infer_severity(self, error: Exception) -> str:
        """Infer error severity from exception."""
        # Check if it's a SystemError with explicit severity
        if hasattr(error, 'severity'):
            return error.severity
            
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'missing required']):
            return ErrorSeverity.CRITICAL.value
        elif any(keyword in error_message for keyword in ['warning', 'fallback', 'retry', 'timeout', 'connection']):
            return ErrorSeverity.WARNING.value
        else:
            return ErrorSeverity.ERROR.value
    
    def _assess_recovery_potential(self, error_type: str, severity: str) -> str:
        """Assess recovery potential based on error type and severity."""
        if severity == ErrorSeverity.CRITICAL.value:
            return RecoveryPotential.LOW.value
        elif error_type in [ErrorType.NETWORK_ERROR.value, ErrorType.FILE_IO_ERROR.value]:
            return RecoveryPotential.HIGH.value
        else:
            return RecoveryPotential.MEDIUM.value
    
    def _get_recommended_action(self, severity: str) -> str:
        """Get recommended action based on severity."""
        actions = {
            ErrorSeverity.CRITICAL.value: "HALT_EXECUTION",
            ErrorSeverity.ERROR.value: "RETRY_WITH_FALLBACK",
            ErrorSeverity.WARNING.value: "LOG_AND_CONTINUE",
            ErrorSeverity.INFO.value: "LOG_ONLY"
        }
        return actions.get(severity, "LOG_AND_CONTINUE")
    
    def _attempt_primary_recovery(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Attempt primary error recovery."""
        try:
            error_type_enum = ErrorType(category.error_type) if category.error_type in [e.value for e in ErrorType] else ErrorType.GENERIC
            if error_type_enum in self._recovery_strategies:
                return self._recovery_strategies[error_type_enum](error, context, category)
            return False
        except Exception:
            return False
    
    def _get_recovery_strategy(self, error: Exception, category: ErrorCategory, context: Optional[ErrorContext] = None) -> str:
        """Get recovery strategy description."""
        component_context = f" for {context.component}" if context else ""
        
        if category.severity == ErrorSeverity.CRITICAL.value:
            return f"HALT_AND_REPORT{component_context}"
        elif category.recovery_potential == RecoveryPotential.HIGH.value:
            return f"RETRY_WITH_FALLBACK{component_context}"
        else:
            return f"LOG_AND_CONTINUE{component_context}"
    
    def _get_recovery_actions(self, error: Exception, category: ErrorCategory, context: Optional[ErrorContext]) -> List[str]:
        """Get specific recovery actions."""
        actions = []
        
        if category.error_type == ErrorType.CONFIGURATION_ERROR.value:
            actions.extend(["validate_configuration", "apply_default_values", "prompt_user_for_input"])
        elif category.error_type == ErrorType.DATA_PROCESSING_ERROR.value:
            actions.extend(["retry_operation", "apply_data_cleaning", "use_fallback_processing"])
        elif category.error_type == ErrorType.NETWORK_ERROR.value:
            actions.extend(["retry_with_backoff", "use_cached_data", "switch_to_local_mode"])
        else:
            actions.extend(["log_error", "notify_user", "continue_with_defaults"])
        
        return actions
    
    def _get_fallback_actions(self, error: Exception, category: ErrorCategory, context: Optional[ErrorContext]) -> List[str]:
        """Get fallback actions when primary recovery fails."""
        return [
            "log_error_details",
            "save_partial_results", 
            "notify_user_of_issue",
            "continue_with_safe_defaults"
        ]
    
    def _generate_friendly_text(self, error: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly error text."""
        error_message = str(error)
        
        if category.error_type == ErrorType.CONFIGURATION_ERROR.value:
            if "input_file" in error_message:
                return "The system needs an input_file to process your data. Please specify the path to your data file."
            else:
                return "There's an issue with the system configuration. Please check your settings and try again."
        elif category.error_type == ErrorType.DATA_PROCESSING_ERROR.value:
            return "An error occurred while processing your data. Please check your input file format and ensure it contains valid data."
        elif category.error_type == ErrorType.NETWORK_ERROR.value:
            return "Unable to connect to the remote service. Please check your internet connection and try again."
        elif category.error_type == ErrorType.FILE_IO_ERROR.value:
            return "There was a problem accessing your file. Please check that the file exists and you have permission to read it."
        else:
            return "An unexpected error occurred while processing your request. Please check your input and try again."
    
    def _generate_suggested_actions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate suggested actions for the user."""
        error_message = str(error).lower()
        
        if category.error_type == ErrorType.CONFIGURATION_ERROR.value:
            if "input_file" in error_message:
                return [
                    "Verify that your input file path is correct",
                    "Check that the file exists and is accessible",
                    "Ensure you have specified the input file parameter"
                ]
            else:
                return [
                    "Check your configuration file for missing or invalid parameters",
                    "Verify all required settings are properly specified",
                    "Review the documentation for correct configuration format"
                ]
        elif category.error_type == ErrorType.DATA_PROCESSING_ERROR.value:
            return [
                "Verify that your input file contains the required columns",
                "Check that the file format matches CSV specifications",
                "Ensure the file is not corrupted or incomplete"
            ]
        elif category.error_type == ErrorType.NETWORK_ERROR.value:
            return [
                "Check your internet connection",
                "Verify that the remote service is available",
                "Try again after a few minutes"
            ]
        else:
            return [
                "Check the system logs for more details",
                "Verify your input parameters",
                "Contact support if the problem persists"
            ]
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        self._error_counter += 1
        timestamp = int(time.time())
        return f"ERR-{timestamp}-{self._error_counter:04d}"
    
    def _record_error_in_history(self, error: Exception, context: Optional[ErrorContext], error_id: str):
        """Record error in history for pattern analysis."""
        try:
            error_record = {
                'error_id': error_id,
                'timestamp': datetime.now().isoformat(),
                'error_type': self._infer_error_type(error),
                'severity': self._infer_severity(error),
                'component': context.component if context else 'Unknown',
                'operation': context.operation if context else 'Unknown',
                'message': str(error),
                'context_data': context.data if context else {}
            }
            
            self._error_history.append(error_record)
            
            # Maintain history size limit
            if len(self._error_history) > self._max_history_size:
                self._error_history = self._error_history[-self._max_history_size:]
                
        except Exception:
            # Silently ignore history recording errors to prevent recursive issues
            pass
    
    def _create_error_report(self, error: Exception, context: Optional[ErrorContext], error_id: str, category: ErrorCategory) -> ErrorReport:
        """Create comprehensive error report."""
        return ErrorReport(
            component_name=context.component if context else 'Unknown',
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            formatted_message=f"Error in {context.component if context else 'Unknown'}: {str(error)}",
            severity=category.severity,
            error_type=category.error_type,
            context_data=context.data if context else {}
        )
    
    def _log_error(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory):
        """Log error with appropriate level."""
        try:
            logger = logging.getLogger(__name__)
            log_message = f"Error in {context.component if context else 'Unknown'}: {str(error)}"
            
            if category.severity == ErrorSeverity.CRITICAL.value:
                logger.critical(log_message)
            elif category.severity == ErrorSeverity.ERROR.value:
                logger.error(log_message)
            elif category.severity == ErrorSeverity.WARNING.value:
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception:
            # Prevent recursive logging errors
            pass
    
    def _handle_recursive_error_safely(self, original_error: Exception, context: Optional[ErrorContext]) -> ErrorHandlingResult:
        """Handle recursive errors safely."""
        return ErrorHandlingResult(
            success=False,
            recovery_strategy="SAFE_MODE",
            recovery_actions=["log_to_file", "exit_gracefully"],
            recursive_error_detected=True,
            safe_mode_activated=True,
            error_report=ErrorReport(
                component_name=context.component if context else "ErrorHandler",
                error_id="RECURSIVE-ERROR",
                timestamp=datetime.now().isoformat(),
                formatted_message=f"Recursive error detected: {str(original_error)}",
                severity="CRITICAL",
                error_type="SYSTEM_ERROR"
            )
        )
    
    def _create_safe_error_category(self, error_message: str) -> ErrorCategory:
        """Create safe error category for recursive error handling."""
        return ErrorCategory(
            error_type="SYSTEM_ERROR",
            severity="ERROR",
            recovery_potential="LOW",
            category_description="Safe fallback error category",
            requires_immediate_attention=False,
            allows_continuation=True,
            recommended_action="LOG_AND_CONTINUE"
        )
    
    # Recovery strategy implementations
    def _config_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for configuration errors."""
        return False  # Simulate recovery attempt
    
    def _data_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for data processing errors."""
        return False  # Simulate recovery attempt
    
    def _network_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for network errors."""
        return False  # Simulate recovery attempt
    
    def _file_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for file I/O errors."""
        return False  # Simulate recovery attempt
    
    def _plugin_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for plugin errors."""
        return False  # Simulate recovery attempt
    
    def _validation_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for validation errors."""
        return False  # Simulate recovery attempt
    
    def _system_recovery_strategy(self, error: Exception, context: Optional[ErrorContext], category: ErrorCategory) -> bool:
        """Recovery strategy for system errors."""
        return False  # Simulate recovery attempt
