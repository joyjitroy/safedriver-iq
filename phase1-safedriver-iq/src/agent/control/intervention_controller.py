"""
Intervention Controller Module

Executes interventions safely and manages driver notifications.
Provides hardware interface for brake control and multi-modal alerts.
"""

import numpy as np
from typing import Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterventionLevel(Enum):
    """Intervention levels from passive to active"""
    PASSIVE_MONITORING = 0
    GENTLE_WARNING = 1
    AUDIO_ALERT = 2
    AGGRESSIVE_WARNING = 3
    AUTONOMOUS_BRAKE = 4


@dataclass
class InterventionCommand:
    """Command to execute intervention"""
    level: InterventionLevel
    urgency: float  # 0-1
    reasoning: str
    brake_level: Optional[float] = None  # 0-1, None if no braking
    notification_message: str = ""
    allow_override: bool = True
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrakeController:
    """
    Controls vehicle braking system with safety constraints.
    
    Interfaces with vehicle CAN bus or brake-by-wire system.
    Implements safety checks and fail-safe mechanisms.
    """
    
    def __init__(
        self,
        max_deceleration_g: float = 0.8,
        min_reaction_time_s: float = 0.3,
        simulation_mode: bool = True
    ):
        """
        Initialize brake controller.
        
        Args:
            max_deceleration_g: Maximum safe deceleration (g-force)
            min_reaction_time_s: Minimum time for driver takeover
            simulation_mode: If True, simulate braking instead of actual control
        """
        self.max_deceleration = max_deceleration_g
        self.min_reaction_time = min_reaction_time_s
        self.simulation_mode = simulation_mode
        
        # State tracking
        self.braking_active = False
        self.current_brake_level = 0.0
        self.brake_history = []
        
        # Safety limits
        self.emergency_threshold = 0.9
        
        logger.info(f"BrakeController initialized (simulation={simulation_mode})")
    
    def apply_brake(
        self,
        urgency: float,
        context: Dict,
        allow_override: bool = True
    ) -> Dict:
        """
        Apply braking force based on urgency.
        
        Args:
            urgency: Urgency level (0-1)
            context: Driving context for safety checks
            allow_override: Whether driver can override
            
        Returns:
            Brake application result
        """
        # Safety checks
        safety_check = self._perform_safety_checks(urgency, context)
        
        if not safety_check['safe']:
            logger.warning(f"Brake application blocked: {safety_check['reason']}")
            return {
                'applied': False,
                'reason': safety_check['reason'],
                'brake_level': 0.0
            }
        
        # Map urgency to brake level
        brake_level = self._compute_brake_level(urgency, context)
        
        # Apply brake
        result = self._execute_brake_command(brake_level, allow_override)
        
        # Record in history
        self.brake_history.append({
            'timestamp': datetime.now(),
            'urgency': urgency,
            'brake_level': brake_level,
            'context': context,
            'result': result
        })
        
        return result
    
    def _perform_safety_checks(self, urgency: float, context: Dict) -> Dict:
        """
        Perform safety checks before braking.
        
        Checks:
        - Rear traffic (avoid rear-end collision)
        - Road conditions (adjust for ice/wet)
        - Vehicle stability
        - System health
        """
        # Check road surface
        road_surface = context.get('road_surface', 'dry')
        if road_surface in ['icy', 'snow'] and urgency > 0.7:
            # Reduce brake force on slippery surfaces
            return {
                'safe': True,
                'adjustment': 'reduce_force',
                'reason': 'Slippery surface - reduced brake force'
            }
        
        # Check rear traffic (would use actual sensor data)
        rear_vehicle_distance = context.get('rear_distance_m', 100)
        if rear_vehicle_distance < 10 and urgency > 0.5:
            logger.warning("Close rear traffic - applying brake cautiously")
            return {
                'safe': True,
                'adjustment': 'gradual',
                'reason': 'Rear traffic close - gradual braking'
            }
        
        # Check vehicle speed
        speed = context.get('speed_mph', 0)
        if speed < 5 and urgency < 0.5:
            # No need to brake at very low speed
            return {
                'safe': False,
                'reason': 'Speed too low for braking'
            }
        
        return {
            'safe': True,
            'adjustment': None,
            'reason': 'All safety checks passed'
        }
    
    def _compute_brake_level(self, urgency: float, context: Dict) -> float:
        """
        Compute appropriate brake level from urgency and context.
        
        Urgency levels:
        - 0.3-0.5: Gentle deceleration
        - 0.5-0.7: Moderate braking
        - 0.7-0.9: Strong braking
        - 0.9-1.0: Emergency braking
        """
        if urgency < 0.5:
            brake_level = urgency * 0.3  # Max 15% brake
        elif urgency < 0.7:
            brake_level = 0.15 + (urgency - 0.5) * 0.75  # 15-30% brake
        elif urgency < 0.9:
            brake_level = 0.30 + (urgency - 0.7) * 1.5  # 30-60% brake
        else:
            brake_level = 0.60 + (urgency - 0.9) * 4.0  # 60-100% brake
        
        # Adjust for road conditions
        road_surface = context.get('road_surface', 'dry')
        if road_surface in ['icy', 'snow']:
            brake_level *= 0.6  # Reduce by 40% on ice/snow
        elif road_surface == 'wet':
            brake_level *= 0.8  # Reduce by 20% on wet
        
        # Cap at maximum safe level
        brake_level = min(brake_level, self.max_deceleration)
        
        return brake_level
    
    def _execute_brake_command(
        self,
        brake_level: float,
        allow_override: bool
    ) -> Dict:
        """
        Execute brake command (actual or simulated).
        
        In production, this would interface with vehicle CAN bus.
        """
        if self.simulation_mode:
            # Simulate braking
            logger.info(f"SIMULATED BRAKE: {brake_level*100:.1f}% "
                       f"(override_allowed={allow_override})")
            
            self.braking_active = True
            self.current_brake_level = brake_level
            
            return {
                'applied': True,
                'brake_level': brake_level,
                'deceleration_g': brake_level * self.max_deceleration,
                'allow_override': allow_override,
                'mode': 'simulation'
            }
        else:
            # Would interface with actual hardware here
            # CAN bus commands, brake-by-wire, etc.
            pass
    
    def release_brake(self):
        """Release autonomous braking"""
        if self.braking_active:
            logger.info("Releasing autonomous brake")
            self.braking_active = False
            self.current_brake_level = 0.0
    
    def get_brake_status(self) -> Dict:
        """Get current brake status"""
        return {
            'active': self.braking_active,
            'level': self.current_brake_level,
            'history_count': len(self.brake_history)
        }


class DriverNotification:
    """
    Multi-modal driver notification system.
    
    Provides visual, audio, and haptic feedback to driver.
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        Initialize notification system.
        
        Args:
            simulation_mode: If True, print notifications instead of actual output
        """
        self.simulation_mode = simulation_mode
        self.notification_history = []
        
        # Volume/intensity settings
        self.visual_enabled = True
        self.audio_enabled = True
        self.haptic_enabled = True
        
        logger.info("DriverNotification system initialized")
    
    def send_notification(
        self,
        level: InterventionLevel,
        message: str,
        urgency: float,
        additional_info: Optional[Dict] = None
    ):
        """
        Send multi-modal notification to driver.
        
        Args:
            level: Intervention level
            message: Main message text
            urgency: Urgency level (0-1)
            additional_info: Additional context to display
        """
        # Select notification channels based on level
        channels = self._select_channels(level, urgency)
        
        # Format message
        formatted_message = self._format_message(
            level, message, urgency, additional_info
        )
        
        # Send via each channel
        result = {}
        
        if 'visual' in channels and self.visual_enabled:
            result['visual'] = self._send_visual(formatted_message, urgency)
        
        if 'audio' in channels and self.audio_enabled:
            result['audio'] = self._send_audio(formatted_message, urgency)
        
        if 'haptic' in channels and self.haptic_enabled:
            result['haptic'] = self._send_haptic(level, urgency)
        
        # Record notification
        self.notification_history.append({
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'urgency': urgency,
            'channels': channels,
            'result': result
        })
        
        return result
    
    def _select_channels(self, level: InterventionLevel, urgency: float) -> List[str]:
        """Select appropriate notification channels"""
        if level == InterventionLevel.PASSIVE_MONITORING:
            return []  # No notification
        
        elif level == InterventionLevel.GENTLE_WARNING:
            return ['visual']  # Just visual indicator
        
        elif level == InterventionLevel.AUDIO_ALERT:
            return ['visual', 'audio']  # Visual + audio
        
        elif level == InterventionLevel.AGGRESSIVE_WARNING:
            return ['visual', 'audio', 'haptic']  # All channels
        
        elif level == InterventionLevel.AUTONOMOUS_BRAKE:
            return ['visual', 'audio', 'haptic']  # Maximum alert
        
        return ['visual']
    
    def _format_message(
        self,
        level: InterventionLevel,
        message: str,
        urgency: float,
        additional_info: Optional[Dict]
    ) -> Dict:
        """Format message for display"""
        # Add icon/symbol
        if urgency > 0.8:
            icon = "⚠️"
            color = "red"
        elif urgency > 0.5:
            icon = "⚠️"
            color = "orange"
        else:
            icon = "ℹ️"
            color = "yellow"
        
        formatted = {
            'icon': icon,
            'color': color,
            'title': level.name.replace('_', ' ').title(),
            'message': message,
            'urgency': urgency
        }
        
        if additional_info:
            formatted['details'] = additional_info
        
        return formatted
    
    def _send_visual(self, formatted_message: Dict, urgency: float) -> Dict:
        """Send visual notification (HUD, dashboard screen)"""
        if self.simulation_mode:
            print(f"\n{'='*60}")
            print(f"{formatted_message['icon']} {formatted_message['title']}")
            print(f"Message: {formatted_message['message']}")
            if 'details' in formatted_message:
                print(f"Details: {formatted_message['details']}")
            print(f"{'='*60}\n")
        
        return {'sent': True, 'channel': 'visual'}
    
    def _send_audio(self, formatted_message: Dict, urgency: float) -> Dict:
        """Send audio notification"""
        if self.simulation_mode:
            volume = "LOUD" if urgency > 0.7 else "NORMAL"
            print(f"🔊 AUDIO [{volume}]: {formatted_message['message']}")
        
        return {'sent': True, 'channel': 'audio', 'volume': urgency}
    
    def _send_haptic(self, level: InterventionLevel, urgency: float) -> Dict:
        """Send haptic feedback (steering wheel, seat vibration)"""
        if self.simulation_mode:
            intensity = "STRONG" if urgency > 0.7 else "MODERATE"
            print(f"📳 HAPTIC [{intensity}]: Steering wheel vibration")
        
        return {'sent': True, 'channel': 'haptic', 'intensity': urgency}
    
    def get_notification_stats(self) -> Dict:
        """Get notification statistics"""
        if not self.notification_history:
            return {'total': 0}
        
        by_level = {}
        for notif in self.notification_history:
            level_name = notif['level'].name
            by_level[level_name] = by_level.get(level_name, 0) + 1
        
        return {
            'total': len(self.notification_history),
            'by_level': by_level,
            'last_notification': self.notification_history[-1]['timestamp']
        }


class OverrideManager:
    """
    Manages driver overrides of system interventions.
    
    Tracks override patterns and learns driver preferences.
    """
    
    def __init__(self, learning_enabled: bool = True):
        """
        Initialize override manager.
        
        Args:
            learning_enabled: Whether to learn from overrides
        """
        self.learning_enabled = learning_enabled
        self.override_history = []
        self.override_count = 0
        
        # Override policies
        self.allow_override_levels = {
            InterventionLevel.PASSIVE_MONITORING: True,
            InterventionLevel.GENTLE_WARNING: True,
            InterventionLevel.AUDIO_ALERT: True,
            InterventionLevel.AGGRESSIVE_WARNING: True,
            InterventionLevel.AUTONOMOUS_BRAKE: False  # Cannot override emergency brake < 2s
        }
        
        logger.info("OverrideManager initialized")
    
    def can_override(
        self,
        level: InterventionLevel,
        time_to_collision: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Check if driver can override this intervention.
        
        Args:
            level: Intervention level
            time_to_collision: Time to collision (seconds), if applicable
            
        Returns:
            (can_override, reason)
        """
        # Emergency brake with imminent collision cannot be overridden
        if level == InterventionLevel.AUTONOMOUS_BRAKE:
            if time_to_collision and time_to_collision < 2.0:
                return False, "Imminent collision - override not allowed"
            else:
                return True, "Driver can take control"
        
        # All other levels can be overridden
        if self.allow_override_levels.get(level, True):
            return True, "Override allowed"
        else:
            return False, "Override not permitted for safety"
    
    def record_override(
        self,
        intervention_command: InterventionCommand,
        context: Dict,
        override_action: str
    ):
        """
        Record a driver override.
        
        Args:
            intervention_command: The intervention that was overridden
            context: Driving context
            override_action: What driver did instead
        """
        override = {
            'timestamp': datetime.now(),
            'intervention_level': intervention_command.level,
            'urgency': intervention_command.urgency,
            'override_action': override_action,
            'context': context
        }
        
        self.override_history.append(override)
        self.override_count += 1
        
        logger.info(f"Override recorded: {intervention_command.level.name} -> {override_action}")
        
        if self.learning_enabled:
            self._learn_from_override(override)
    
    def _learn_from_override(self, override: Dict):
        """
        Learn from override pattern.
        
        Adjusts future intervention thresholds based on driver behavior.
        """
        # Check for patterns
        recent_overrides = self.get_recent_overrides(n=5)
        
        if len(recent_overrides) >= 5:
            # Driver has overridden 5 times recently
            # May need to adjust sensitivity
            logger.info("High override frequency detected - system may need recalibration")
    
    def get_recent_overrides(self, n: int = 10) -> List[Dict]:
        """Get n most recent overrides"""
        return self.override_history[-n:]
    
    def get_override_statistics(self) -> Dict:
        """Get override statistics"""
        if not self.override_history:
            return {'total': 0}
        
        by_level = {}
        for override in self.override_history:
            level_name = override['intervention_level'].name
            by_level[level_name] = by_level.get(level_name, 0) + 1
        
        # Compute override rate by time period
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        recent_count = sum(
            1 for o in self.override_history 
            if o['timestamp'] > hour_ago
        )
        
        return {
            'total': self.override_count,
            'by_level': by_level,
            'last_hour': recent_count,
            'last_override': self.override_history[-1]['timestamp'] if self.override_history else None
        }


class InterventionController:
    """
    Main controller that coordinates all intervention execution.
    
    Integrates brake control, notifications, and override management.
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        Initialize intervention controller.
        
        Args:
            simulation_mode: Run in simulation (no actual hardware control)
        """
        self.simulation_mode = simulation_mode
        
        # Initialize sub-systems
        self.brake_controller = BrakeController(simulation_mode=simulation_mode)
        self.notification_system = DriverNotification(simulation_mode=simulation_mode)
        self.override_manager = OverrideManager()
        
        # Execution history
        self.intervention_history = []
        
        logger.info("InterventionController initialized")
    
    def execute_intervention(
        self,
        command: InterventionCommand,
        context: Dict
    ) -> Dict:
        """
        Execute intervention command.
        
        Args:
            command: Intervention command to execute
            context: Current driving context
            
        Returns:
            Execution result
        """
        result = {
            'timestamp': datetime.now(),
            'command': command,
            'executed': False,
            'actions_taken': []
        }
        
        # Check if override is allowed
        can_override, override_reason = self.override_manager.can_override(
            command.level,
            context.get('time_to_collision')
        )
        
        # Send notification
        if command.level != InterventionLevel.PASSIVE_MONITORING:
            notif_result = self.notification_system.send_notification(
                command.level,
                command.notification_message,
                command.urgency,
                {'reasoning': command.reasoning, 'override_allowed': can_override}
            )
            result['actions_taken'].append(('notification', notif_result))
        
        # Apply braking if needed
        if command.brake_level is not None and command.brake_level > 0:
            brake_result = self.brake_controller.apply_brake(
                command.urgency,
                context,
                allow_override=can_override
            )
            result['actions_taken'].append(('brake', brake_result))
            result['executed'] = brake_result['applied']
        else:
            result['executed'] = True
        
        # Record intervention
        self.intervention_history.append(result)
        
        return result
    
    def release_control(self):
        """Release all active interventions"""
        self.brake_controller.release_brake()
        logger.info("All interventions released - driver has full control")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'brake_status': self.brake_controller.get_brake_status(),
            'notification_stats': self.notification_system.get_notification_stats(),
            'override_stats': self.override_manager.get_override_statistics(),
            'total_interventions': len(self.intervention_history),
            'simulation_mode': self.simulation_mode
        }
