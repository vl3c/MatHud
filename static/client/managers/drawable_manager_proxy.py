"""
MatHud Drawable Manager Proxy System

Resolves circular dependencies during manager initialization using the Proxy Pattern.
Enables specialized managers to access each other through the main DrawableManager without 
creating dependency cycles during object construction.

Design Pattern:
    - Proxy Pattern: Defers attribute resolution until runtime access
    - Dependency Injection: Provides access to DrawableManager functionality
    - Circular Dependency Resolution: Breaks initialization order constraints

Use Cases:
    - SegmentManager accessing PointManager functionality
    - TriangleManager accessing both PointManager and SegmentManager
    - Any manager needing to call methods on other managers

Implementation:
    - __getattr__ magic method forwards all attribute access to real manager
    - Lazy evaluation ensures managers are fully initialized before access
    - Transparent proxy - specialized managers treat it as the real DrawableManager

Architecture Benefits:
    - Clean initialization order independence
    - Maintains strong typing and IDE support
    - No performance overhead after initialization
    - Enables complex inter-manager operations
"""

class DrawableManagerProxy:
    """A proxy for the DrawableManager that forwards all attribute access to the real manager.
    
    This breaks circular dependencies during initialization by deferring attribute resolution
    until runtime access. Specialized managers can access other managers through this proxy
    without creating initialization order constraints.
    """
    
    def __init__(self, real_manager):
        """
        Initialize the proxy with a reference to the real manager.
        
        Args:
            real_manager: The DrawableManager instance this proxy represents
        """
        self._real_manager = real_manager
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the real manager.
        This is called when an attribute doesn't exist on the proxy.
        
        Args:
            name: The name of the attribute to access
            
        Returns:
            The attribute from the real manager
        """
        return getattr(self._real_manager, name) 