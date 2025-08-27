from typing import List
from basic_functions.maze import Maze, Object, Persona

def describe_cell(x: int, y: int, z: int, maze: Maze) -> str:
    """Describe entities in the given cell with location info."""

    # Determine which named area covers this location, if any
    area_name = None
    for area in getattr(maze, "areas", []):
        if hasattr(area, "contains") and area.contains(x, y):
            area_name = area.name
            break
        if (
            hasattr(area, "x1")
            and hasattr(area, "x2")
            and hasattr(area, "y1")
            and hasattr(area, "y2")
            and area.x1 <= x <= area.x2
            and area.y1 <= y <= area.y2
        ):
            area_name = area.name
            break

    parts: List[str] = []
    for e in maze.spatial.nearby(x, y, z, radius=0.0):
        if not isinstance(e, (Object, Persona)):
            continue
        desc = e.describe()
        parts.append(desc)

    return "；".join(parts)


def describe_cell_excluding_persona(x: int, y: int, z: int, maze: Maze, exclude_persona: Persona) -> str:
    """Describe entities in the given cell excluding the specified persona."""
    parts: List[str] = []
    for e in maze.spatial.nearby(x, y, z, radius=0.0):
        if not isinstance(e, (Object, Persona)):
            continue
        # Skip the persona making the perception
        if isinstance(e, Persona) and e == exclude_persona:
            continue
        desc = e.describe()
        parts.append(desc)

    return "；".join(parts)


def get_environment_context(x: int, y: int, maze: Maze) -> str:
    """Get environmental context for a location."""
    context_parts = []
    
    # Determine area type
    area_name = None
    for area in getattr(maze, "areas", []):
        if (
            hasattr(area, "x1")
            and hasattr(area, "x2")
            and hasattr(area, "y1")
            and hasattr(area, "y2")
            and area.x1 <= x <= area.x2
            and area.y1 <= y <= area.y2
        ):
            area_name = area.name
            break
    
    if area_name:
        context_parts.append(f"in {area_name}")
    
    # Add environmental hints based on area
    if area_name:
        if "kitchen" in area_name.lower():
            context_parts.append("cooking area")
        elif "forest" in area_name.lower() or "woods" in area_name.lower():
            context_parts.append("natural environment")
        elif "clearing" in area_name.lower():
            context_parts.append("open space")
        elif "corner" in area_name.lower():
            context_parts.append("enclosed area")
    
    return " ".join(context_parts) if context_parts else "general area"


def gather_surrounding_descriptions(
    persona: Persona,
    maze: Maze,
    radius: int = 3  # Increased from 1 to 3 for better perception
) -> List[str]:
    """
    Look at every cell within Chebyshev radius around the persona's position,
    describe each non-empty cell, and return the list of descriptions.
    Enhanced with environmental context and priority-based descriptions.
    """
    cx, cy, cz = int(persona.x), int(persona.y), int(persona.z)
    descriptions: List[str] = []
    nearby_objects = []
    nearby_personas = []
    environmental_context = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if not (
                    0 <= nx < maze.width
                    and 0 <= ny < maze.height
                    and 0 <= nz < maze.depth
                ):
                    continue

                # Get cell description
                desc = describe_cell_excluding_persona(nx, ny, nz, maze, persona)
                if desc:
                    # Categorize what was found
                    if any(keyword in desc.lower() for keyword in ["persona", "bob", "jean"]):
                        nearby_personas.append(desc)
                    else:
                        nearby_objects.append(desc)
                
                # Get environmental context for current location
                if dx == 0 and dy == 0:  # Current location
                    env_context = get_environment_context(nx, ny, maze)
                    if env_context:
                        environmental_context.append(env_context)

    # Prioritize descriptions: environmental context first, then nearby personas, then objects
    if environmental_context:
        descriptions.extend(environmental_context)
    
    if nearby_personas:
        descriptions.extend(nearby_personas)
    
    if nearby_objects:
        descriptions.extend(nearby_objects)

    # Add "myself" to represent the persona's self-perception
    descriptions.append("myself")

    return descriptions


def get_enhanced_perception_summary(persona: Persona, maze: Maze, radius: int = 3) -> str:
    """Get an enhanced perception summary with context and priorities."""
    descriptions = gather_surrounding_descriptions(persona, maze, radius)
    
    if not descriptions or descriptions == ["myself"]:
        return "empty space"
    
    # Filter out "myself" for the summary
    relevant_descriptions = [desc for desc in descriptions if desc != "myself"]
    
    if not relevant_descriptions:
        return "empty space"
    
    # Create a more natural description
    if len(relevant_descriptions) == 1:
        return relevant_descriptions[0]
    elif len(relevant_descriptions) <= 3:
        return " and ".join(relevant_descriptions)
    else:
        # For many objects, group them by type
        objects = []
        areas = []
        personas = []
        
        for desc in relevant_descriptions:
            if any(keyword in desc.lower() for keyword in ["persona", "bob", "jean"]):
                personas.append(desc)
            elif any(keyword in desc.lower() for keyword in ["kitchen", "forest", "woods", "clearing", "corner"]):
                areas.append(desc)
            else:
                objects.append(desc)
        
        parts = []
        if areas:
            parts.append(" in " + " and ".join(areas))
        if personas:
            parts.append(" near " + " and ".join(personas))
        if objects:
            parts.append(" with " + " and ".join(objects[:3]))  # Limit to 3 objects
        
        return "".join(parts) if parts else " and ".join(relevant_descriptions[:3])
