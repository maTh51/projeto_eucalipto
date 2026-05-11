"""Process RayExtract tree structure files (.txt) to extract DBH and volume.

RayExtract outputs cylindrical tree structures in .txt format with columns:
x, y, z, radius, parent_id, section_id

This module provides functions to:
- Parse the tree structure files
- Extract DBH (diameter at breast height, z=1.3m)
- Calculate volume from cylinder sum
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def read_rayextract_tree_file(filepath: str) -> pd.DataFrame:
    """Read RayExtract tree structure file (.txt format).
    
    RayExtract format has multiple cylinders per line, comma-separated.
    Each cylinder: x,y,z,radius,parent_id,section_id
    
    Parameters
    ----------
    filepath : str
        Path to the RayExtract .txt file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, z, radius, parent_id, section_id
    """
    filepath = Path(filepath)
    
    cylinders = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and header
            if not line or line.startswith('#'):
                continue
            
            # Parse comma-separated values, each 6 values = 1 cylinder
            values = [v.strip() for v in line.split(',') if v.strip()]
            
            # Process groups of 6 values
            for i in range(0, len(values), 6):
                if i + 6 <= len(values):
                    try:
                        cyl = {
                            'x': float(values[i]),
                            'y': float(values[i + 1]),
                            'z': float(values[i + 2]),
                            'radius': float(values[i + 3]),
                            'parent_id': int(values[i + 4]),
                            'section_id': int(values[i + 5])
                        }
                        cylinders.append(cyl)
                    except (ValueError, IndexError):
                        continue
    
    df = pd.DataFrame(cylinders)
    
    if df.empty:
        raise ValueError(f"No cylinder data found in {filepath}")
    
    return df


def extract_tree_ids(df: pd.DataFrame) -> List[int]:
    """Extract unique tree IDs from structure file.
    
    In RayExtract format, trees are identified by distinct connected components.
    Root nodes have parent_id = -1.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from read_rayextract_tree_file()
        
    Returns
    -------
    List[int]
        Sorted list of unique tree IDs (or root parent_ids)
    """
    # Find root nodes (parent_id == -1)
    root_nodes = df[df['parent_id'] == -1]
    
    # Group by section to identify trees
    # Assuming each connected component is a separate tree
    tree_ids = sorted(root_nodes.index.tolist())
    
    return tree_ids


def calculate_dbh_from_structure(df: pd.DataFrame, 
                                 tree_root_idx: int,
                                 dbh_height: float = 1.3) -> Optional[float]:
    """Calculate DBH from tree structure by finding cylinder at breast height.
    
    Finds the cylinder segment closest to z = dbh_height and returns
    diameter = 2 * radius.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full tree structure DataFrame
    tree_root_idx : int
        Index of the root node of the tree
    dbh_height : float
        Height at which to measure DBH (default 1.3m = breast height)
        
    Returns
    -------
    float or None
        DBH in cm, or None if calculation fails
    """
    try:
        # Get all segments of this tree by traversing from root
        tree_segments = _get_tree_segments_from_root(df, tree_root_idx)
        
        if tree_segments.empty:
            return None
        
        # Find segment closest to dbh_height
        tree_segments = tree_segments.copy()
        tree_segments['z_dist'] = np.abs(tree_segments['z'] - dbh_height)
        closest_segment = tree_segments.loc[tree_segments['z_dist'].idxmin()]
        
        # DBH = 2 * radius (convert m to cm)
        dbh_cm = 2 * closest_segment['radius'] * 100
        
        return float(dbh_cm)
    
    except Exception as e:
        print(f"Error calculating DBH: {e}")
        return None


def calculate_volume_from_structure(df: pd.DataFrame,
                                    tree_root_idx: int,
                                    wood_density_kg_m3: float = 600.0) -> Tuple[float, float, float]:
    """Calculate volume and mass from tree cylinder structure.
    
    Sums volumes of all cylinders: V = π * r² * h
    
    Parameters
    ----------
    df : pd.DataFrame
        Full tree structure DataFrame
    tree_root_idx : int
        Index of the root node of the tree
    wood_density_kg_m3 : float
        Wood density for mass calculation (default 600 kg/m³)
        
    Returns
    -------
    Tuple[float, float, float]
        (volume_m3, mass_kg, height_m)
    """
    try:
        # Get all segments of this tree
        tree_segments = _get_tree_segments_from_root(df, tree_root_idx)
        
        if tree_segments.empty:
            return 0.0, 0.0, 0.0
        
        # Calculate volume for each cylinder
        # Approximate height as distance to parent
        tree_segments = tree_segments.copy()
        tree_segments['height'] = 0.0
        
        # For each segment, find height to next segment
        for idx, row in tree_segments.iterrows():
            child_segments = tree_segments[tree_segments['parent_id'] == idx]
            if not child_segments.empty:
                # Height to first child (or average)
                heights_to_children = np.linalg.norm(
                    child_segments[['x', 'y', 'z']].values - row[['x', 'y', 'z']].values,
                    axis=1
                )
                tree_segments.loc[idx, 'height'] = heights_to_children.mean()
            else:
                # Leaf segment - estimate height from previous segment
                parent = tree_segments[tree_segments.index == row['parent_id']]
                if not parent.empty:
                    height_to_parent = np.linalg.norm(
                        row[['x', 'y', 'z']].values - parent.iloc[0][['x', 'y', 'z']].values
                    )
                    tree_segments.loc[idx, 'height'] = height_to_parent
        
        # Volume = π * r² * h (radius in meters)
        tree_segments['volume'] = np.pi * (tree_segments['radius'] ** 2) * tree_segments['height']
        
        volume_m3 = float(tree_segments['volume'].sum())
        mass_kg = float(volume_m3 * wood_density_kg_m3)
        height_m = float(tree_segments['z'].max() - tree_segments['z'].min())
        
        return volume_m3, mass_kg, height_m
    
    except Exception as e:
        print(f"Error calculating volume: {e}")
        return 0.0, 0.0, 0.0


def _get_tree_segments_from_root(df: pd.DataFrame, root_idx: int) -> pd.DataFrame:
    """Get all segments belonging to a tree from its root.
    
    Traverses the tree structure using parent_id relationships.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full tree structure DataFrame
    root_idx : int
        Index of the root node
        
    Returns
    -------
    pd.DataFrame
        All segments belonging to this tree
    """
    visited = set()
    to_visit = [root_idx]
    segments_indices = []
    
    while to_visit:
        idx = to_visit.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        segments_indices.append(idx)
        
        # Find children
        children = df[df['parent_id'] == idx].index.tolist()
        to_visit.extend(children)
    
    return df.loc[segments_indices]


def process_rayextract_file(filepath: str,
                           wood_density_kg_m3: float = 600.0) -> pd.DataFrame:
    """Process a complete RayExtract tree file and extract metrics for all trees.
    
    Parameters
    ----------
    filepath : str
        Path to RayExtract .txt file
    wood_density_kg_m3 : float
        Wood density for mass calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tree_id, dbh_cm, volume_m3, mass_kg, height_m
    """
    df = read_rayextract_tree_file(filepath)
    
    # Find root nodes (starting points of each tree)
    root_nodes = df[df['parent_id'] == -1].index.tolist()
    
    results = []
    
    for tree_idx, root_idx in enumerate(root_nodes):
        dbh_cm = calculate_dbh_from_structure(df, root_idx)
        volume_m3, mass_kg, height_m = calculate_volume_from_structure(
            df, root_idx, wood_density_kg_m3
        )
        
        results.append({
            'tree_id': tree_idx,
            'dbh_cm': dbh_cm,
            'volume_m3': volume_m3,
            'mass_kg': mass_kg,
            'height_m': height_m
        })
    
    return pd.DataFrame(results)
