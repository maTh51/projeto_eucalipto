import os
import sys
import glob
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path so we can import eucalipto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eucalipto.io import load_cloud_auto
from eucalipto.dbh_methods import estimate_dbh
from eucalipto.volume_methods import estimate_volume

app = FastAPI(title="Eucalipto Biometrics Tuning Dashboard")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for loaded point cloud
class CloudCache:
    filepath: Optional[str] = None
    points: Optional[np.ndarray] = None
    extras: Optional[Dict[str, np.ndarray]] = None
    tree_ids: Optional[np.ndarray] = None
    trunk_labels: Optional[np.ndarray] = None
    tree_id_col: Optional[str] = None
    trunk_col: Optional[str] = None
    unique_tree_ids: List[int] = []

cache = CloudCache()

class LoadRequest(BaseModel):
    filepath: str

class EstimateRequest(BaseModel):
    tree_id: int
    dbh_method: str
    dbh_params: Dict[str, Any]
    volume_method: str
    volume_params: Dict[str, Any]
    wood_density_kg_m3: float = 900.0

class EstimateAllRequest(BaseModel):
    dbh_method: str
    dbh_params: Dict[str, Any]
    volume_method: str
    volume_params: Dict[str, Any]
    wood_density_kg_m3: float = 900.0

@app.get("/api/files")
def list_files():
    # Scan the workspace for ply, las, laz files
    search_dir = str(PROJECT_ROOT)
    extensions = ["*.las", "*.laz", "*.ply"]
    files = []
    
    # Check default project directories
    for ext in extensions:
        # Recursive search up to depth 3 to find relevant point clouds
        for p in glob.glob(os.path.join(search_dir, "**", ext), recursive=True):
            if ".gemini" in p or "scratch" in p or "third_party" in p:
                continue
            # Store relative path for convenience
            rel_path = os.path.relpath(p, search_dir)
            files.append({
                "name": os.path.basename(p),
                "path": rel_path,
                "abs_path": p,
                "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2)
            })
    return sorted(files, key=lambda x: x["path"])

def _prepare_loaded_cloud(filepath: str, points: np.ndarray, extras: Dict[str, np.ndarray], format_label: str) -> dict:
    # Save plot centering offsets
    cache.center_x = float(np.median(points[:, 0]))
    cache.center_y = float(np.median(points[:, 1]))
    cache.min_z = float(points[:, 2].min())

    # Detect tree ID column
    tree_id_col = None
    tree_id_candidates = ['tree_id', 'instance_pred', 'treeID', 'final_segs']
    for candidate in tree_id_candidates:
        if candidate in extras:
            tree_id_col = candidate
            break

    # Detect trunk/leaf column
    trunk_col = None
    trunk_candidates = ['trunk_leaf_label', 'semantic_seg', 'semantic_pred', 'leafwood_pred']
    for candidate in trunk_candidates:
        if candidate in extras:
            trunk_col = candidate
            break

    cache.filepath = filepath
    cache.points = points
    cache.extras = extras
    cache.tree_id_col = tree_id_col
    cache.trunk_col = trunk_col
    
    if tree_id_col is not None:
        tree_ids = extras[tree_id_col].astype(int)
        cache.tree_ids = tree_ids
        unique_ids = np.unique(tree_ids)
        cache.unique_tree_ids = sorted([int(x) for x in unique_ids if x >= 0])
    else:
        cache.tree_ids = None
        cache.unique_tree_ids = []

    if trunk_col is not None:
        cache.trunk_labels = extras[trunk_col].astype(int)
    else:
        cache.trunk_labels = None

    # Downsample entire plot point cloud for 3D layout display (max 80,000 points)
    max_display_points = 80000
    if points.shape[0] > max_display_points:
        step = int(np.ceil(points.shape[0] / max_display_points))
        display_pts = points[::step]
        
        if "red" in extras and "green" in extras and "blue" in extras:
            r = extras["red"][::step]
            g = extras["green"][::step]
            b = extras["blue"][::step]
            display_colors = np.vstack((r, g, b)).T.astype(int).tolist()
        else:
            display_colors = None
            
        if cache.trunk_labels is not None:
            display_trunk = (cache.trunk_labels[::step] == 1).astype(int).tolist()
        else:
            display_trunk = None
    else:
        display_pts = points
        if "red" in extras and "green" in extras and "blue" in extras:
            r = extras["red"]
            g = extras["green"]
            b = extras["blue"]
            display_colors = np.vstack((r, g, b)).T.astype(int).tolist()
        else:
            display_colors = None
            
        if cache.trunk_labels is not None:
            display_trunk = (cache.trunk_labels == 1).astype(int).tolist()
        else:
            display_trunk = None

    display_pts_centered = display_pts.copy()
    display_pts_centered[:, 0] -= cache.center_x
    display_pts_centered[:, 1] -= cache.center_y
    display_pts_centered[:, 2] -= cache.min_z

    return {
        "status": "success",
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "format": format_label,
        "num_points": points.shape[0],
        "tree_id_column": tree_id_col,
        "trunk_column": trunk_col,
        "tree_count": len(cache.unique_tree_ids),
        "tree_ids": cache.unique_tree_ids[:200],
        "plot_points": display_pts_centered.tolist(),
        "plot_colors": display_colors,
        "plot_is_trunk": display_trunk
    }

@app.post("/api/load")
def load_file(req: LoadRequest):
    abs_path = os.path.join(str(PROJECT_ROOT), req.filepath)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.filepath}")
    
    try:
        points, extras, format_label = load_cloud_auto(abs_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")
        
    if points.size == 0:
        raise HTTPException(status_code=400, detail="Point cloud is empty")

    rel_path = os.path.relpath(str(abs_path), str(PROJECT_ROOT))
    return _prepare_loaded_cloud(rel_path, points, extras, format_label)

@app.get("/api/tree/{tree_id}")
def get_tree(tree_id: int):
    if cache.points is None:
        raise HTTPException(status_code=400, detail="No point cloud loaded")
    if cache.tree_ids is None:
        raise HTTPException(status_code=400, detail="Loaded cloud does not contain segment IDs")
    
    mask = cache.tree_ids == tree_id
    if not np.any(mask):
        raise HTTPException(status_code=404, detail=f"Tree ID {tree_id} not found")
        
    tree_points = cache.points[mask]
    
    # Find bounding box and center points to prevent WebGL floating point precision jitter
    min_x, min_y, min_z = tree_points.min(axis=0)
    max_x, max_y, max_z = tree_points.max(axis=0)
    center_x = float(np.median(tree_points[:, 0]))
    center_y = float(np.median(tree_points[:, 1]))
    base_z = float(min_z)

    # Get trunk labels if available
    is_trunk = []
    if cache.trunk_labels is not None:
        is_trunk = (cache.trunk_labels[mask] == 1).astype(int).tolist()
    else:
        # Default fallback: everything is trunk
        is_trunk = [1] * tree_points.shape[0]

    # Convert coordinates to list centered at (center_x, center_y, base_z)
    pts_centered = tree_points.copy()
    pts_centered[:, 0] -= center_x
    pts_centered[:, 1] -= center_y
    pts_centered[:, 2] -= base_z

    # Use standard RGB colors if available
    colors = []
    if "red" in cache.extras and "green" in cache.extras and "blue" in cache.extras:
        r = cache.extras["red"][mask]
        g = cache.extras["green"][mask]
        b = cache.extras["blue"][mask]
        colors = np.vstack((r, g, b)).T.astype(int).tolist()
    else:
        # Default color based on height
        z_rel = pts_centered[:, 2]
        z_norm = (z_rel - z_rel.min()) / max(1.0, z_rel.max() - z_rel.min())
        # Teal to green gradient
        colors = [[int(30), int(100 + 155 * z), int(150)] for z in z_norm]

    return {
        "tree_id": tree_id,
        "num_points": tree_points.shape[0],
        "center": [center_x, center_y, base_z],
        "bounds": {
            "min": [float(min_x), float(min_y), float(min_z)],
            "max": [float(max_x), float(max_y), float(max_z)],
            "height": float(max_z - min_z)
        },
        "points": pts_centered.tolist(),
        "is_trunk": is_trunk,
        "colors": colors
    }

@app.post("/api/estimate")
def run_estimation(req: EstimateRequest):
    if cache.points is None:
        raise HTTPException(status_code=400, detail="No point cloud loaded")
    if cache.tree_ids is None:
        raise HTTPException(status_code=400, detail="Loaded cloud does not contain segment IDs")

    mask = cache.tree_ids == req.tree_id
    if not np.any(mask):
        raise HTTPException(status_code=404, detail=f"Tree ID {req.tree_id} not found")

    tree_points = cache.points[mask]
    center_x = float(np.median(tree_points[:, 0]))
    center_y = float(np.median(tree_points[:, 1]))
    base_z = float(tree_points[:, 2].min())

    # Separate trunk points for fitting
    if cache.trunk_labels is not None:
        trunk_mask = cache.trunk_labels[mask] == 1
        trunk_points = tree_points[trunk_mask]
    else:
        trunk_points = tree_points

    if trunk_points.shape[0] < 5:
        raise HTTPException(status_code=400, detail="Insufficient trunk points for calculations")

    # 1. Estimate DBH
    try:
        dbh_cm, dbh_info = estimate_dbh(
            trunk_points,
            method=req.dbh_method,
            **req.dbh_params
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DBH error: {str(e)}")

    # Calculate trunk height
    z = tree_points[:, 2]
    height_m = float(z.max() - z.min())

    # 2. Estimate Volume
    try:
        volume_info = estimate_volume(
            trunk_points,
            dbh_cm=dbh_cm,
            height_m=height_m,
            method=req.volume_method,
            generate_mesh=True,
            wood_density_kg_m3=req.wood_density_kg_m3,
            **req.volume_params
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volume error: {str(e)}")

    # Prepare response
    res = {
        "dbh_cm": float(dbh_cm) if dbh_cm is not None else None,
        "height_m": height_m,
        "volume_m3": float(volume_info.get("volume_m3")) if volume_info.get("volume_m3") is not None else None,
        "mass_kg": float(volume_info.get("mass_kg")) if volume_info.get("mass_kg") is not None else None,
        "details": {k: v for k, v in volume_info.items() if k not in ["mesh", "volume_m3", "mass_kg"]},
        "dbh_details": {k: v for k, v in dbh_info.items() if k != "method"}
    }

    # Center and package the 3D mesh
    mesh_data = volume_info.get("mesh")
    if mesh_data is not None:
        v = mesh_data["vertices"].copy()
        # Center mesh coordinates to align with the tree points
        v[:, 0] -= center_x
        v[:, 1] -= center_y
        v[:, 2] -= base_z
        
        res["mesh"] = {
            "vertices": v.tolist(),
            "faces": mesh_data["faces"].tolist(),
            "colors": mesh_data["colors"].tolist() if "colors" in mesh_data else None
        }

    return res

# Make sure uploads dir exists
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

@app.post("/api/upload")
async def upload_point_cloud(file: UploadFile = File(...)):
    filename = file.filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    save_path = UPLOADS_DIR / safe_filename
    
    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
    rel_path = os.path.relpath(str(save_path), str(PROJECT_ROOT))
    
    try:
        points, extras, format_label = load_cloud_auto(str(save_path))
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail=f"Error loading uploaded cloud: {str(e)}")
        
    if points.size == 0:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Point cloud is empty")

    return _prepare_loaded_cloud(rel_path, points, extras, format_label)

@app.post("/api/estimate-all")
def run_batch_estimation(req: EstimateAllRequest):
    if cache.points is None:
        raise HTTPException(status_code=400, detail="No point cloud loaded")
    if cache.tree_ids is None:
        raise HTTPException(status_code=400, detail="Loaded cloud does not contain segment IDs")

    results = []
    for tree_id in cache.unique_tree_ids:
        mask = cache.tree_ids == tree_id
        tree_points = cache.points[mask]
        
        if cache.trunk_labels is not None:
            trunk_mask = cache.trunk_labels[mask] == 1
            trunk_points = tree_points[trunk_mask]
        else:
            trunk_points = tree_points
            
        z = tree_points[:, 2]
        height_m = float(z.max() - z.min())
        
        if trunk_points.shape[0] < 5:
            results.append({
                "tree_id": tree_id,
                "dbh_cm": None,
                "height_m": height_m,
                "volume_m3": None,
                "mass_kg": None,
                "status": "failed",
                "error": "Insufficient trunk points"
            })
            continue
            
        try:
            dbh_cm, _ = estimate_dbh(
                trunk_points,
                method=req.dbh_method,
                **req.dbh_params
            )
            
            volume_info = estimate_volume(
                trunk_points,
                dbh_cm=dbh_cm,
                height_m=height_m,
                method=req.volume_method,
                generate_mesh=True,
                wood_density_kg_m3=req.wood_density_kg_m3,
                **req.volume_params
            )
            
            # Center and package the 3D mesh using plot offsets
            mesh_data = volume_info.get("mesh")
            mesh_dict = None
            if mesh_data is not None:
                v = mesh_data["vertices"].copy()
                v[:, 0] -= cache.center_x
                v[:, 1] -= cache.center_y
                v[:, 2] -= cache.min_z
                mesh_dict = {
                    "vertices": v.tolist(),
                    "faces": mesh_data["faces"].tolist(),
                    "colors": mesh_data["colors"].tolist() if "colors" in mesh_data else None
                }
            
            results.append({
                "tree_id": tree_id,
                "dbh_cm": float(dbh_cm) if dbh_cm is not None else None,
                "height_m": height_m,
                "volume_m3": float(volume_info.get("volume_m3")) if volume_info.get("volume_m3") is not None else None,
                "mass_kg": float(volume_info.get("mass_kg")) if volume_info.get("mass_kg") is not None else None,
                "status": "success",
                "mesh": mesh_dict
            })
        except Exception as e:
            results.append({
                "tree_id": tree_id,
                "dbh_cm": None,
                "height_m": height_m,
                "volume_m3": None,
                "mass_kg": None,
                "status": "failed",
                "error": str(e)
            })

    valid_dbhs = [r["dbh_cm"] for r in results if r["status"] == "success" and r["dbh_cm"] is not None]
    valid_heights = [r["height_m"] for r in results if r["status"] == "success" and r["height_m"] is not None]
    valid_volumes = [r["volume_m3"] for r in results if r["status"] == "success" and r["volume_m3"] is not None]
    valid_masses = [r["mass_kg"] for r in results if r["status"] == "success" and r["mass_kg"] is not None]
    
    summary = {
        "tree_count": len(results),
        "successful_count": sum(1 for r in results if r["status"] == "success"),
        "total_volume_m3": float(sum(valid_volumes)) if valid_volumes else 0.0,
        "total_mass_kg": float(sum(valid_masses)) if valid_masses else 0.0,
        "mean_dbh_cm": float(np.mean(valid_dbhs)) if valid_dbhs else 0.0,
        "mean_height_m": float(np.mean(valid_heights)) if valid_heights else 0.0,
    }

    return {
        "summary": summary,
        "results": results
    }

# Serve UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
