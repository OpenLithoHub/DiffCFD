use numpy::{PyReadonlyArray1, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute B-spline airfoil SDF using parallel rayon.
///
/// For each grid point, computes:
/// 1. Minimum distance to polygon edges
/// 2. Inside/outside via winding number (ray casting)
///
/// Returns SDF array (flattened, ny*nx), positive outside, negative inside.
#[pyfunction]
pub fn bspline_sdf<'py>(
    py: Python<'py>,
    grid_x: PyReadonlyArray1<'py, f64>,
    grid_y: PyReadonlyArray1<'py, f64>,
    control_points_x: PyReadonlyArray1<'py, f64>,
    control_points_y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let gx = grid_x.as_slice()?;
    let gy = grid_y.as_slice()?;
    let cpx = control_points_x.as_slice()?;
    let cpy = control_points_y.as_slice()?;
    let n_pts = cpx.len();

    let results: Vec<f64> = (0..gx.len())
        .into_par_iter()
        .map(|idx| {
            let px = gx[idx];
            let py = gy[idx];
            let mut min_dist_sq = f64::MAX;

            // Minimum distance to each edge
            for i in 0..n_pts {
                let j = (i + 1) % n_pts;
                let ax = cpx[i]; let ay = cpy[i];
                let bx = cpx[j]; let by = cpy[j];
                let abx = bx - ax; let aby = by - ay;
                let apx = px - ax; let apy = py - ay;
                let t = ((apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-30))
                    .clamp(0.0, 1.0);
                let cx = ax + t * abx;
                let cy = ay + t * aby;
                let dsq = (px - cx) * (px - cx) + (py - cy) * (py - cy);
                if dsq < min_dist_sq {
                    min_dist_sq = dsq;
                }
            }

            let dist = min_dist_sq.sqrt();

            // Winding number (ray casting)
            let mut inside = false;
            for i in 0..n_pts {
                let j = (i + 1) % n_pts;
                let yi = cpy[i]; let yj = cpy[j];
                let xi = cpx[i]; let xj = cpx[j];

                if (yi > py) != (yj > py) {
                    let x_intersect = (py - yi) / (yj - yi + 1e-30) * (xj - xi) + xi;
                    if px < x_intersect {
                        inside = !inside;
                    }
                }
            }

            if inside { -dist } else { dist }
        })
        .collect();

    Ok(PyArray1::from_vec(py, results).unbind())
}
