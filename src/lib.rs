use pyo3::prelude::*;

mod momentum;
mod pressure;
mod sdf;
mod simple;

#[pymodule]
fn _diffcfd_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(momentum::build_momentum_u, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::build_momentum_v, m)?)?;
    m.add_function(wrap_pyfunction!(pressure::build_pressure_system, m)?)?;
    m.add_function(wrap_pyfunction!(sdf::bspline_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(simple::solve_steady_simple, m)?)?;
    Ok(())
}
