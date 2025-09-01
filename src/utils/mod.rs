use crate::prelude::*;

/// Creates a linearly spaced array of numbers.
///
/// # Arguments:
///
/// * `start`: The starting value of the sequence.
/// * `end`: The ending value of the sequence.
/// * `num`: The number of elements in the sequence.
///
/// # Returns:
///
/// A 1-dimensional `DynArray` containing the linearly spaced values.
///
/// # Example
///
/// ```
/// use tiny_matrix::prelude::*;
/// let arr = utils::linspace(0.0, 1.0, 5);
/// assert_eq!(arr, DynArray::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]));
/// ```
pub fn linspace<T>(start: T, end: T, num: usize) -> DynArray<T>
where
    T: ArrayElem + num_traits::Float + num_traits::FromPrimitive,
{
    if num < 2 {
        return DynArray::uniform(start, &[1]);
    }

    let step = (end - start) / T::from_usize(num - 1).unwrap();
    DynArray::from_iter(
        (0..num).map(|i| start + step * T::from_usize(i).unwrap()),
        &[num],
    )
    .unwrap()
}

/// Create a 1-dimensional array spaced evenly logrithmically
///
/// # Arguments:
/// * `start`: log 10 of the  starting value of the sequence.
/// * `end`: log 10 of the ending value of the sequence.
/// * `num`: The number of elements in the sequence.
///
/// # Returns:
///
/// A 1-dimensional `DynArray` containing the spaced values.
///
/// # Example
///
/// ```
/// use tiny_matrix::prelude::*;
/// let arr = utils::logspace(-1.0, 1.0, 3);
/// assert_eq!(arr, DynArray::from_vec(vec![0.1, 1.0, 10.0]));
/// ```
pub fn logspace<T>(start: T, end: T, num: usize) -> DynArray<T>
where
    T: ArrayElem + num_traits::Float + num_traits::FromPrimitive,
{
    let ten = T::from_usize(10).unwrap();
    if num < 2 {
        return DynArray::uniform(ten.powf(start), &[1]);
    }

    let log_start = start;
    let log_end = end;
    let step = (log_end - log_start) / T::from_usize(num - 1).unwrap();
    DynArray::from_iter(
        (0..num).map(|i| ten.powf(log_start + step * T::from_usize(i).unwrap())),
        &[num],
    )
    .unwrap()
}
