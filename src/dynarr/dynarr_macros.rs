#[macro_export]
macro_rules! dynarray {
    ($data:expr, $shape:expr) => {{
        let data = $data;
        let shape = $shape;
        DynArray::from_vec(data, shape)
    }};
}
