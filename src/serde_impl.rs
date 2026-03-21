//! Serde serialization for numeris types (requires `serde` feature).
//!
//! - `Matrix<T, M, N>` serializes as row-major `[[T; N]; M]` (matching `Matrix::new()`).
//! - `Vector<T, N>` = `Matrix<T, N, 1>` serializes as flat `[T; N]`.
//! - `Quaternion<T>` serializes as `{w, x, y, z}` (via derive).
//! - `DynMatrix<T>` serializes as `{nrows, ncols, data: [[row-major]]}`.
//! - `DynVector<T>` serializes as flat `[T]`.

use crate::Matrix;
use crate::traits::Scalar;

use serde::ser::{Serialize, SerializeTuple, Serializer};
use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};
use core::fmt;
use core::marker::PhantomData;

// ── Matrix<T, M, N>: row-major ─────────────────────────────────────

impl<T: Serialize + Copy, const M: usize, const N: usize> Serialize for Matrix<T, M, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if N == 1 {
            // Column vector: flat [v0, v1, ..., vM-1]
            let mut seq = serializer.serialize_tuple(M)?;
            for i in 0..M {
                seq.serialize_element(&self.data[0][i])?;
            }
            seq.end()
        } else {
            // Matrix: [[row0], [row1], ...]
            let mut rows = serializer.serialize_tuple(M)?;
            for i in 0..M {
                // Serialize each row as a tuple of N elements
                rows.serialize_element(&RowRef::<T, N> {
                    data: core::array::from_fn(|j| self.data[j][i]),
                })?;
            }
            rows.end()
        }
    }
}

/// Helper to serialize a row as a fixed-size sequence.
struct RowRef<T, const N: usize> {
    data: [T; N],
}

impl<T: Serialize, const N: usize> Serialize for RowRef<T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_tuple(N)?;
        for j in 0..N {
            seq.serialize_element(&self.data[j])?;
        }
        seq.end()
    }
}

impl<'de, T: Scalar + Deserialize<'de>, const M: usize, const N: usize> Deserialize<'de>
    for Matrix<T, M, N>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MatVisitor<T, const M: usize, const N: usize>(PhantomData<T>);

        impl<'de, T: Scalar + Deserialize<'de>, const M: usize, const N: usize> Visitor<'de>
            for MatVisitor<T, M, N>
        {
            type Value = Matrix<T, M, N>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if N == 1 {
                    write!(f, "an array of {} elements", M)
                } else {
                    write!(f, "a {}x{} matrix as row-major array", M, N)
                }
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                if N == 1 {
                    // Vector: flat [v0, v1, ..., vM-1]
                    let mut mat = Matrix::<T, M, N>::zeros();
                    for i in 0..M {
                        mat.data[0][i] = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                    }
                    Ok(mat)
                } else {
                    // Matrix: [[row0], [row1], ...]
                    let mut mat = Matrix::<T, M, N>::zeros();
                    for i in 0..M {
                        let row: RowDeserialize<T, N> = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                        for j in 0..N {
                            mat.data[j][i] = row.data[j];
                        }
                    }
                    Ok(mat)
                }
            }
        }

        deserializer.deserialize_tuple(M, MatVisitor::<T, M, N>(PhantomData))
    }
}

/// Helper to deserialize a row as a fixed-size sequence.
struct RowDeserialize<T, const N: usize> {
    data: [T; N],
}

impl<'de, T: Scalar + Deserialize<'de>, const N: usize> Deserialize<'de>
    for RowDeserialize<T, N>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RowVisitor<T, const N: usize>(PhantomData<T>);

        impl<'de, T: Scalar + Deserialize<'de>, const N: usize> Visitor<'de> for RowVisitor<T, N> {
            type Value = RowDeserialize<T, N>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "an array of {} elements", N)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut data = [T::zero(); N];
                for j in 0..N {
                    data[j] = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(j, &self))?;
                }
                Ok(RowDeserialize { data })
            }
        }

        deserializer.deserialize_tuple(N, RowVisitor::<T, N>(PhantomData))
    }
}

// ── DynMatrix / DynVector ──────────────────────────────────────────

#[cfg(feature = "alloc")]
mod dyn_serde {
    use crate::dynmatrix::{DynMatrix, DynVector};
    use crate::traits::Scalar;

    use alloc::format;
    use alloc::vec::Vec;
    use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};
    use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
    use core::fmt;
    use core::marker::PhantomData;

    impl<T: Serialize + Scalar> Serialize for DynMatrix<T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut s = serializer.serialize_struct("DynMatrix", 3)?;
            s.serialize_field("nrows", &self.nrows())?;
            s.serialize_field("ncols", &self.ncols())?;
            // Row-major: build row vecs (column-major storage means no row slices)
            let row_vecs: Vec<Vec<T>> = (0..self.nrows())
                .map(|i| (0..self.ncols()).map(|j| self[(i, j)]).collect())
                .collect();
            s.serialize_field("data", &row_vecs)?;
            s.end()
        }
    }

    impl<'de, T: Scalar + Deserialize<'de>> Deserialize<'de> for DynMatrix<T> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(serde::Deserialize)]
            #[serde(field_identifier, rename_all = "lowercase")]
            enum Field {
                Nrows,
                Ncols,
                Data,
            }

            struct DynMatVisitor<T>(PhantomData<T>);

            impl<'de, T: Scalar + Deserialize<'de>> Visitor<'de> for DynMatVisitor<T> {
                type Value = DynMatrix<T>;

                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "a DynMatrix with nrows, ncols, and row-major data")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let mut nrows: Option<usize> = None;
                    let mut ncols: Option<usize> = None;
                    let mut data: Option<Vec<Vec<T>>> = None;

                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::Nrows => nrows = Some(map.next_value()?),
                            Field::Ncols => ncols = Some(map.next_value()?),
                            Field::Data => data = Some(map.next_value()?),
                        }
                    }

                    let nrows = nrows.ok_or_else(|| de::Error::missing_field("nrows"))?;
                    let ncols = ncols.ok_or_else(|| de::Error::missing_field("ncols"))?;
                    let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                    if data.len() != nrows {
                        return Err(de::Error::custom(format!(
                            "expected {} rows, got {}",
                            nrows,
                            data.len()
                        )));
                    }

                    let flat: Vec<T> = data.into_iter().flat_map(|row| row.into_iter()).collect();

                    if flat.len() != nrows * ncols {
                        return Err(de::Error::custom("row lengths inconsistent with ncols"));
                    }

                    Ok(DynMatrix::from_rows(nrows, ncols, &flat))
                }
            }

            deserializer.deserialize_struct(
                "DynMatrix",
                &["nrows", "ncols", "data"],
                DynMatVisitor::<T>(PhantomData),
            )
        }
    }

    impl<T: Serialize + Scalar> Serialize for DynVector<T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(Some(self.len()))?;
            for i in 0..self.len() {
                seq.serialize_element(&self[i])?;
            }
            seq.end()
        }
    }

    impl<'de, T: Scalar + Deserialize<'de>> Deserialize<'de> for DynVector<T> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let data: Vec<T> = Vec::deserialize(deserializer)?;
            Ok(DynVector::from_vec(data))
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::Matrix;
    use crate::matrix::vector::Vector;

    #[test]
    fn matrix_roundtrip_json() {
        let m = Matrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "[[1.0,2.0],[3.0,4.0]]");
        let m2: Matrix<f64, 2, 2> = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn vector_roundtrip_json() {
        let v = Vector::from_array([1.0_f64, 2.0, 3.0]);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "[1.0,2.0,3.0]");
        let v2: Vector<f64, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn nonsquare_matrix_roundtrip() {
        let m = Matrix::new([[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "[[1.0,2.0,3.0],[4.0,5.0,6.0]]");
        let m2: Matrix<f64, 2, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn integer_matrix_roundtrip() {
        let m = Matrix::new([[1_i32, 2], [3, 4]]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Matrix<i32, 2, 2> = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn scalar_matrix_roundtrip() {
        let m = Matrix::new([[42.0_f64]]);
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "[42.0]");
        let m2: Matrix<f64, 1, 1> = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn quaternion_roundtrip_json() {
        use crate::Quaternion;
        let q = Quaternion::new(1.0_f64, 0.0, 0.0, 0.0);
        let json = serde_json::to_string(&q).unwrap();
        assert!(json.contains("\"w\":1.0"));
        let q2: Quaternion<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(q, q2);
    }

    #[cfg(feature = "alloc")]
    mod dyn_tests {
        use crate::dynmatrix::{DynMatrix, DynVector};

        #[test]
        fn dynmatrix_roundtrip_json() {
            let m = DynMatrix::from_rows(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let json = serde_json::to_string(&m).unwrap();
            assert!(json.contains("\"nrows\":2"));
            assert!(json.contains("\"ncols\":3"));
            let m2: DynMatrix<f64> = serde_json::from_str(&json).unwrap();
            assert_eq!(m, m2);
        }

        #[test]
        fn dynvector_roundtrip_json() {
            let v = DynVector::from_slice(&[1.0_f64, 2.0, 3.0]);
            let json = serde_json::to_string(&v).unwrap();
            assert_eq!(json, "[1.0,2.0,3.0]");
            let v2: DynVector<f64> = serde_json::from_str(&json).unwrap();
            assert_eq!(v, v2);
        }
    }
}
