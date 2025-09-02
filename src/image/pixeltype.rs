pub trait PixelType: Sized + Copy {
    fn channels(&self) -> usize;
}

impl PixelType for u8 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for u16 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for u32 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for u64 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for i8 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for i16 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for i32 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for i64 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for f32 {
    fn channels(&self) -> usize {
        1
    }
}
impl PixelType for f64 {
    fn channels(&self) -> usize {
        1
    }
}
