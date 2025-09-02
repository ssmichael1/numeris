
use super::PixelType;

#[derive(Copy, Clone, Debug)]
pub struct RGB<T> where T: num_traits::Unsigned + num_traits::PrimInt
{
    pub r: T,
    pub g: T,
    pub b: T,
}

impl<T> RGB<T> where T: num_traits::Zero + Copy + num_traits::Unsigned + num_traits::PrimInt + std::fmt::UpperHex
{
    pub fn as_hex_string(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

impl<T> RGB<T> where T: num_traits::Zero + Copy + num_traits::Unsigned + num_traits::PrimInt {
    pub fn is_zero(&self) -> bool {
        self.r.is_zero() && self.g.is_zero() && self.b.is_zero()
    }

    pub fn is_black(&self) -> bool {
        self.r.is_zero() && self.g.is_zero() && self.b.is_zero()
    }

    /// Convert from a hexidecimal string
    ///
    /// # Arguments:
    ///
    /// - `s`: A string slice that represents the hexadecimal color code, starting with '#'
    ///
    /// # Examples
    ///
    /// ```
    /// use numeris::prelude::rgb::*;
    /// let color = RGB8::from_hex_string("#FF5733");
    /// ```
    pub fn from_hex_string(s: &str) -> Option<Self> {
        // Parse hexidecimal "#FFFFFF" format or "#FFF" format
        let s = s.trim();
        if let Some(hex) = s.strip_prefix('#') {
            let len = hex.len();
            let r = if len == 6 {
                u8::from_str_radix(&hex[0..2], 16).ok()?
            } else if len == 3 {
                u8::from_str_radix(&hex[0..1], 16).ok()? * 0x11
            } else {
                return None;
            };
            let g = if len == 6 {
                u8::from_str_radix(&hex[2..4], 16).ok()?
            } else if len == 3 {
                u8::from_str_radix(&hex[1..2], 16).ok()? * 0x11
            } else {
                return None;
            };
            let b = if len == 6 {
                u8::from_str_radix(&hex[4..6], 16).ok()?
            } else if len == 3 {
                u8::from_str_radix(&hex[2..3], 16).ok()? * 0x11
            } else {
                return None;
            };
            Some(RGB { r: T::from(r).unwrap(), g: T::from(g).unwrap(), b: T::from(b).unwrap() })
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RGBA<T> where T: num_traits::Unsigned + num_traits::PrimInt
{
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
}

impl<T> PixelType for RGB<T> where T: num_traits::Unsigned + num_traits::PrimInt {
    fn channels(&self) -> usize {
        3
    }
}

impl<T> PixelType for RGBA<T> where T: num_traits::Unsigned + num_traits::PrimInt {
    fn channels(&self) -> usize {
        4
    }
}

impl<T> Default for RGB<T> where T: num_traits::Zero + Copy + num_traits::Unsigned + num_traits::PrimInt{
    fn default() -> Self {
        RGB { r: T::zero(), g: T::zero(), b: T::zero() }
    }
}

impl<T> Default for RGBA<T> where T: num_traits::Zero + Copy + num_traits::Unsigned + num_traits::PrimInt {
    fn default() -> Self {
        RGBA { r: T::zero(), g: T::zero(), b: T::zero(), a: T::zero() }
    }
}

pub type RGB8 = RGB<u8>;
pub type RGBA8 = RGBA<u8>;
pub type RGB16 = RGB<u16>;
pub type RGBA16 = RGBA<u16>;