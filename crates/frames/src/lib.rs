//! Small time/frame helpers.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seconds(pub f64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Epoch(pub f64);

pub type Frame = InertialFrame;
pub type EpochSeconds = Seconds;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InertialFrame {
    Eci,
    Eme2000,
    Icrf,
}

impl Seconds {
    pub fn value(self) -> f64 {
        self.0
    }
}

impl From<f64> for Seconds {
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<Seconds> for f64 {
    fn from(s: Seconds) -> Self {
        s.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_roundtrip() {
        let s = Seconds::from(123.456);
        let v: f64 = s.into();
        assert_eq!(v, 123.456);
        assert_eq!(InertialFrame::Eci, InertialFrame::Eci);
    }
}
