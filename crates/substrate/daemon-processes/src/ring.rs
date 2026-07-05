// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The rolling per-process output buffer (hermes' 200 KB `output_buffer` window): appends are
//! byte-exact, the front is dropped once the cap is exceeded, and reads render lossily (a cut
//! multi-byte char at the retention edge becomes U+FFFD instead of corrupting the string).

/// A byte ring keeping the last `cap` bytes appended.
pub struct RingBuffer {
    buf: Vec<u8>,
    cap: usize,
}

impl RingBuffer {
    /// An empty ring retaining at most `cap` bytes.
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::new(),
            cap,
        }
    }

    /// Append `chunk`, dropping the oldest bytes once over the cap.
    pub fn append(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
        if self.buf.len() > self.cap {
            let drop = self.buf.len() - self.cap;
            self.buf.drain(..drop);
        }
    }

    /// The whole retained window, lossily decoded.
    pub fn to_lossy_string(&self) -> String {
        String::from_utf8_lossy(&self.buf).into_owned()
    }

    /// The last `n` bytes, lossily decoded (the hermes `output_buffer[-n:]` tails).
    pub fn tail_lossy(&self, n: usize) -> String {
        let start = self.buf.len().saturating_sub(n);
        String::from_utf8_lossy(&self.buf[start..]).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_exactly_the_last_cap_bytes() {
        let mut ring = RingBuffer::new(10);
        for i in 0u8..5 {
            ring.append(&[i; 4]); // 20 bytes total
        }
        // The retained window is the last 10 bytes: bytes 10..20 = [2,2,3,3,3,3,4,4,4,4].
        assert_eq!(
            ring.to_lossy_string().as_bytes(),
            &[2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        );
    }

    #[test]
    fn tail_and_boundary_are_lossy_safe() {
        let mut ring = RingBuffer::new(5);
        ring.append("aé é".as_bytes()); // 6 bytes: a,0xC3,0xA9,0x20,0xC3,0xA9 → front byte dropped
                                        // The leading half of the first 'é' was cut: rendering must not panic.
        let s = ring.to_lossy_string();
        assert!(s.ends_with('é'));
        assert_eq!(ring.tail_lossy(2), "é");
        assert_eq!(ring.tail_lossy(100), s);
    }
}
