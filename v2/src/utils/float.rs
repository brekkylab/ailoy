pub mod float16 {
    pub fn f16_to_f32(val: u16) -> f32 {
        let sign = (val >> 15) & 0x1;
        let exponent = (val >> 10) & 0x1f;
        let fraction = val & 0x3ff;

        let result_bits = if exponent == 0 {
            // Subnormal 또는 Zero
            if fraction == 0 {
                // Zero
                (sign as u32) << 31
            } else {
                // Subnormal f16을 normal f32로 변환
                let mut exponent_u32 = exponent as u32;
                let mut fraction_u32 = fraction as u32;
                while (fraction_u32 & 0x400) == 0 {
                    fraction_u32 <<= 1;
                    exponent_u32 -= 1;
                }
                exponent_u32 += 1;
                fraction_u32 &= !0x400; // 암시적 1 비트 제거

                let f32_exponent = exponent_u32 + (127 - 15);
                let f32_fraction = fraction_u32 << 13;

                (sign as u32) << 31 | (f32_exponent << 23) | f32_fraction
            }
        } else if exponent == 0x1f {
            // Infinity 또는 NaN
            if fraction == 0 {
                // Infinity
                (sign as u32) << 31 | 0x7f800000
            } else {
                // NaN
                (sign as u32) << 31 | 0x7f800000 | ((fraction as u32) << 13)
            }
        } else {
            // Normalized number
            // 지수 bias 조정 (f16: 15, f32: 127)
            let f32_exponent = (exponent as u32) + (127 - 15);
            // 가수 비트 확장 (f16: 10비트 -> f32: 23비트)
            let f32_fraction = (fraction as u32) << 13;

            (sign as u32) << 31 | (f32_exponent << 23) | f32_fraction
        };

        f32::from_bits(result_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        assert_eq!(float16::f16_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(float16::f16_to_f32(0xc000), -2.0); // -2.0
        assert!(float16::f16_to_f32(0x7c00).is_infinite()); // +inf
        assert!(float16::f16_to_f32(0xfc00).is_infinite()); // -inf
        assert!(float16::f16_to_f32(0x7e00).is_nan()); // NaN
    }
}
